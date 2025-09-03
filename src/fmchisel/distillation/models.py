import functools
import random
from collections import OrderedDict
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, Callable, Iterator

import lightning.pytorch as pl
import torch
from lightning.fabric.strategies.fsdp import (
    _move_torchmetrics_to_device,
    _setup_activation_checkpointing,
)
from lightning.pytorch.strategies.fsdp import FSDPStrategy
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from transformers.generation.utils import GenerationConfig

import fmchisel
from fmchisel.distillation.losses import DISTILL_LOSS_MAP
from fmchisel.optimizers.adamw_schedulefree import AdamWScheduleFree
from fmchisel.utils.train_utils import load_model

_TEACHER_MODEL_NAME_KEY = "teacher_model"


class _FSDPForwardRedirection:
    """
    Modified based on
    https://github.com/Lightning-AI/pytorch-lightning/blob/d3f9c83d6efa4f1def36aa6c199600946cdb9117/src/lightning/pytorch/strategies/strategy.py#L601-L648

    Redirect a method call through FullyShardedDataParallel.forward so that the FSDP module's root pre-forward and
    post-forward can be properly executed around the method call.

    This is needed in cases where we call a submodule of a FSDP module. For instance, when we want to call only
    the `LlamaModel` part out of a FSDP-wrapped `LlamaForCausalLM` to get the hidden states without involving
    GPU-memory-heavy `lm_head` and cross entropy computation, doing this directly (i.e. `model.model.forward()`)
    will not work because the first `nn.Emebedding` layer is not independently wrapped as a FSDP module (because of
    the transformer-based wrapping policy), and not calling it through FSDP root module forward will not all-gather
    its parameter, thus resulting in "RuntimeError: 'weight' must be 2-D" error. Similarly, if we want to call just
    the `lm_head` part of a model, we need this trick too to properly get its params all-gathered.
    """

    def __call__(self, wrapper_module: FullyShardedDataParallel, method: Callable, *args: Any, **kwargs: Any):
        """Reroutes a method call through the `wrapper_module`'s `forward` method.

        Args:
            wrapper_module: The FSDP root module whose `forward`, pre-forward and post-forward hooks we want
                to use to wrap the method call.
            method: The function that should be called after inputs get redirected through the
                `wrapper_module`'s `forward` method.
            *args: The positional arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.
            **kwargs: The keyword arguments to the method `method_name`. They will get passed to a patched
                `forward` method instead.

        """
        assert isinstance(wrapper_module, FullyShardedDataParallel)
        original_module = wrapper_module._fsdp_wrapped_module
        original_forward = original_module.forward

        def wrapped_forward(*_args: Any, **_kwargs: Any) -> Any:
            # Unpatch ourselves immediately before calling the method `method_name`
            # because itself may want to call the real `forward`
            original_module.forward = original_forward  # type: ignore[method-assign]
            # Call the actual method e.g. `.training_step(...)`
            out = method(*_args, **_kwargs)
            return out

        # Patch the original_module's forward so we can redirect the arguments back to the real method
        original_module.forward = wrapped_forward  # type: ignore[method-assign]

        wrapper_output = wrapper_module(*args, **kwargs)
        return wrapper_output


def forward_redirect(student_module, teacher_module, method, *args, **kwargs):
    """
    Call `method` under both `student_module` and `teacher_module` FSDP instances'
    root pre/post-forward.
    """
    first_layer = functools.partial(_FSDPForwardRedirection(), wrapper_module=student_module, method=method)
    return _FSDPForwardRedirection()(teacher_module, first_layer, *args, **kwargs)


class DistillLanguageModel(pl.LightningModule):
    """
    A PyTorch Lightning module for knowledge distillation training of language models.

    This class implements a comprehensive knowledge distillation framework that supports
    various distillation strategies. It uses FSDP for efficient distributed training.

    Example:
        ```python
        config = DistillTrainingConfig(
            enable_distill=True,
            teacher_model_path="/path/to/teacher_model",
            temperature=1.0,
            distillation_loss_ratio=0.9,
            distill_loss="forward_kl"
        )

        model = DistillLanguageModel(
            model_path="/path/to/student_model",
            max_lr=1e-4,
            weight_decay=0.01,
            warmup_ratio=0.1,
            optimizer="adamw",
            distill_training_config=config
        )
        ```
    """

    def __init__(
        self,
        model_path: str,
        max_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        optimizer: str,
        distill_training_config: "fmchisel.distillation.DistillTrainingConfig",
        tokenizer=None,
        ignore_index=-100,
        use_liger=False,
    ):
        super().__init__()
        self.model_path = model_path
        self.distill_training_config = distill_training_config
        self.model = None
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.teacher_model = None
        self.distill_loss_fn = DISTILL_LOSS_MAP[self.distill_training_config.distill_loss](
            **self.distill_training_config.distill_loss_kwargs
        )
        if self.distill_training_config.compile_distill_loss:
            self.distill_loss_fn = torch.compile(self.distill_loss_fn)
        self.sample_fraction = self.distill_training_config.sample_fraction
        self.include_prompt_loss = self.distill_training_config.include_prompt_loss
        self.generation_config = GenerationConfig(
            max_new_tokens=self.distill_training_config.max_new_tokens,
            temperature=self.distill_training_config.sample_temperature,
            do_sample=True,
            top_k=0,
            use_cache=True,
        )
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.use_liger = use_liger
        self.optimizer = optimizer

    def configure_model(self):
        """
        Configure and initialize both student and teacher models.
        """
        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        strategy = self.trainer.strategy
        assert isinstance(strategy, FSDPStrategy), "Only FSDP is supported for distillation training."
        # Right now we limit the support to FSDP for simplicity. For distillation training,
        # We need to manually wrap teacher and student separately with FSDP, otherwise if
        # leveraging pl trainer, it'll treat the entire `DistillLanguageModel` as a single model
        # and wrap it with FSDP, which causes various issues.
        # TODO(@yudai): do we need to support DeepSpeed as well?
        if self.model is None:
            student_model = load_model(
                trainer_precision=self.trainer.precision,
                model_path=self.model_path,
                low_cpu_mem_usage=isinstance(self.trainer.strategy, FSDPStrategy),
                use_liger=self.use_liger,
            )
            self.model = FullyShardedDataParallel(
                module=student_model,
                cpu_offload=strategy.cpu_offload,
                mixed_precision=strategy.mixed_precision_config,
                sharding_strategy=strategy.sharding_strategy,
                device_id=strategy.root_device.index,
                **strategy.kwargs,
            )

        if self.teacher_model is None:
            teacher_model = load_model(
                # force teacher weight to be bf16 because we are not training it
                trainer_precision="bf16-true",
                model_path=self.distill_training_config.teacher_model_path,
                low_cpu_mem_usage=isinstance(self.trainer.strategy, FSDPStrategy),
                use_liger=self.use_liger,
            )
            # Allow teacher to belong to a different class than student.
            # This is to allow rare case where teacher and student are of different model class
            # but they actually share the same vocab and tokenizer (for example qwen3 & qwen2.5)
            # so that KD can still be applied.
            teacher_decoder_layer_cls = teacher_model.model.layers[0].__class__
            teacher_wrap_kwargs = deepcopy(strategy.kwargs)
            teacher_wrap_kwargs["auto_wrap_policy"] = ModuleWrapPolicy({teacher_decoder_layer_cls})

            self.teacher_model = FullyShardedDataParallel(
                module=teacher_model,
                cpu_offload=strategy.cpu_offload,
                sharding_strategy=strategy.sharding_strategy,
                # force teacher weight to be bf16 because we are not training it
                mixed_precision=MixedPrecision(param_dtype=torch.bfloat16),
                device_id=strategy.root_device.index,
                **teacher_wrap_kwargs,
            )
        self.teacher_model.eval()

        # [ Why this seems unnecessary but we need it ]
        # A side effect of redirecting `training_step` through teacher FSDP root module's
        # forward (see docstring of `_FSDPForwardRedirection` for why) is that teacher FSDP
        # root module's backward will be unproperly called even though `no_grad()` is set
        # when getting teacher hidden state output. FSDP post-backward hook assumes all
        # trainable FlatParam's grad to be saved but this is not gonna be the case with teacher
        # model. Thus we have to explicitly set teacher flat params' `requires_grad` to False
        # to let FSDP not to add this post-backward hook.
        for p in self.teacher_model.parameters():
            p.requires_grad = False

        _move_torchmetrics_to_device(self.model, strategy.root_device)
        _setup_activation_checkpointing(self.model, strategy._activation_checkpointing_kwargs)

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        """
        Forward pass through the student model.

        Args:
            input_ids (torch.Tensor): Input token IDs
            attention_mask (torch.Tensor): Attention mask for the input
            labels (torch.Tensor, optional): Target labels for loss computation
            **kwargs: Additional arguments passed to the model's forward method

        Returns:
            The output from the student model's forward pass
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def configure_optimizers(self):
        """
        Configure optimizers and learning rate schedulers for training.

        This method sets up the optimizer and learning rate scheduler based on the
        specified optimizer type. It supports both AdamW and AdamW Schedule-Free optimizers.
        """
        if self.optimizer == "adamw_schedulefree":
            # AdamW-SF doesn't need LR decay so directly return the optimizer without LR scheduler
            return AdamWScheduleFree(
                self.model.parameters(),
                lr=self.max_lr,
                weight_decay=self.weight_decay,
                warmup_steps=int(self.trainer.estimated_stepping_batches * self.warmup_ratio),
            )
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.max_lr,
            weight_decay=self.weight_decay,
        )
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=self.warmup_ratio,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": lr_scheduler, "interval": "step"},
        }

    def compute_distillation_loss(self, batch, logits, teacher_logits, **kwargs):
        """
        Compute the distillation loss between student and teacher model outputs.

        Args:
            batch (Dict): Batch containing input data and labels
            logits (torch.Tensor): Logits from the student model
            teacher_logits (torch.Tensor): Logits from the teacher model

        Returns:
            torch.Tensor: The computed distillation loss value
        """
        return self.distill_loss_fn(
            batch["labels"],
            logits,
            teacher_logits,
            temperature=self.distill_training_config.temperature,
            student_model_lm_head_weight=self.model.lm_head.weight if self.use_liger else None,
            teacher_model_lm_head_weight=self.teacher_model.lm_head.weight if self.use_liger else None,
            compute_hard_loss=kwargs.get("compute_hard_loss", False),
            is_eval=kwargs.get("is_eval", False),
        )

    def exclude_teacher_model_optimizer_parameters(
        self, name_key: str = _TEACHER_MODEL_NAME_KEY, recurse: bool = True
    ) -> Iterator[nn.Parameter]:
        """
        Yield parameters that should be optimized, excluding teacher model parameters.
        """
        for name, param in self.named_parameters(recurse=recurse):
            if not name.startswith(name_key):
                yield param

    def get_parameters(self):
        print("Exclude teacher model parameters before set the optimizer")
        return self.exclude_teacher_model_optimizer_parameters()

    # Refering to the huggingface GKD trainer `generate_on_policy_outputs` function
    @torch.no_grad()
    def _sample_single_response(self, batch, model, generation_config, pad_token_id=None):
        """
        This method generates responses using the provided model and generation configuration.
        It handles the generation process, manages attention masks, and processes padding tokens
        according to the distillation configuration.

        Args:
            batch (Dict): Batch containing prompt input IDs and attention masks
            model: The model to use for generation (student or teacher)
            generation_config: Configuration for text generation
            pad_token_id (int, optional): ID of the padding token

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Generated tokens, attention mask, and labels
        """
        # set it true only when doing generation because for gradient checkpointing during
        # training to work, it has to be False
        model.config.use_cache = True

        generated_outputs = model.generate(
            input_ids=batch["prompt_input_ids"],
            attention_mask=batch["prompt_attention_mask"],
            generation_config=generation_config,
            return_dict_in_generate=True,
            synced_gpus=True,  # True is required to use differently sized data with FSDP + generate (current default is False)
        )

        # Get the generated token IDs
        generated_tokens = generated_outputs.sequences
        # Calculate new attention mask
        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        # If there's pad_token_id, set attention mask to 0 for padding tokens
        if pad_token_id is not None:
            if self.include_prompt_loss:
                # keep the prompt tokens as well in the distillation loss
                new_labels[new_labels == pad_token_id] = self.ignore_index
            else:
                # only compute distillation loss for the generated tokens
                per_batch_max_prompt_token_length = batch["prompt_input_ids"].shape[1]
                # mask left prompt token
                new_labels[:, :per_batch_max_prompt_token_length] = self.ignore_index
                # mask right generation padding token
                new_labels[:, per_batch_max_prompt_token_length:][
                    new_labels[:, per_batch_max_prompt_token_length:] == pad_token_id
                ] = self.ignore_index
            new_attention_mask[generated_tokens == pad_token_id] = 0

        model.config.use_cache = False
        return generated_tokens, new_attention_mask, new_labels

    def _get_kd_batch(self, batch):
        """
        This method determines whether to use the original batch or generate new samples
        based on the sampling strategy and fraction. It supports three sampling methods:

        - "supervised": Always uses the original batch (ground truth labels)
        - "on-policy": Generates responses using the student model
        - "sequence-level": Generates responses using the teacher model

        The method uses the sample_fraction to determine the probability of generating
        new samples vs using the original batch.

        Args:
            batch (Dict): Original batch containing input data

        Returns:
            Tuple[Dict, bool]: Modified batch and indicator for whether new data was generated
        """
        if self.distill_training_config.sample_method == "supervised" or random.random() > self.sample_fraction:
            return batch, False  # indicator for returning old data
        if self.distill_training_config.sample_method == "on-policy":
            model_to_generate = self.model
        elif self.distill_training_config.sample_method == "sequence-level":
            model_to_generate = self.teacher_model
        generated_tokens, new_attention_mask, new_labels = self._sample_single_response(
            batch,
            model_to_generate,
            self.generation_config,
            pad_token_id=(None if self.tokenizer is None else self.tokenizer.pad_token_id),
        )
        return {
            "input_ids": generated_tokens,
            "attention_mask": new_attention_mask,
            "labels": new_labels,
        }, True  # indicator for generating new data

    def _get_last_hidden_state(self, model, batch):
        """
        Get the last hidden state from a model.
        """
        return model.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        ).last_hidden_state

    def training_step(self, batch, batch_idx):
        """
        Perform a single training step with knowledge distillation.

        Args:
            batch: Training batch data
            batch_idx: Index of the current batch

        Returns:
            torch.Tensor: Total training loss
        """
        return forward_redirect(
            self.model,
            self.teacher_model,
            self._training_step,
            batch=batch,
            batch_idx=batch_idx,
        )

    def _compute_flce(self, lm_head_weight, batch, hidden_states):
        """
        Compute fused linear cross-entropy loss using Liger.

        Args:
            lm_head_weight (torch.Tensor): Weight matrix of the language modeling head
            batch (Dict): Batch containing labels
            hidden_states (torch.Tensor): Hidden states from the model

        Returns:
            SimpleNamespace: Object containing the computed loss
        """
        from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

        shifted_hidden_states = hidden_states[:, :-1, :].contiguous()
        shifted_labels = batch["labels"][:, 1:].contiguous()

        # flatten tokens
        shifted_hidden_states = shifted_hidden_states.view(-1, shifted_hidden_states.size(-1))
        shifted_labels = shifted_labels.view(-1)

        loss = LigerFusedLinearCrossEntropyLoss()(lm_head_weight, shifted_hidden_states, shifted_labels)
        return SimpleNamespace(loss=loss)

    def _training_step(self, batch, batch_idx):
        """
        Perform the actual training step with knowledge distillation.
        """
        kd_batch, is_new_data = self._get_kd_batch(batch)

        if self.use_liger:
            with torch.no_grad():
                teacher_hidden_states = self._get_last_hidden_state(self.teacher_model, kd_batch)
            student_hidden_states = self._get_last_hidden_state(self.model, kd_batch)

            student_outputs = self._compute_flce(
                lm_head_weight=self.model.lm_head.weight,
                batch=kd_batch,
                hidden_states=student_hidden_states,
            )
            distillation_loss = self.compute_distillation_loss(kd_batch, student_hidden_states, teacher_hidden_states)
        else:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=kd_batch["input_ids"],
                    attention_mask=kd_batch["attention_mask"],
                )

            student_outputs = self.model(
                input_ids=kd_batch["input_ids"],
                attention_mask=kd_batch["attention_mask"],
                labels=kd_batch["labels"],
            )
            distillation_loss = self.compute_distillation_loss(kd_batch, student_outputs.logits, teacher_outputs.logits)
        # get NLL loss
        outputs = (
            student_outputs
            if not is_new_data
            else self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
        )
        metrics = {"train_loss": outputs.loss}
        total_loss = (
            outputs.loss * (1 - self.distill_training_config.distillation_loss_ratio)
            + distillation_loss * self.distill_training_config.distillation_loss_ratio
        )
        metrics.update({"distillation_loss": distillation_loss, "total_train_loss": total_loss})
        self.log_dict(
            metrics,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=False,
        )
        return total_loss

    def validation_step(self, batch):
        """
        Perform a validation step with optional distillation loss computation.

        Args:
            batch: Validation batch data

        Returns:
            torch.Tensor: Validation loss (language modeling or combined)
        """
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        logits = outputs.logits
        returned_loss = lm_loss = outputs.loss
        metrics = {"val_loss": lm_loss}
        # only including distillation loss when not doing sampling to reduce time
        if (
            self.distill_training_config.val_include_distill_loss
            and self.distill_training_config.sample_method == "supervised"
        ):
            with torch.no_grad():
                teacher_outputs = self.teacher_model(
                    input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
                )
            distillation_loss = self.compute_distillation_loss(batch, logits, teacher_outputs.logits, is_eval=True)
            total_loss = (
                lm_loss * (1 - self.distill_training_config.distillation_loss_ratio)
                + distillation_loss * self.distill_training_config.distillation_loss_ratio
            )
            metrics.update({"distillation_loss": distillation_loss, "total_train_loss": total_loss})
            returned_loss = total_loss
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            rank_zero_only=True,
        )
        return returned_loss

    # Below state dict related changes are needed for excluding teacher
    # from the intermediate checkpoint to cut down the disk space and save
    # time at each evaluation from saving the teacher state dict
    def state_dict(self, *args, **kwargs):
        result = OrderedDict()
        for k, v in super().state_dict(*args, **kwargs).items():
            if not k.startswith(_TEACHER_MODEL_NAME_KEY):
                result[k] = v
        return result

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Load the state dictionary with non-strict mode.

        Args:
            state_dict: State dictionary to load
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        kwargs["strict"] = False
        super().load_state_dict(state_dict, *args, **kwargs)
