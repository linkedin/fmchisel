import logging
from typing import List, Optional, Union

import lightning.pytorch as pl
import torch

from fmchisel.optimizers.adamw_schedulefree import AdamWScheduleFree
from fmchisel.utils.train_utils import load_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class LanguageModel(pl.LightningModule):
    """
    A PyTorch Lightning module for demonstration and testing of advanced PyTorch features,
    Gradient Checkpointing and CPU Offloading.

    Args:
        model_path (str): Path to the pre-trained model to be fine-tuned.
        max_lr (float): Maximum learning rate for the optimizer.
        weight_decay (float): Weight decay for regularization in the optimizer.
        warmup_ratio (float): The ratio of total steps used for learning rate warmup.
        enable_cpu_offload (bool, optional): If True, offloads model parameters to CPU to
            reduce GPU memory usage. Defaults to False.
        enable_gradient_checkpointing (bool, optional): If True, enables gradient checkpointing
            to reduce memory usage during backpropagation at the cost of increased compute time.
            Defaults to True.

    Example:
        >>> model = LanguageModel(
        >>>     model_path="/path/to/model",
        >>>     max_lr=4e-5,
        >>>     weight_decay=0.01,
        >>>     warmup_ratio=0.1,
        >>> )
        >>> trainer = pl.Trainer(max_epochs=1, precision="bf16")
        >>> trainer.fit(model, train_dataloader)

    """

    def __init__(
        self,
        model_path: str,
        max_lr: float,
        weight_decay: float,
        warmup_ratio: float,
        optimizer: str,
        use_liger: bool = False,
        enable_cpu_offload: bool = False,
        enable_gradient_checkpointing: bool = True,
        use_lora: bool = False,
        lora_rank: Optional[int] = None,
        lora_target_modules: Optional[Union[List[str], str]] = None,
        lora_alpha_to_rank_ratio: Optional[float] = None,
    ):
        super().__init__()
        self.model_path = model_path
        self.max_lr = max_lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.model = None
        self.optimizer = optimizer
        self.num_correct = 0
        self.num_total = 0
        self.enable_gradient_checkpointing = enable_gradient_checkpointing
        self.enable_cpu_offload = enable_cpu_offload
        self.use_liger = use_liger
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_target_modules = lora_target_modules
        self.lora_alpha_to_rank_ratio = lora_alpha_to_rank_ratio

    def log_model_stage(self, stage: str):
        """
        Logs the current state of the model with a description of the stage.

        Args:
            stage (str): Description of the current model stage.
        """
        log.warning(f"Model at stage: {stage}\n{self.model}")

    def configure_model(self):

        # https://lightning.ai/docs/pytorch/stable/advanced/model_parallel/fsdp.html#speed-up-model-initialization
        if self.model is not None:
            return
        self.model = load_model(
            trainer_precision=self.trainer.precision,
            model_path=self.model_path,
            low_cpu_mem_usage=True,
            use_liger=self.use_liger,
            use_lora=self.use_lora,
            lora_rank=self.lora_rank,
            lora_target_modules=self.lora_target_modules,
            lora_alpha_to_rank_ratio=self.lora_alpha_to_rank_ratio,
        )
        self.model.train()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    def training_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = outputs.loss
        self.log_dict(
            {"train_loss": loss},
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=False,
        )
        return loss

    def on_train_batch_start(self, batch, batch_idx):
        super().on_train_batch_start(batch, batch_idx)
        self.hand_roll_base_zero_grad()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        self.hand_roll_base_zero_grad()

    def on_before_optimizer_step(self, optimizer):
        super().on_before_optimizer_step(optimizer)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_epoch_end(self) -> None:
        val_acc = self.num_correct / self.num_total
        self.log(
            "val_accuracy_epoch",
            val_acc,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )

        self.num_correct = 0
        self.num_total = 0

    def get_correct_eval_num(self, batch, outputs):
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = batch["labels"][..., 1:].contiguous()
        mask = shift_labels != -100
        correct = (shift_logits.argmax(dim=-1) == shift_labels) & mask
        self.num_correct += correct.sum().item()
        self.num_total += mask.sum().item()

    def validation_step(self, batch):
        outputs = self.model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.get_correct_eval_num(batch, outputs)
        self.log_dict(
            {"val_loss": outputs.loss},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            rank_zero_only=True,
            sync_dist=True,
        )
        return outputs.loss

    def get_parameters(self):
        print("Get parameters before set the optimizer")
        return self.parameters()

    def configure_optimizers(self):
        if self.optimizer == "adamw_schedulefree":
            return AdamWScheduleFree(
                self.get_parameters(),
                lr=self.max_lr,
                weight_decay=self.weight_decay,
                warmup_steps=self.trainer.estimated_stepping_batches * self.warmup_ratio,
            )
        optimizer = torch.optim.AdamW(
            self.get_parameters(),
            lr=self.max_lr,
            weight_decay=self.weight_decay,
            fused=True,
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
