import os
from datetime import timedelta

import lightning.pytorch as pl
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.strategies import FSDPStrategy
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision
from transformers import AutoTokenizer, HfArgumentParser

from fmchisel.config import DataLoadingConfig, TrainingArgs
from fmchisel.distillation import DistillLanguageModel, DistillTrainingConfig
from fmchisel.models import LanguageModel
from fmchisel.utils import (
    GradientClippingCallback,
    SparseTrainingCallback,
    consolidate_ckpt_to_hf,
    get_training_logger,
    get_wrapping_policy,
)


def get_dataset_module(tokenizer: AutoTokenizer, data_config: DataLoadingConfig):
    if data_config.dataset == "cnn_dailymail":
        from fmchisel.data.datasets import CNNModule

        data_module = CNNModule(
            tokenizer=tokenizer,
            data_load_config=data_config,
        )
    else:
        raise ValueError("Unkown dataset.")
    return data_module


def get_language_model(
    model_training_config: TrainingArgs,
    distill_training_config: DistillTrainingConfig,
):
    kwargs = {
        "model_path": model_training_config.model_path,
        "max_lr": model_training_config.lr,
        "weight_decay": model_training_config.weight_decay,
        "warmup_ratio": model_training_config.warmup_ratio,
        "use_liger": model_training_config.use_liger,
        "optimizer": model_training_config.optimizer,
    }
    if not distill_training_config.enable_distill:
        kwargs.update(
            {
                "use_lora": model_training_config.use_lora,
                "lora_rank": model_training_config.lora_rank,
                "lora_target_modules": model_training_config.lora_target_modules,
                "lora_alpha_to_rank_ratio": model_training_config.lora_alpha_to_rank_ratio,
            }
        )

    if distill_training_config.enable_distill:
        model = DistillLanguageModel(
            **kwargs,
            distill_training_config=distill_training_config,
        )
        model.generation_config.do_sample = False
        return model
    else:
        kwargs.update(
            {
                "enable_cpu_offload": model_training_config.cpu_offload,
                "enable_gradient_checkpointing": model_training_config.enable_gradient_checkpointing,
            }
        )
        return LanguageModel(**kwargs)


def convert_checkpoints_to_hf(model_training_config, output_path, best_model_path=None):
    dist.barrier()
    if dist.get_rank() == 0:
        consolidate_ckpt_to_hf(
            best_model_path,
            model_training_config.model_path,
            output_path=output_path,
            exclude_modules=["teacher"],
            remove_original_ckpt=True,
            is_ckpt_sharded=False,
            use_lora=model_training_config.use_lora,
            lora_rank=model_training_config.lora_rank,
            lora_target_modules=model_training_config.lora_target_modules,
            lora_alpha_to_rank_ratio=model_training_config.lora_alpha_to_rank_ratio,
            verify_lora_saving_correctness=model_training_config.verify_lora_saving_correctness,
        )


def train():
    pl.seed_everything(42)
    parser = HfArgumentParser((TrainingArgs, DataLoadingConfig, DistillTrainingConfig))
    (model_training_config, data_config, distill_config, _) = parser.parse_args_into_dataclasses(
        return_remaining_strings=True
    )
    if (
        distill_config.enable_distill
        and not data_config.return_prompt_input_ids
        and distill_config.sample_method in ["on-policy", "sequence-level"]
    ):
        raise ValueError("For using sampling based distillation loss, return_prompt_input_ids must be True.")
    if not os.path.exists(model_training_config.output_dir):
        os.makedirs(model_training_config.output_dir)

    wrap_policy = get_wrapping_policy(model_training_config.model_path, use_lora=model_training_config.use_lora)
    fsdp_strategy = FSDPStrategy(
        auto_wrap_policy=wrap_policy,
        sharding_strategy="FULL_SHARD",
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        sync_module_states=True,
        activation_checkpointing_policy=(wrap_policy if model_training_config.enable_gradient_checkpointing else None),
        # for FSDP, we set mixed precision here instead of passing precision to PL trainer.
        # precision="bf16-true" in PL trainer means pure half precision (including optimizer update etc.)
        # while precision="bf16-mixed" results in unshard allgather performed in fp32:
        # https://github.com/Lightning-AI/pytorch-lightning/blob/bf25167bbf64f50ba335aa759318946b21775cd2/src/lightning/fabric/plugins/precision/fsdp.py#L83
        mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16),
        cpu_offload=model_training_config.cpu_offload,
    )

    fsdp_strategy._timeout = timedelta(seconds=7200)
    callbacks = [GradientClippingCallback(clip_val=1.0, log_grad_norm=True)]

    if model_training_config.keep_sparse:
        callbacks.append(SparseTrainingCallback())

    if model_training_config.save_on_best_validation:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(model_training_config.output_dir, "ckpts"),
            save_top_k=1,
            monitor="val_loss",
        )
        callbacks.append(checkpoint_callback)

    logger_list = [
        get_training_logger(run_name=model_training_config.output_dir),
        CSVLogger(model_training_config.output_dir, flush_logs_every_n_steps=10),
    ]
    # Remove None loggers (e.g., if MLflow is not enabled)
    logger = [l for l in logger_list if l is not None]  # noqa: E741

    trainer = pl.Trainer(
        accelerator="cuda",
        strategy=fsdp_strategy,
        devices=torch.cuda.device_count(),
        enable_checkpointing=True,
        default_root_dir=model_training_config.output_dir,
        log_every_n_steps=1,
        max_epochs=model_training_config.num_epoch,
        logger=logger,
        callbacks=callbacks,
        val_check_interval=model_training_config.val_check_interval,
    )
    # Only log hyperparams if MLflow logger is enabled
    if any(l.__class__.__name__ == "MLFlowLogger" for l in logger):  # noqa: E741
        trainer.logger.log_hyperparams(
            {
                **vars(model_training_config),
                **vars(data_config),
                **vars(distill_config),
            }
        )

    tokenizer = AutoTokenizer.from_pretrained(
        model_training_config.model_path, padding_side="left", truncation_side="left"
    )
    tokenizer.pad_token = tokenizer.eos_token
    data_module = get_dataset_module(tokenizer, data_config)
    model = get_language_model(model_training_config, distill_config)
    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(f"{model_training_config.output_dir}/ckpts/model.ckpt")
    # Check sparsity for Language model
    if model_training_config.keep_sparse:  # Sparsity sanity check
        dist.barrier()
        if dist.get_rank() == 0:
            from fmchisel.utils.train_utils import check_sparsity

            if not check_sparsity(f"{model_training_config.output_dir}/ckpts/model.ckpt"):
                raise ValueError("Sparsity sanity check failed.")
    # If save model checkpoint on the best val accuracy, convert the checkpoint to HF
    if model_training_config.save_on_best_validation:
        final_model_path = os.path.join(model_training_config.output_dir, "best_model")
        convert_checkpoints_to_hf(model_training_config, final_model_path, checkpoint_callback.best_model_path)

    trainer.print(torch.cuda.memory_summary())


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train()
