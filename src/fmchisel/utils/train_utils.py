import fnmatch
import functools
import logging
import os
import shutil
from typing import List, Optional, Union

import torch
import torch.nn as nn
from lightning.pytorch.loggers import Logger, MLFlowLogger
from packaging import version
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFIX_TO_REMOVE_FROM_STATE_DICT = "model."
SUBSTRING_TO_REMOVE_FROM_STATE_DICT = "_orig_mod."

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger(__name__)


class ModelFamily:
    LLAMA = "llama"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"
    QWEN2 = "qwen2"
    R1_QWEN = "r1-distill-qwen"
    QWEN3 = "qwen3"


def get_wrapping_policy(model_name: str, use_lora: bool = False):
    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
    from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
    from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer
    from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer

    if ModelFamily.LLAMA in model_name.lower():
        transformer_layer = LlamaDecoderLayer
    elif ModelFamily.MIXTRAL in model_name.lower():
        transformer_layer = MixtralDecoderLayer
    elif ModelFamily.MISTRAL in model_name.lower():
        transformer_layer = MistralDecoderLayer
    elif ModelFamily.QWEN2 in model_name.lower():
        transformer_layer = Qwen2DecoderLayer
    elif ModelFamily.QWEN3 in model_name.lower():
        transformer_layer = Qwen3DecoderLayer
    elif ModelFamily.R1_QWEN in model_name.lower():
        transformer_layer = Qwen2DecoderLayer
    else:
        raise ValueError(f"Model {model_name} not supported for FSDP wrapping")
    if not use_lora:
        return {transformer_layer}
    return fsdp_auto_wrap_policy_for_lora(transformer_layer)


# code inspired by https://github.com/AnswerDotAI/fsdp_qlora/blob/05ed9f2a60f96a0795cb082bceab70a9b19fd213/train.py#L468
def fsdp_auto_wrap_policy_for_lora(transformer_layer: torch.nn.Module):

    from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
    from transformers.models.mistral.modeling_mistral import (
        MistralAttention,
        MistralMLP,
    )
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralAttention,
        MixtralSparseMoeBlock,
    )
    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2MLP
    from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention, Qwen3MLP

    def lambda_policy_fn(module):
        # LoRA and DoRA trainable layers.
        return isinstance(module, nn.Sequential) and all(m.weight.requires_grad for m in module)

    def self_attn_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaAttention, MistralAttention, MixtralAttention, Qwen2Attention, Qwen3Attention))

    def mlp_policy_fn(module):
        # Check module name is self_attn.
        return isinstance(module, (LlamaMLP, MistralMLP, MixtralSparseMoeBlock, Qwen2MLP, Qwen3MLP))

    lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
    self_attn_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=self_attn_policy_fn)
    mlp_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=mlp_policy_fn)
    transformer_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer},
    )
    policies = [lambda_policy, transformer_wrap_policy]
    policies.extend([self_attn_policy, mlp_policy])
    return functools.partial(_or_policy, policies=policies)


def is_flash_attn_available() -> bool:
    try:
        from flash_attn import flash_attn_func, flash_attn_qkvpacked_func  # noqa: F401
    except Exception:
        return False
    return True


def get_training_logger(run_name) -> Logger:
    if "MLFLOW_EXPERIMENT_NAME" in os.environ and "MLFLOW_TRACKING_URI" in os.environ:
        training_logger = MLFlowLogger(
            experiment_name=os.environ["MLFLOW_EXPERIMENT_NAME"],
            run_name=run_name,
            tracking_uri=os.environ["MLFLOW_TRACKING_URI"],
        )
        return training_logger
    else:
        print("MLflow environment variables not set. Skipping MLflow logging.")
        return None  # Or return a dummy logger if your framework requires it


def check_sparsity(model_path, sparsity_threshold=0.5) -> bool:

    state_dict = torch.load(model_path)["state_dict"]
    ok_layers = 0
    tot_layers = 0
    for key, value in state_dict.items():
        if "proj" in key and "weight" in key and "teacher" not in key:
            tot_layers += 1
            sparsity = 1 - (torch.abs(value) > 0).sum() / value.shape[0] / value.shape[1]
            if sparsity >= sparsity_threshold:
                ok_layers += 1
    print(f"{ok_layers} out of {tot_layers} layers are sparse.")
    if ok_layers == tot_layers:
        return True
    return False


def load_model(
    trainer_precision: str,
    model_path: str,
    low_cpu_mem_usage: bool,
    attn_implementation: Optional[str] = None,
    use_cache: bool = False,
    use_liger: bool = False,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_target_modules: Optional[Union[List[str], str]] = None,
    lora_alpha_to_rank_ratio: Optional[float] = None,
) -> AutoModelForCausalLM:
    from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT_STR

    if trainer_precision not in _PRECISION_INPUT_STR.__args__:
        raise ValueError(f"Invalid trainer_precision: {trainer_precision}")

    if "mixed" in trainer_precision or trainer_precision == "32-true":
        dtype = torch.float32
    elif "f16" in trainer_precision:
        dtype = torch.float16
    elif "bf16" in trainer_precision:
        dtype = torch.bfloat16
    else:  # TODO: could be fp8 for H100
        dtype = None

    if attn_implementation is None:
        attn_implementation = "flash_attention_2" if is_flash_attn_available() else "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=False,
        use_cache=use_cache,
        torch_dtype=dtype,
        attn_implementation=attn_implementation,
        low_cpu_mem_usage=low_cpu_mem_usage,
    )

    # patch the model modules with liger-kernel modules
    if use_liger:
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        _apply_liger_kernel_to_instance(
            model=model,
            rope=True,
            cross_entropy=False,
            fused_linear_cross_entropy=True,
            rms_norm=False,
            swiglu=True,
        )
    if use_lora:
        from peft import LoraConfig, get_peft_model

        model.enable_input_require_grads()
        lora_config = LoraConfig(
            r=lora_rank, target_modules=lora_target_modules, lora_alpha=lora_alpha_to_rank_ratio * lora_rank
        )
        model = get_peft_model(
            model=model,
            peft_config=lora_config,
        )
    return model


def remove_prefix_from_dict_keys(dict_to_update_keys: dict, prefix: str) -> None:
    """Helper function to update dict's keys in place by removing specified prefix if it is present
    :param dict_to_update_keys: dict to remove prefix from keys
    :param prefix: prefix to remove from keys
    :return: None, update the dict in place
    """
    dict_to_update_keys_keys = list(dict_to_update_keys.keys())
    for k in dict_to_update_keys_keys:
        if k.startswith(prefix):
            new_k = k[len(prefix) :]
            dict_to_update_keys[new_k] = dict_to_update_keys[k]
            del dict_to_update_keys[k]


def remove_substring_from_dict_keys(
    dict_to_update_keys: dict, substring: str = SUBSTRING_TO_REMOVE_FROM_STATE_DICT
) -> None:
    """
    Helper function to remove a specified substring from all keys in a dictionary.

    :param dict_to_update_keys: The dictionary to update keys for.
    :param substring: The substring to remove from keys.
    """
    dict_keys = list(dict_to_update_keys.keys())
    for k in dict_keys:
        # Remove the substring
        updated_key = k.replace(substring, "")
        # Update the dictionary with the new key
        dict_to_update_keys[updated_key] = dict_to_update_keys[k]
        if updated_key != k:
            del dict_to_update_keys[k]


def consolidate_ckpt_to_hf_nonsharded(
    save_ckpt: str,
    original_model: str,
    output_path: Optional[str] = None,
    exclude_modules: Optional[List[str]] = None,
    remove_original_ckpt: bool = False,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_target_modules: Optional[Union[List[str], str]] = None,
    lora_alpha_to_rank_ratio: Optional[float] = None,
    verify_lora_saving_correctness: bool = False,
) -> None:
    """
    Consolidate a torch checkpoint to Hugging Face model format.
    :param save_ckpt: The path of the checkpoint **file** to consolidate.
    :param original_model: The original model to load the checkpoint into.
    :param output_path: The path to save the consolidated model to. If None, the model will be saved in the same directory
    as the checkpoint.
    :param exclude_modules: A list of module names to exclude from the checkpoint.
    :param remove_original_ckpt: Whether to remove the original checkpoint file after consolidation.
    """
    if os.path.isdir(save_ckpt):
        checkpoint_files = fnmatch.filter(os.listdir(save_ckpt), "*.ckpt")
        if len(checkpoint_files) == 0:
            raise ValueError(f"Failed to find any checkpoint in: {save_ckpt}")
        if len(checkpoint_files) > 1:
            raise ValueError(f"Found multiple checkpoints in: {save_ckpt}")
        save_ckpt = os.path.join(save_ckpt, checkpoint_files[0])
    elif not fnmatch.fnmatch(save_ckpt, "*.ckpt"):
        raise ValueError(f"Invalid checkpoint path: {save_ckpt}")

    if output_path is None:
        dir_name = os.path.dirname(save_ckpt)
        file_name = f"{os.path.splitext(os.path.basename(save_ckpt))[0]}-HF"
        output_path = os.path.join(dir_name, file_name)

    with torch.no_grad():
        state_dict = torch.load(save_ckpt, map_location=torch.device("cpu"))["state_dict"]
        if exclude_modules:
            exclude_keys = [key for key in state_dict.keys() if any(module in key for module in exclude_modules)]
            for key in exclude_keys:
                state_dict.pop(key)
        save_state_dict = {}
        local_model = AutoModelForCausalLM.from_pretrained(original_model, device_map="cpu")

        if use_lora:
            from peft import LoraConfig, get_peft_model

            local_model.enable_input_require_grads()
            lora_config = LoraConfig(
                r=lora_rank, target_modules=lora_target_modules, lora_alpha=lora_alpha_to_rank_ratio * lora_rank
            )
            local_model = get_peft_model(
                model=local_model,
                peft_config=lora_config,
            )

        local_model_keys = local_model.state_dict().keys()
        remove_substring_from_dict_keys(dict_to_update_keys=state_dict, substring=SUBSTRING_TO_REMOVE_FROM_STATE_DICT)
        for fsdp_key, local_key in zip(sorted(state_dict.keys()), sorted(local_model_keys)):
            if local_key not in fsdp_key:
                raise ValueError("Failed to match original and saved models keys.")
            save_state_dict.update({local_key: state_dict[fsdp_key]})
        local_model.load_state_dict(save_state_dict)

        if use_lora:
            peft_model_location = f"{output_path}-PEFT"
            local_model.save_pretrained(peft_model_location)
            local_model = local_model.merge_and_unload()

        local_model.save_pretrained(output_path)

    tokenizer = AutoTokenizer.from_pretrained(original_model)
    tokenizer.save_pretrained(output_path)

    if remove_original_ckpt:
        os.remove(save_ckpt)
    print(f"Saved {output_path} successfully.")


def consolidate_ckpt_to_hf_sharded(
    save_dir,
    original_model,
    output_path: Optional[str] = None,
    exclude_modules: Optional[List[str]] = None,
    remove_original_ckpt: bool = False,
) -> None:
    """
    Consolidate a torch distributed checkpoint to Hugging Face model format.

    :param save_ckpt: The path of the checkpoint **file** to consolidate.
    :param original_model: The original model to load the checkpoint into.
    :param output_path: The path to save the consolidated model to. If None, the model will be saved in the same directory
    as the checkpoint.
    :param exclude_modules: A list of module names to exclude from the checkpoint.
    :param remove_original_ckpt: Whether to remove the original checkpoint file after consolidation.
    """
    # saving
    distributed_checkpoint_location = save_dir
    hf_model_location = output_path or os.path.join(f"{distributed_checkpoint_location}-HF")
    os.makedirs(name=hf_model_location, exist_ok=True)
    if len(os.listdir(hf_model_location)) > 0:
        raise ValueError(f"Output directory '{hf_model_location}' is not empty.")

    # required for _load_distributed_checkpoint function, using gloo backend for CPU run
    log.info(f"Consolidating distributed checkpoint from {distributed_checkpoint_location}")

    from pathlib import Path

    from lightning.fabric.utilities.load import _load_distributed_checkpoint

    checkpoint = _load_distributed_checkpoint(Path(distributed_checkpoint_location))
    log.info("Formatting consolidated checkpoint")
    # Delete all keys except 'model' and convert key to 'state_dict'
    keys_to_delete = [key for key in checkpoint if key != "state_dict"]
    for key in keys_to_delete:
        del checkpoint[key]
    log.info("Updating keys in consolidated checkpoint state dict")

    if exclude_modules:
        exclude_keys = [
            key for key in checkpoint["state_dict"].keys() if any(module in key for module in exclude_modules)
        ]
        for key in exclude_keys:
            checkpoint["state_dict"].pop(key)
    remove_prefix_from_dict_keys(dict_to_update_keys=checkpoint["state_dict"], prefix=PREFIX_TO_REMOVE_FROM_STATE_DICT)
    remove_substring_from_dict_keys(
        dict_to_update_keys=checkpoint["state_dict"], substring=SUBSTRING_TO_REMOVE_FROM_STATE_DICT
    )

    log.info(f"Loading model {original_model}")
    model = AutoModelForCausalLM.from_pretrained(
        original_model,
        trust_remote_code=False,
        use_cache=False,
        torch_dtype=None,  # model loading with fp32 for FSDP intentionally
        attn_implementation="flash_attention_2",
        low_cpu_mem_usage=True,
    )
    log.info("Updating model state dict with consolidated checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    del checkpoint
    log.info("Deleted checkpoint")
    log.info(f"Saving model to {hf_model_location}")
    model.save_pretrained(hf_model_location)
    log.info(f"Model successfully saved to {hf_model_location}")

    log.info(f"Saving tokenizer to {hf_model_location}")
    tokenizer = AutoTokenizer.from_pretrained(original_model)
    tokenizer.save_pretrained(hf_model_location)
    log.info(f"Tokenizer successfully saved to {hf_model_location}")

    if remove_original_ckpt:
        log.info("Cleaning up raw fsdp checkpoint")
        shutil.rmtree(save_dir)
        log.info("Raw fsdp checkpoint cleaned up")


def consolidate_ckpt_to_hf(
    save_dir,
    original_model,
    output_path: Optional[str] = None,
    exclude_modules: Optional[List[str]] = None,
    remove_original_ckpt: bool = False,
    is_ckpt_sharded: bool = False,
    use_lora: bool = False,
    lora_rank: Optional[int] = None,
    lora_target_modules: Optional[Union[List[str], str]] = None,
    lora_alpha_to_rank_ratio: Optional[float] = None,
    verify_lora_saving_correctness: Optional[bool] = False,
) -> None:
    """Deprecated function. Please use `consolidate_ckpt_to_hf_sharded` or `consolidate_ckpt_to_hf_nonsharded` instead."""
    if is_ckpt_sharded:
        consolidate_ckpt_to_hf_sharded(
            save_dir=save_dir,
            original_model=original_model,
            output_path=output_path,
            exclude_modules=exclude_modules,
            remove_original_ckpt=remove_original_ckpt,
        )
    else:
        consolidate_ckpt_to_hf_nonsharded(
            save_ckpt=save_dir,
            original_model=original_model,
            output_path=output_path,
            exclude_modules=exclude_modules,
            remove_original_ckpt=remove_original_ckpt,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_target_modules=lora_target_modules,
            lora_alpha_to_rank_ratio=lora_alpha_to_rank_ratio,
            verify_lora_saving_correctness=verify_lora_saving_correctness,
        )


def check_cuda_version(required_version: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please install CUDA to use GPU features.")

    cuda_version = torch.version.cuda
    if cuda_version is None:
        raise RuntimeError("CUDA version could not be determined.")

    # Normalize the version string (e.g., '12.0' to '12.0.0')
    cuda_version_normalized = ".".join(cuda_version.split(".")[:2]) + ".0"

    if version.parse(cuda_version_normalized) < version.parse(required_version):
        raise RuntimeError(f"CUDA version {cuda_version} is installed, but version >= {required_version} is required.")


def check_torch_version(required_version: str):
    current_version = torch.__version__
    parsed_current = version.parse(current_version)
    parsed_required = version.parse(required_version)

    if parsed_current >= parsed_required:
        # Current version is acceptable
        return
    elif parsed_current.base_version == parsed_required.base_version and parsed_current.is_prerelease:
        # Current version is a pre-release of the required version
        return
    else:
        raise RuntimeError(
            f"PyTorch version {current_version} is installed, but version >= {required_version} is required."
        )


def is_sm89_or_later():
    # Float8 is only supported on SM89 or later (H100+ GPUs)
    return torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)
