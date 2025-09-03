import logging

from llmcompressor import oneshot
from transformers import AutoTokenizer

from fmchisel.config import CalibrationDataConfig, PruningConfig
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader
from fmchisel.pruning.osscar.utils.helpers import cleanup_after_prune, pack

SPARSE_GPT = "SparseGPT"
WANDA = "wanda"
ALPS = "ALPS"

logger = logging.getLogger(__name__)


def get_pruning_modifier(
    pruning_strategy: str,
    sparsity: float,
    mask_structure: str,
):

    common_kwargs = {
        "sparsity": sparsity,
        "mask_structure": mask_structure,
        "targets": "__ALL_PRUNABLE__",
    }
    if pruning_strategy == SPARSE_GPT:
        from llmcompressor.modifiers.obcq import SparseGPTModifier

        recipe = SparseGPTModifier(**common_kwargs)
        return recipe
    elif pruning_strategy == WANDA:
        from llmcompressor.modifiers.pruning import WandaPruningModifier

        recipe = WandaPruningModifier(**common_kwargs)
        return recipe
    elif pruning_strategy == ALPS:
        from fmchisel.pruning.alps.base import ALPSModifier

        recipe = ALPSModifier(**common_kwargs)
        return recipe
    else:
        raise ValueError(f"Unsupported pruning strategy: {pruning_strategy}")


def prune(pruning_config: PruningConfig, data_config: CalibrationDataConfig):

    tokenizer = AutoTokenizer.from_pretrained(pruning_config.model)
    max_seq_length = pruning_config.model_max_length or tokenizer.model_max_length

    tokenized_dataset = HFCalibrationDataLoader(
        nsamples=data_config.num_calibration_samples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset=data_config.dataset,
        data_field=data_config.data_field,
        data_dir=data_config.data_dir,
        data_split=data_config.data_split,
    ).get_tokenized_calibration()

    if pruning_config.pruning_yaml_recipe and "yaml" in pruning_config.pruning_yaml_recipe:
        logger.info("Found a yaml recipe, ignoring pruning_strategy, sparsity, prunen and prunem.")
        recipe = pruning_config.pruning_yaml_recipe
    else:
        logger.info(
            "No yaml recipe provided, creating the recipe based on pruning_strategy, sparsity, prunen and prunem."
        )
        recipe = get_pruning_modifier(
            pruning_strategy=pruning_config.pruning_strategy,
            sparsity=pruning_config.sparsity,
            mask_structure=f"{pruning_config.prunen}:{pruning_config.prunem}",
        )

    oneshot(
        model=pruning_config.model,
        dataset=tokenized_dataset,
        recipe=recipe,
        save_compressed=pruning_config.save_compressed,
        output_dir=pruning_config.output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=data_config.num_calibration_samples,
    )
    # The lm_head is always saved with the model checkpoint in llmcompressor,
    # even if the model has tied word embeddings. This leads to a bug where
    # models with tied word embeddings get a random lm_head.
    # As a workaround, after pruning, we load the model and copy it to a new
    # one with correct lm_head settings.
    if not pruning_config.save_compressed:
        pack(
            pruning_config.output_dir,
            0,
            0,
            pruning_config.model,
        )
        cleanup_after_prune(pruning_config.output_dir)
