from llmcompressor import oneshot
from transformers import AutoTokenizer

from fmchisel.config import CalibrationDataConfig, PruningConfig
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader
from fmchisel.pruning.osscar.base import OSSCARModifier
from fmchisel.pruning.osscar.utils.helpers import cleanup_after_prune, pack


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

    recipe = OSSCARModifier(
        num_drop_mlp_neuron=pruning_config.num_drop_mlp_neuron,
        num_drop_attn_group=pruning_config.num_drop_attn_group,
    )

    oneshot(
        model=pruning_config.model,
        dataset=tokenized_dataset,
        recipe=recipe,
        save_compressed=pruning_config.save_compressed,  # We have custom packing functions for this type of pruning.
        output_dir=pruning_config.output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=data_config.num_calibration_samples,
    )

    if not pruning_config.save_compressed:
        # The lm_head is always saved with the model checkpoint in llmcompressor,
        # even if the model has tied word embeddings. This leads to a bug where
        # models with tied word embeddings get a random lm_head.
        # As a workaround, after pruning, we load the model and copy it to a new
        # one with correct lm_head settings.
        pack(
            pruning_config.output_dir,
            0,
            0,
            pruning_config.model,
        )

    else:
        pack(
            pruning_config.output_dir,
            pruning_config.num_drop_mlp_neuron,
            pruning_config.num_drop_attn_group,
            pruning_config.model,
        )

    cleanup_after_prune(pruning_config.output_dir)
