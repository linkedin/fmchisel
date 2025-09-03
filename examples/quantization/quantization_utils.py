from llmcompressor import oneshot
from transformers import AutoTokenizer

from fmchisel.config import CalibrationDataConfig, QuantizationConfig
from fmchisel.data.calibration_datautil import HFCalibrationDataLoader


def quantize(quantization_config: QuantizationConfig, data_config: CalibrationDataConfig):

    tokenizer = AutoTokenizer.from_pretrained(quantization_config.model)
    max_seq_length = quantization_config.model_max_length or tokenizer.model_max_length

    tokenized_dataset = HFCalibrationDataLoader(
        nsamples=data_config.num_calibration_samples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset=data_config.dataset,
        data_field=data_config.data_field,
        data_dir=data_config.data_dir,
        data_split=data_config.data_split,
    ).get_tokenized_calibration()

    if quantization_config.quantization_recipe and "yaml" in quantization_config.quantization_recipe:
        recipe = quantization_config.quantization_recipe
    else:
        raise ValueError("No valid quantization recipe was provided.")

    oneshot(
        model=quantization_config.model,
        dataset=tokenized_dataset,
        recipe=recipe,
        save_compressed=True,
        output_dir=quantization_config.output_dir,
        max_seq_length=max_seq_length,
        num_calibration_samples=data_config.num_calibration_samples,
    )
