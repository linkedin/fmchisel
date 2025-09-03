import logging

from quantization_utils import quantize
from transformers import HfArgumentParser

from fmchisel.config import CalibrationDataConfig, QuantizationConfig

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = HfArgumentParser((QuantizationConfig, CalibrationDataConfig))
    (quantization_config, data_config) = parser.parse_args_into_dataclasses()
    logger.info(f"quantization_config = {quantization_config}")
    logger.info(f"data_config = {data_config}")

    quantize(quantization_config, data_config)
