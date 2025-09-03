import logging

from structured_pruning_utils import prune
from transformers import HfArgumentParser

from fmchisel.config import CalibrationDataConfig, StructuredPruningConfig

logger = logging.getLogger(__name__)


if __name__ == "__main__":

    parser = HfArgumentParser((StructuredPruningConfig, CalibrationDataConfig))
    (pruning_config, data_config) = parser.parse_args_into_dataclasses()
    logger.info(f"pruning_config = {pruning_config}")
    logger.info(f"data_config = {data_config}")

    prune(pruning_config, data_config)
