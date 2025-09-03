import json
from dataclasses import replace

import pytest

from fmchisel.distillation import DISTILL_LOSS_MAP, DistillTrainingConfig

_DEFAULT_CONFIG = DistillTrainingConfig(
    enable_distill=True,
    teacher_model_path="foo",
    temperature=1.0,
    distillation_loss_ratio=0.9,
    distill_loss="forward_kl",
    sample_method="supervised",
)


def test_get_jsd():
    js_beta = 0.3
    config = replace(_DEFAULT_CONFIG, distill_loss="js", js_beta=js_beta)
    loss = DISTILL_LOSS_MAP[config.distill_loss](**config.distill_loss_kwargs)
    assert loss.beta == js_beta


def test_get_tvd():
    tvd_log_scale = True
    config = replace(_DEFAULT_CONFIG, distill_loss="tvd", tvd_log_scale=tvd_log_scale)
    loss = DISTILL_LOSS_MAP[config.distill_loss](**config.distill_loss_kwargs)
    assert loss.log_scale == tvd_log_scale


def test_get_skl_srkl():
    skl_alpha = 0.3
    for loss_type in ["skl", "srkl"]:
        config = replace(_DEFAULT_CONFIG, distill_loss=loss_type, skl_alpha=skl_alpha)
        loss = DISTILL_LOSS_MAP[config.distill_loss](**config.distill_loss_kwargs)
        assert loss.alpha == skl_alpha


def test_get_combined_kl():
    forward_ratio = 0.3
    config = replace(_DEFAULT_CONFIG, distill_loss="combined_kl", forward_ratio=forward_ratio)
    loss = DISTILL_LOSS_MAP[config.distill_loss](**config.distill_loss_kwargs)
    assert loss.forward_ratio == forward_ratio


def test_ser_deser_successful():
    serialized_str = _DEFAULT_CONFIG.to_json()
    deserialized_cfg = DistillTrainingConfig.from_json(serialized_str)
    assert _DEFAULT_CONFIG == deserialized_cfg


def test_deser_with_unknown():
    json_dict = json.loads(_DEFAULT_CONFIG.to_json())
    # add an unknown field
    json_dict["foo"] = "bar"
    serialized_str = json.dumps(json_dict)
    with pytest.raises(Exception, match="undefined"):
        _ = DistillTrainingConfig.from_json(serialized_str)
