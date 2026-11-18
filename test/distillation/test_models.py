from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from fmchisel.distillation import (
    DISTILL_LOSS_MAP,
    DistillLanguageModel,
    DistillTrainingConfig,
)

VOCAB_SIZE = 64


def _tiny_llama(hidden_size):
    config = LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=hidden_size,
        intermediate_size=2 * hidden_size,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(config).eval()


@pytest.mark.parametrize("distill_loss", ["js", "forward_kl", "reverse_kl", "tvd", "fljsd"])
def test_liger_training_step_distillation_loss_matches_logits(distill_loss):
    torch.manual_seed(0)
    # The teacher is wider than the student, as with a 1B student and an 8B teacher.
    student, teacher = _tiny_llama(16), _tiny_llama(32)
    for param in teacher.parameters():
        param.requires_grad_(False)
    input_ids = torch.randint(0, VOCAB_SIZE, (2, 12))
    batch = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids), "labels": input_ids.clone()}

    config = DistillTrainingConfig(enable_distill=True, teacher_model_path="teacher", distill_loss=distill_loss)
    reference_loss_fn = DISTILL_LOSS_MAP[distill_loss](**config.distill_loss_kwargs)
    reference_loss_fn.use_liger = False
    with torch.no_grad():
        expected = reference_loss_fn(input_ids, student(input_ids).logits, teacher(input_ids).logits)

    model = DistillLanguageModel(
        model_path="student",
        max_lr=1e-4,
        weight_decay=0.0,
        warmup_ratio=0.0,
        optimizer="adamw",
        distill_training_config=config,
        use_liger=True,
    )
    model.model, model.teacher_model = student, teacher
    # The fused linear cross entropy kernel needs Triton; the hard loss is not under test here.
    model._compute_flce = lambda **kwargs: SimpleNamespace(loss=torch.tensor(0.0))
    logged = {}
    model.log_dict = lambda metrics, **kwargs: logged.update(metrics)

    model._training_step(batch, 0)
    distillation_loss = logged["distillation_loss"]

    torch.testing.assert_close(distillation_loss.detach(), expected, rtol=1e-4, atol=1e-6)
    distillation_loss.backward()
    assert student.lm_head.weight.grad is not None
    assert all(param.grad is None for param in teacher.parameters())
