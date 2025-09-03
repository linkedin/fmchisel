from unittest.mock import Mock

import pytest
import torch
from torch.nn import functional as F

from fmchisel.distillation.losses import JSDiv

if torch.cuda.is_available():
    CUDA_IS_AVAILABLE = True
    DEV = "cuda:0"
else:
    CUDA_IS_AVAILABLE = False
    DEV = "cpu"


# This is the exact implementation from DistiLLM https://arxiv.org/pdf/2402.03898,
# copied from https://github.com/jongwooko/distillm/blob/master/distillm/losses.py#L32
def distillm_js_distance(logits, teacher_logits, no_model_batch, lam=0.9):
    teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_probs = F.softmax(logits, dim=-1, dtype=torch.float32)
    mixed_probs = (1 - lam) * teacher_probs + lam * student_probs

    teacher_logprobs = F.log_softmax(teacher_logits, dim=-1, dtype=torch.float32)
    student_logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
    mixed_logprobs = torch.log(mixed_probs)

    mask = (no_model_batch["label"] != -100).int()
    inf_mask = torch.isinf(logits) | torch.isinf(teacher_logits)

    prod_probs = torch.masked_fill(student_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(student_probs * student_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss = lam * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

    prod_probs = torch.masked_fill(teacher_probs * mixed_logprobs, inf_mask, 0)
    prod_probs -= torch.masked_fill(teacher_probs * teacher_logprobs, inf_mask, 0)
    x = torch.sum(prod_probs, dim=-1).view(-1)
    distil_loss += (1 - lam) * -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)
    return distil_loss


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, vocab_size, beta, num_token_ignored_per_seq",
    [
        (10, 1024, 2048, 512, 0.4, 1),
        (32, 128, 1024, 128, 0.5, 8),
        # very smol size
        (1, 2, 16, 2, 0.5, 0),
    ],
)
def test_jsd_correctness(batch_size, seq_len, hidden_size, vocab_size, beta, num_token_ignored_per_seq):
    student_hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(DEV)
    teacher_hidden_states = torch.randn(batch_size, seq_len, hidden_size).to(DEV)

    student_lm_head_weight = torch.ones(hidden_size, vocab_size).to(DEV)
    teacher_lm_head_weight = torch.ones(hidden_size, vocab_size).to(DEV)

    # Mock student and teacher models
    student_model, teacher_model = Mock(), Mock()
    student_model.lm_head, teacher_model.lm_head = Mock(), Mock()
    student_model.lm_head.weight = student_lm_head_weight.T.to(DEV)
    teacher_model.lm_head.weight = teacher_lm_head_weight.T.to(DEV)

    logits = student_hidden_states @ student_lm_head_weight
    teacher_logits = teacher_hidden_states @ teacher_lm_head_weight

    logits_1 = logits.clone().requires_grad_(True)
    logits_2 = logits.clone().requires_grad_(True)

    label = torch.randint(0, vocab_size, (batch_size, seq_len)).to(DEV)

    # randomly put some tokens to be ignored
    ignore_index = -100
    for i in range(batch_size):
        ignore_indices = torch.randperm(seq_len)[:num_token_ignored_per_seq]
        label[i, ignore_indices] = ignore_index

    jsd = JSDiv(beta=beta)
    loss = jsd(label, logits_1, teacher_logits)

    jsd_liger = JSDiv(beta=beta, use_liger=True)
    loss_liger = jsd_liger(
        label,
        student_hidden_states,
        teacher_hidden_states,
        student_model_lm_head_weight=student_lm_head_weight.T,
        teacher_model_lm_head_weight=teacher_lm_head_weight.T,
    )

    ref_loss = distillm_js_distance(
        logits_2[:, :-1, :], teacher_logits[:, :-1, :], {"label": label[:, 1:]}, lam=1 - beta
    )

    assert torch.allclose(loss, ref_loss)
    assert torch.allclose(loss_liger, ref_loss)

    loss.backward()
    ref_loss.backward()
    assert torch.allclose(logits_1.grad, logits_2.grad)


@pytest.mark.parametrize(
    "batch_size, seq_len, hidden_size, beta",
    [
        (10, 1024, 2048, 0.4),
        (32, 128, 1024, 0.5),
        (1, 2, 16, 0.5),
    ],
)
def test_jsd_all_ignored(batch_size, seq_len, hidden_size, beta):
    logits = torch.randn(batch_size, seq_len, hidden_size)
    teacher_logits = torch.randn(batch_size, seq_len, hidden_size)

    logits_1 = logits.clone().requires_grad_(True)

    label = torch.empty((batch_size, seq_len))
    label.fill_(-100)

    jsd = JSDiv(beta=beta)

    loss = jsd(label, logits_1, teacher_logits)
    assert torch.allclose(loss, torch.zeros_like(loss))

    loss.backward()
    assert logits_1.grad is None
