import functools
import importlib.metadata
from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F

# from liger_kernel.transformers import LigerFusedLinearJSD
from liger_kernel.chunked_loss import LigerFusedLinearJSDLoss
from packaging import version
from torch.nn import Module

IGNORE_INDEX = -100


# Check if liger-kernel version requires passing bias into LigerFusedLinearJSDLoss
try:
    current_version = importlib.metadata.version("liger-kernel")
    SUPPORTS_BIAS = version.parse(current_version) >= version.parse("0.5.5")
except (importlib.metadata.PackageNotFoundError, Exception):
    SUPPORTS_BIAS = False


class PerTokenDistillationLoss(ABC, Module):
    def __init__(self, use_liger=False, *args, **kwargs):
        super().__init__()
        self.use_liger = use_liger

    @abstractmethod
    def per_token_loss(self, probs, teacher_probs, inf_mask):
        pass

    def liger_chunked_fused_linear_loss(
        self,
        shift_student_hidden_states,
        student_model_lm_head_weight,
        shift_teacher_hidden_states,
        teacher_model_lm_head_weight,
        shift_labels,
        ignore_index,
        temperature=1.0,
    ):
        pass

    @staticmethod
    def _shift_and_mask(
        gt_token_ids,
        logits_or_hidden_states,
        teacher_logits_hidden_states,
        ignore_index,
        is_shifted,
    ):
        # shift logits or hidden/_states if needed or just shift label with mask check
        shifted_labels = gt_token_ids[:, 1:].contiguous()  # label always shift
        loss_mask = (shifted_labels != ignore_index).int()
        # If all tokens are ignored, skip the loss computation,
        # Otherwise loss will be NaN
        if torch.all(1 - loss_mask):
            return torch.tensor(0.0, requires_grad=True)
        shifted_logits_or_hidden_states = (
            logits_or_hidden_states if is_shifted else logits_or_hidden_states[:, :-1, :].contiguous()
        )
        shifted_teacher_logits_or_hidden_states = (
            teacher_logits_hidden_states if is_shifted else teacher_logits_hidden_states[:, :-1, :].contiguous()
        )
        return shifted_labels, shifted_logits_or_hidden_states, shifted_teacher_logits_or_hidden_states, loss_mask

    def forward(
        self,
        gt_token_ids,
        logits_or_hidden_states,
        teacher_logits_or_hidden_states,
        logits_or_hidden_states_shifted=False,
        ignore_index=IGNORE_INDEX,
        temperature=1.0,
        student_model_lm_head_weight=None,
        teacher_model_lm_head_weight=None,
        **kwargs,
    ):
        """
        Forward pass for the distillation loss computation.

        Args:
            gt_token_ids (torch.Tensor): Ground truth token IDs. Shape: (batch_size, sequence_length)
            logits_or_hidden_states (torch.Tensor): Logits from the student model. Shape: (batch_size, sequence_length, vocab_size),
                if use_liger=True, this will be hidden_states and student_model_lm_head_weight needs to be provided to get the lm_head projection
            teacher_logits_or_hidden_states (torch.Tensor): Logits from the teacher model. Shape: (batch_size, sequence_length, vocab_size)
                if use_liger=True, this will be hidden_states and teacher_model_lm_head_weight needs to be provided to get the lm_head projection
            logits_or_hidden_states_shifted (bool, optional): Whether the logits are already shifted. Defaults to False.
            ignore_index (int, optional): Index to ignore in the loss computation. Defaults to -100.
            temperature (float, optional): Temperature for distillation. Defaults to 1.0.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Computed distillation loss. Shape: scalar
        """
        # If the teacher and student token size is different, pad student logits to match the teacher's.
        # This only applies to cases where they share exactly the same vocab and tokenizer just
        # that teacher logit is padded for some training efficiency such as
        # https://huggingface.co/Qwen/Qwen1.5-72B-Chat/discussions/1#662883f568adf59b07b176d2
        if torch.all(gt_token_ids == -100):
            return torch.tensor(0.0, requires_grad=True, device=logits_or_hidden_states.device)

        # don't use liger if is_eval is True
        is_eval = kwargs.get("is_eval", False)

        if self.use_liger and not is_eval:
            shift_labels, shift_student_hidden_states, shift_teacher_hidden_states, _ = self._shift_and_mask(
                gt_token_ids,
                logits_or_hidden_states,
                teacher_logits_or_hidden_states,
                ignore_index,
                logits_or_hidden_states_shifted,
            )

            shift_student_hidden_states = shift_student_hidden_states.reshape(
                -1, shift_student_hidden_states.shape[-1]
            ).contiguous()
            shift_teacher_hidden_states = shift_teacher_hidden_states.reshape(
                -1, shift_teacher_hidden_states.shape[-1]
            ).contiguous()
            shift_labels = shift_labels.reshape(-1)

            return self.liger_chunked_fused_linear_loss(
                # TODO: need to cast to float32 because torch.compile gives
                # RuntimeError: attempting to assign a gradient with dtype 'float'
                # to a tensor with dtype 'c10::BFloat16'. Issue reference:
                # https://github.com/pytorch/pytorch/issues/111317
                shift_student_hidden_states.to(dtype=torch.float32),
                student_model_lm_head_weight.to(dtype=torch.float32),
                shift_teacher_hidden_states.to(dtype=torch.float32),
                teacher_model_lm_head_weight.to(dtype=torch.float32),
                shift_labels,
                ignore_index,
                temperature,
            )
        else:
            logits = logits_or_hidden_states
            teacher_logits = teacher_logits_or_hidden_states
            logits_shifted = logits_or_hidden_states_shifted
            if teacher_logits.shape[-1] > logits.shape[-1]:
                pad_size = teacher_logits.shape[-1] - logits.shape[-1]
                pad_tensor = torch.zeros((*logits.shape[:-1], pad_size), dtype=logits.dtype, device=logits.device)
                logits = torch.cat([logits, pad_tensor], dim=-1)

            if temperature != 1.0:
                logits = logits / temperature
                teacher_logits = teacher_logits / temperature
            shifted_teacher_logits = teacher_logits if logits_shifted else teacher_logits[:, :-1, :].contiguous()
            shift_labels, shifted_logits, shifted_teacher_logits, loss_mask = self._shift_and_mask(
                gt_token_ids, logits, teacher_logits, ignore_index, logits_shifted
            )
            inf_mask = torch.isinf(shifted_logits)  # do we need this? (maybe won't have inf logits)

            teacher_probs = F.softmax(shifted_teacher_logits, dim=-1, dtype=torch.float32)
            probs = F.softmax(shifted_logits, dim=-1, dtype=torch.float32)
            per_token_loss = self.per_token_loss(probs, teacher_probs, inf_mask)  # [B * T,]
            distill_loss = torch.sum(per_token_loss * loss_mask) / torch.sum(loss_mask)
            # Perform temperature scaling on the loss based on Hinton's 2015 paper
            # Mathematically we should perform temperature T^2 scaling on the
            # loss to compensate for the scaling of the logits during the
            # gradient computation.

            # TODO: Check if this is necessary since GKDTrainer
            # https://github.com/huggingface/trl/blob/main/trl/trainer/gkd_trainer.py#L167
            # does not perform such temperature scaling
            return distill_loss * (temperature**2)


class ForwardKLDiv(PerTokenDistillationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def liger_chunked_fused_linear_loss(
        self,
        shift_student_hidden_states,
        student_model_lm_head_weight,
        shift_teacher_hidden_states,
        teacher_model_lm_head_weight,
        shift_labels,
        ignore_index,
        temperature=1.0,
    ):
        liger_fkl = LigerFusedLinearJSDLoss(beta=1.0, weight_hard_loss=0.0, weight_soft_loss=1.0)
        kwargs = {"student_bias": None, "teacher_bias": None} if SUPPORTS_BIAS else {}
        return liger_fkl(
            shift_student_hidden_states,
            student_model_lm_head_weight,
            shift_teacher_hidden_states,
            teacher_model_lm_head_weight,
            shift_labels,
            **kwargs,
        )

    @staticmethod
    def _per_token_loss(probs, teacher_probs, inf_mask):
        prod_probs = torch.masked_fill(-teacher_probs * torch.log(probs), inf_mask, 0)
        return torch.sum(prod_probs, dim=-1)

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        return self._per_token_loss(probs, teacher_probs, inf_mask)


class ReverseKLDiv(PerTokenDistillationLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def liger_chunked_fused_linear_loss(
        self,
        shift_student_hidden_states,
        student_model_lm_head_weight,
        shift_teacher_hidden_states,
        teacher_model_lm_head_weight,
        shift_labels,
        ignore_index,
        temperature=1.0,
    ):
        liger_rkl = LigerFusedLinearJSDLoss(beta=0.0, weight_hard_loss=0.0, weight_soft_loss=1.0)
        kwargs = {"student_bias": None, "teacher_bias": None} if SUPPORTS_BIAS else {}
        return liger_rkl(
            shift_student_hidden_states,
            student_model_lm_head_weight,
            shift_teacher_hidden_states,
            teacher_model_lm_head_weight,
            shift_labels,
            **kwargs,
        )

    @staticmethod
    def _per_token_loss(probs, teacher_probs, inf_mask):
        prod_probs = torch.masked_fill(probs * (torch.log(probs) - torch.log(teacher_probs)), inf_mask, 0)
        return torch.sum(prod_probs, dim=-1)

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        return self._per_token_loss(probs, teacher_probs, inf_mask)


class CombinedKLDiv(PerTokenDistillationLoss):
    def __init__(self, forward_ratio: float = 0.5, *args, **kwargs):
        super().__init__()
        self.forward_ratio = forward_ratio

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        forward_kl = ForwardKLDiv._per_token_loss(probs, teacher_probs, inf_mask)
        reverse_kl = ReverseKLDiv._per_token_loss(probs, teacher_probs, inf_mask)
        return torch.lerp(forward_kl, reverse_kl, 1 - self.forward_ratio)


class JSDiv(PerTokenDistillationLoss):
    """
    Jensen-Shannon Divergence (JSD) based per-token distillation loss. if use_liger=True, we will do it in a
    memory efficient way leveraging Liger-Kernel:
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_jsd.py#L20


    This class implements a loss function based on the Jensen-Shannon Divergence (JSD) for distillation
    purposes. It interpolates between the student and teacher Kullback-Leibler (KL) divergences to compute
    the final loss per token.

    Args:
        beta (float, optional): A weighting factor between the student's and teacher's KL divergence.
            - If `beta = 0`, the loss becomes Forward KL divergence.
            - If `beta = 1`, the loss becomes Reverse KL divergence.
            - If `beta = 0.5` (default), the loss is an equal mixture of both.
        *args: Additional arguments passed to the parent class.
        **kwargs: Additional keyword arguments passed to the parent class.

    Methods:
        per_token_loss(probs, teacher_probs, inf_mask):
            Computes the per-token Jensen-Shannon Divergence loss using a linear interpolation of the
            student and teacher KL divergences.

    """

    def __init__(self, beta: float = 0.5, use_liger=False, *args, **kwargs):
        super().__init__(use_liger=use_liger)
        self.beta = beta

    def liger_chunked_fused_linear_loss(
        self,
        shift_student_hidden_states,
        student_model_lm_head_weight,
        shift_teacher_hidden_states,
        teacher_model_lm_head_weight,
        shift_labels,
        ignore_idx,
        temperature,
    ):
        # set weight_hard_loss to 0.0 and weight_soft_loss to 1.0 to only compute the distillation loss
        fused_linear_jsd = LigerFusedLinearJSDLoss(
            beta=self.beta, ignore_index=ignore_idx, temperature=temperature, weight_hard_loss=0.0, weight_soft_loss=1.0
        )
        kwargs = {"student_bias": None, "teacher_bias": None} if SUPPORTS_BIAS else {}
        return fused_linear_jsd(
            shift_student_hidden_states,
            student_model_lm_head_weight,
            shift_teacher_hidden_states,
            teacher_model_lm_head_weight,
            shift_labels,
            **kwargs,
        )

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        avg_probs = torch.lerp(probs, teacher_probs, self.beta)
        # Here we choose to always use Reverse KL way of computing since we eliminate the
        # teacher log teacher term in forward KL to reduce compute but here it cannot be
        # removed as the avg_probs term is mixed
        if self.beta == 0.0:
            # Return teacher KL only which is equivalent to normal forward KL (computed using Reverse KL)
            # Compute with reverse KL here to align with other conditions
            return ReverseKLDiv._per_token_loss(teacher_probs, avg_probs, inf_mask)
        elif self.beta == 1.0:
            # Return student KL only which is equivalent to normal reverse KL
            return ReverseKLDiv._per_token_loss(probs, avg_probs, inf_mask)
        else:
            student_kl = ReverseKLDiv._per_token_loss(probs, avg_probs, inf_mask)
            teacher_kl = ReverseKLDiv._per_token_loss(teacher_probs, avg_probs, inf_mask)
            return torch.lerp(student_kl, teacher_kl, self.beta)


# Refer to https://arxiv.org/pdf/2307.15190 for the TVD loss used in the context of distillation
class TVDist(PerTokenDistillationLoss):
    def __init__(self, log_scale: bool = False, *args, **kwargs):
        super().__init__()
        self.log_scale = log_scale

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        if self.log_scale:
            probs, teacher_probs = torch.log(probs), torch.log(teacher_probs)
        abs_diff = torch.masked_fill(torch.abs(probs - teacher_probs), inf_mask, 0)
        return 0.5 * torch.sum(abs_diff, dim=-1)


class SkewKLDiv(PerTokenDistillationLoss):
    """
    Skew KLD in https://arxiv.org/pdf/2402.03898

    Args:
        alpha (float, default to 0.1): alpha in original DistiLLM paper, interpolate
            between student and teacher prob
        reverse (bool, default to False): whether to do Skew Reverse KLD
    """

    def __init__(self, alpha: float = 0.1, reverse=False, *args, **kwargs):
        super().__init__()
        self.alpha = alpha
        self.reverse = reverse

    def per_token_loss(self, probs, teacher_probs, inf_mask):
        if self.reverse:
            teacher_probs, probs = probs, teacher_probs
        avg_probs = torch.lerp(probs, teacher_probs, self.alpha)
        return ReverseKLDiv._per_token_loss(teacher_probs, avg_probs, inf_mask)


DISTILL_LOSS_MAP = {
    "forward_kl": ForwardKLDiv,
    "reverse_kl": ReverseKLDiv,
    "combined_kl": CombinedKLDiv,
    "js": JSDiv,
    "tvd": TVDist,
    "default": ForwardKLDiv,
    "skl": SkewKLDiv,
    "srkl": functools.partial(SkewKLDiv, reverse=True),
    "fljsd": functools.partial(JSDiv, use_liger=True),
}
