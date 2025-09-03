import functools
from typing import Dict, Iterable, List, Optional, Tuple, Union

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn.utils import clip_grad_norm_
from torch.utils._foreach_utils import (
    _device_has_foreach_support,
    _group_tensors_by_device_and_dtype,
    _has_foreach_support,
)

# copied from torch main branch now
# after torch bump to 2.6.0+ with this pr merged https://github.com/pytorch/pytorch/commit/2ee91db03d41103dce2dc76537480ea17d5e89f7
# we can just do `from torch.nn.utils import get_total_norm`
_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def _no_grad(func):
    """
    This wrapper is needed to avoid a circular import when using @torch.no_grad on the exposed functions
    clip_grad_norm_ and clip_grad_value_ themselves.
    """

    def _no_grad_wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)

    functools.update_wrapper(_no_grad_wrapper, func)
    return _no_grad_wrapper


@_no_grad
def get_total_norm(
    tensors: _tensor_or_tensors,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: Optional[bool] = None,
) -> torch.Tensor:
    r"""Compute the norm of an iterable of tensors.

    The norm is computed over the norms of the individual tensors, as if the norms of
    the individual tensors were concatenated into a single vector.

    Args:
        tensors (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will be normalized
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of :attr:`tensors` is ``nan``, ``inf``, or ``-inf``.
            Default: ``False``
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``

    Returns:
        Total norm of the tensors (viewed as a single vector).
    """
    if isinstance(tensors, torch.Tensor):
        tensors = [tensors]
    else:
        tensors = list(tensors)
    norm_type = float(norm_type)
    if len(tensors) == 0:
        return torch.tensor(0.0)
    first_device = tensors[0].device
    grouped_tensors: Dict[
        Tuple[torch.device, torch.dtype], Tuple[List[List[Tensor]], List[int]]
    ] = _group_tensors_by_device_and_dtype(
        [tensors]  # type: ignore[list-item]
    )  # type: ignore[assignment]

    norms: List[Tensor] = []
    for (device, _), ([device_tensors], _) in grouped_tensors.items():  # type: ignore[assignment]
        if (foreach is None and _has_foreach_support(device_tensors, device)) or (
            foreach and _device_has_foreach_support(device)
        ):
            norms.extend(torch._foreach_norm(device_tensors, norm_type))
        elif foreach:
            raise RuntimeError(f"foreach=True was passed, but can't use the foreach API on {device.type} tensors")
        else:
            norms.extend([torch.linalg.vector_norm(g, norm_type) for g in device_tensors])

    total_norm = torch.linalg.vector_norm(torch.stack([norm.to(first_device) for norm in norms]), norm_type)

    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f"The total norm of order {norm_type} for gradients from "
            "`parameters` is non-finite, so it cannot be clipped. To disable "
            "this error and scale the gradients by the non-finite norm anyway, "
            "set `error_if_nonfinite=False`"
        )
    return total_norm


class GradientClippingCallback(pl.Callback):
    def __init__(self, clip_val, log_grad_norm=False):
        super().__init__()
        self.clip_val = clip_val
        self.log_grad_norm = log_grad_norm

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        root_fsdp_module = self._get_root_fsdp_module(pl_module)

        # TODO: QQ, note for FSDP2 it does not work for PP case, need to change to https://github.com/pytorch/torchtitan/blob/main/torchtitan/utils.py#L276
        if root_fsdp_module:
            # FSDP gradients clipping
            total_norm_before = root_fsdp_module.clip_grad_norm_(self.clip_val)
        else:
            # Standard gradient clipping
            total_norm_before = clip_grad_norm_(pl_module.parameters(), self.clip_val)

        if self.log_grad_norm:
            total_norm_after = self.compute_total_grad_norm(pl_module)
            trainer.logger.log_metrics(
                {
                    "grad_norm_after_clipping": total_norm_after,
                    "grad_norm_before_clipping": total_norm_before,
                },
                step=trainer.global_step,
            )

    def _get_root_fsdp_module(self, pl_module):
        if isinstance(pl_module, FSDP) and pl_module._is_root:
            return pl_module
        for module in pl_module.modules():
            if isinstance(module, FSDP) and module._is_root:
                return module
        return None

    @staticmethod
    def compute_total_grad_norm(pl_module):
        grads = [p.grad for p in pl_module.parameters() if p.grad is not None]
        total_norm = get_total_norm(grads, norm_type=2.0, error_if_nonfinite=False, foreach=None)
        return total_norm


class SparseTrainingCallback(pl.Callback):
    """
    A callback for sparsity aware training, to make sure params that are supposed to stay zero
    will not be updated throughout the training.
    """

    def __init__(self):
        super().__init__()
        self.mask_dict = {}

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        dev = pl_module.device
        module = pl_module.model if hasattr(pl_module, "model") else pl_module
        for name, layer in module.named_modules():
            if "proj" in name:
                self.mask_dict[name] = (torch.abs(layer.weight.detach()) > 0).to(dev)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
    ) -> None:
        module = pl_module.model if hasattr(pl_module, "model") else pl_module
        for name, layer in module.named_modules():
            if "proj" in name:
                state_dict = layer.state_dict()
                state_dict["weight"] *= self.mask_dict[name]
                layer.load_state_dict(state_dict)
