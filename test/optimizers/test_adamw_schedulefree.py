import copy

import numpy as np
import pytest
import torch

from fmchisel.optimizers.adamw_schedulefree import AdamWScheduleFree

_DTYPE_TO_TOLERANCE = {
    torch.float16: 1e-3,
    torch.float32: 1e-6,
}

_TORCH_TO_NP_DTYPE = {
    torch.float16: np.float16,
    torch.float32: np.float32,
}


def adam_sf_update_numpy(
    y,
    g_t,
    v,
    t,
    T_warmup,
    z,
    x,
    beta_1,
    beta_2,
    lr=0.001,
    wd=0.0,
    epsilon=1e-6,
    lr_weight_sum=0.0,
):
    # this is an exact match of the algorithm in the ScheduleFree paper:
    # https://arxiv.org/pdf/2405.15682 (Algorithm 1)
    v_t = beta_2 * v + (1 - beta_2) * (g_t**2)
    v_hat = v_t / (1 - beta_2**t)
    lr_t = lr * min(1, t / T_warmup) if T_warmup > 0 else lr
    update = g_t / (np.sqrt(v_hat) + epsilon) + wd * y
    z_t = z - lr_t * update
    lr_weight_sum_t = lr_weight_sum + lr_t**2
    c_t = lr_t**2 / lr_weight_sum_t if lr_weight_sum_t > 0 else 0.0
    x_t = (1 - c_t) * x + c_t * z_t
    y_t = (1 - beta_1) * z_t + beta_1 * x_t
    return [x_t, y_t, z_t, v_t, lr_weight_sum_t]


@pytest.mark.parametrize(
    "dtype, T_warmup, weight_decay",
    [
        (torch.float32, 0, 0.0),
        (torch.float32, 0, 0.1),
        (torch.float32, 5, 0.0),
        (torch.float32, 5, 0.1),
        (torch.float16, 0, 0.0),
        (torch.float16, 0, 0.1),
        (torch.float16, 5, 0.0),
        (torch.float16, 5, 0.1),
    ],
)
def test_dense(dtype, T_warmup, weight_decay):
    var0_np = np.array([1.0, 2.0], dtype=_TORCH_TO_NP_DTYPE[dtype])
    grad0_np = np.array([0.1, 0.1], dtype=_TORCH_TO_NP_DTYPE[dtype])
    v0_np = np.array([0.0, 0.0], dtype=_TORCH_TO_NP_DTYPE[dtype])
    z0_np = copy.deepcopy(var0_np)
    x0_np = copy.deepcopy(var0_np)

    var0 = torch.tensor(var0_np, dtype=dtype, requires_grad=True)

    learning_rate = 0.01
    beta_1 = 0.9
    beta_2 = 0.999
    epsilon = 1e-7
    lr_weight_sum = np.float32(0.0).astype(_TORCH_TO_NP_DTYPE[dtype])

    optim = AdamWScheduleFree(
        [var0],
        lr=learning_rate,
        betas=(beta_1, beta_2),
        eps=epsilon,
        warmup_steps=T_warmup,
        weight_decay=weight_decay,
    )
    for t in range(1, 11):
        optim.zero_grad()
        var0.grad = torch.tensor(grad0_np, dtype=dtype)
        optim.step()
        print(var0)

        [x0_np, var0_np, z0_np, v0_np, lr_weight_sum] = adam_sf_update_numpy(
            y=var0_np,
            g_t=grad0_np,
            v=v0_np,
            t=t,
            T_warmup=T_warmup,
            z=z0_np,
            x=x0_np,
            beta_1=beta_1,
            beta_2=beta_2,
            lr=learning_rate,
            wd=weight_decay,
            epsilon=epsilon,
            lr_weight_sum=lr_weight_sum,
        )
        print(var0_np)
    np.testing.assert_allclose(
        var0_np,
        var0.detach().numpy(),
        rtol=_DTYPE_TO_TOLERANCE[dtype],
        atol=_DTYPE_TO_TOLERANCE[dtype],
    )
    np.testing.assert_allclose(
        z0_np,
        optim.state[var0]["z"].numpy(),
        rtol=_DTYPE_TO_TOLERANCE[dtype],
        atol=_DTYPE_TO_TOLERANCE[dtype],
    )
    np.testing.assert_allclose(
        v0_np,
        optim.state[var0]["exp_avg_sq"].numpy(),
        rtol=_DTYPE_TO_TOLERANCE[dtype],
        atol=_DTYPE_TO_TOLERANCE[dtype],
    )
    # # test switching mode
    optim.eval()
    np.testing.assert_allclose(
        x0_np,
        var0.detach().numpy(),
        rtol=_DTYPE_TO_TOLERANCE[dtype],
        atol=_DTYPE_TO_TOLERANCE[dtype],
    )
    optim.train()
    np.testing.assert_allclose(
        var0_np,
        var0.detach().numpy(),
        rtol=_DTYPE_TO_TOLERANCE[dtype],
        atol=_DTYPE_TO_TOLERANCE[dtype],
    )
