"""Tests for internal ops newly wired in musa_functions.yaml."""

# pylint: disable=missing-function-docstring, redefined-outer-name
import torch

from torch_musa import testing


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_compute_linear_combination_basic():
    # coefficients: [m, n], input: [n, ...] => output: [m, ...]
    # simplest case: input is 1D
    n = 5
    m = 3
    inp = torch.randn(n, dtype=torch.float32)  # [n]
    coeff = torch.randn(m, n, dtype=torch.float32)  # [m, n]

    # expected: out[i] = sum_j coeff[i, j] * inp[j]
    expected = coeff @ inp

    cpu_res = torch.ops.aten._compute_linear_combination.default(inp, coeff)
    musa_res = torch.ops.aten._compute_linear_combination.default(
        inp.to("musa"), coeff.to("musa")
    ).cpu()

    comparator = testing.DefaultComparator(abs_diff=1e-6, rel_diff=1e-5)
    assert comparator(expected, cpu_res)
    assert comparator(cpu_res, musa_res)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_fused_adagrad_scalar_lr():
    # single-parameter fused adagrad with scalar lr
    p = torch.nn.Parameter(torch.randn(4, 4, dtype=torch.float32))
    g = torch.randn_like(p)
    state_sum = torch.zeros_like(p)
    # step is a scalar tensor per param
    state_step = torch.zeros((), dtype=torch.int64)

    p_cpu = p.clone().detach()
    g_cpu = g.clone().detach()
    s_cpu = state_sum.clone().detach()
    step_cpu = state_step.clone().detach()

    p_musa = p.clone().detach().to("musa")
    g_musa = g.clone().detach().to("musa")
    s_musa = state_sum.clone().detach().to("musa")
    step_musa = state_step.clone().detach().to("musa")

    lr = 0.1
    lr_decay = 0.0
    weight_decay = 0.0
    eps = 1e-8
    maximize = False

    torch.ops.aten._fused_adagrad_(
        [p_cpu],
        [g_cpu],
        [s_cpu],
        [step_cpu],
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=None,
        found_inf=None,
    )

    torch.ops.aten._fused_adagrad_(
        [p_musa],
        [g_musa],
        [s_musa],
        [step_musa],
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=None,
        found_inf=None,
    )

    comparator = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-3)
    assert comparator(p_cpu, p_musa.cpu())
    assert comparator(s_cpu, s_musa.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_fused_adagrad_tensor_lr():
    # single-parameter fused adagrad with tensor lr
    p = torch.nn.Parameter(torch.randn(8, dtype=torch.float32))
    g = torch.randn_like(p)
    state_sum = torch.zeros_like(p)
    state_step = torch.zeros((), dtype=torch.int64)

    p_cpu = p.clone().detach()
    g_cpu = g.clone().detach()
    s_cpu = state_sum.clone().detach()
    step_cpu = state_step.clone().detach()

    p_musa = p.clone().detach().to("musa")
    g_musa = g.clone().detach().to("musa")
    s_musa = state_sum.clone().detach().to("musa")
    step_musa = state_step.clone().detach().to("musa")

    lr = torch.tensor(0.05, dtype=torch.float32)
    lr_decay = 0.1
    weight_decay = 0.01
    eps = 1e-8
    maximize = False

    torch.ops.aten._fused_adagrad_.tensor_lr(
        [p_cpu],
        [g_cpu],
        [s_cpu],
        [step_cpu],
        lr=lr,
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=None,
        found_inf=None,
    )

    torch.ops.aten._fused_adagrad_.tensor_lr(
        [p_musa],
        [g_musa],
        [s_musa],
        [step_musa],
        lr=lr.to("musa"),
        lr_decay=lr_decay,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        grad_scale=None,
        found_inf=None,
    )

    comparator = testing.DefaultComparator(abs_diff=1e-4, rel_diff=1e-3)
    assert comparator(p_cpu, p_musa.cpu())
    assert comparator(s_cpu, s_musa.cpu())
