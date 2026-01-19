"""Test elemwise operators for complex dtypes."""

# pylint: disable=C0116, W0611
import pytest
import torch
import torch_musa
from torch_musa import testing


@pytest.mark.parametrize(
    "case",
    [
        {
            "self": torch.randn(1024).uniform_(-1, 1),
            "other": 1,
        },
        {
            "self": torch.randn(8, 128).uniform_(-1, 1),
            "other": 1,
            "alpha": 2,
        },
        {
            "self": torch.randn(1200).uniform_(-1, 1),
            "other": -1,
        },
        {
            "self": torch.randn(12, 100).uniform_(-1, 1),
            "other": -1,
            "alpha": 2,
        },
        {
            "self": torch.randn(1024).uniform_(-1, 1),
            "other": torch.ones(1),
        },
        {
            "self": torch.randn(1024).uniform_(-1, 1),
            "other": torch.randn(1024).uniform_(-1, 1),
            "alpha": 2,
        },
        {
            "self": torch.ones(1),
            "other": torch.randn(12, 100).uniform_(-1, 1),
        },
        {
            "self": torch.randn(1200).uniform_(-1, 1),
            "other": torch.randn(1200).uniform_(-1, 1),
            "alpha": 2,
        },
    ],
)
@pytest.mark.parametrize("func", [torch.add, torch.sub])
@pytest.mark.parametrize(
    "dtypes",
    [
        [torch.complex64, torch.float],
    ],
)
def test_complex_add_sub_tensor_contig(case, func, dtypes):
    c_t, f_t = dtypes
    self_c, other_c = case["self"], case["other"]

    if isinstance(self_c, torch.Tensor):
        self_c = self_c.to(f_t)
        self_m = self_c.musa()
    else:
        self_m = self_c

    if isinstance(other_c, torch.Tensor):
        other_c = other_c.to(f_t)
        other_m = other_c.musa()
    else:
        other_m = other_c

    if isinstance(self_c, torch.Tensor) and self_c.numel() != 1:
        self_c = self_c.view(c_t)
        self_m = self_m.view(c_t)

    if isinstance(other_c, torch.Tensor) and other_c.numel() != 1:
        other_c = other_c.view(c_t)
        other_m = other_m.view(c_t)

    alpha = case.get("alpha", 1)

    res_c = func(self_c, other_c, alpha=alpha)
    res_m = func(self_m, other_m, alpha=alpha)

    cmp = testing.DefaultComparator()

    def check(c, m):
        assert c.shape == m.shape
        c = c.view(f_t)
        m = m.view(f_t).cpu()
        assert cmp(c, m)

    check(res_c, res_m)

    res_c = res_c.view(f_t).zero_().view(c_t)
    res_m = res_m.view(f_t).zero_().view(c_t)

    func(self_c, other_c, alpha=alpha, out=res_c)
    func(self_m, other_m, alpha=alpha, out=res_m)
    check(res_c, res_m)


@pytest.mark.parametrize(
    "case",
    [
        {
            "self": torch.randn(1024).uniform_(-1, 1),
            "other": 2,
        },
        {
            "self": -2.5,
            "other": torch.randn(12, 100).uniform_(-1, 1),
        },
        {
            "self": torch.randn(8, 1024).uniform_(-1, 1),
            "other": torch.tensor([-2]),
        },
        {
            "self": torch.randn(1024).uniform_(-1, 1),
            "other": torch.randn(1024).uniform_(-1, 1),
        },
        {
            "self": torch.tensor([2.5]),
            "other": torch.randn(12, 100).uniform_(-1, 1),
        },
        {
            "self": torch.randn(1200).uniform_(-1, 1),
            "other": torch.randn(1200).uniform_(-1, 1),
        },
    ],
)
@pytest.mark.parametrize("func", [torch.mul, torch.div])
@pytest.mark.parametrize(
    "dtypes",
    [
        [torch.complex64, torch.float],
    ],
)
def test_complex_mul_div_contig(case, func, dtypes):
    c_t, f_t = dtypes
    self_c, other_c = case["self"], case["other"]

    if isinstance(self_c, torch.Tensor):
        self_c = self_c.to(f_t)
        self_m = self_c.musa()
    else:
        self_m = self_c

    if isinstance(other_c, torch.Tensor):
        other_c = other_c.to(f_t)
        other_m = other_c.musa()
    else:
        other_m = other_c

    if isinstance(self_c, torch.Tensor) and self_c.numel() != 1:
        self_c = self_c.view(c_t)
        self_m = self_m.view(c_t)

    if isinstance(other_c, torch.Tensor) and other_c.numel() != 1:
        other_c = other_c.view(c_t)
        other_m = other_m.view(c_t)

    res_c = func(self_c, other_c)
    res_m = func(self_m, other_m)

    cmp = testing.DefaultComparator(abs_diff=1e-6)

    def check(c, m):
        assert c.shape == m.shape
        c = c.view(f_t)
        m = m.view(f_t).cpu()
        assert cmp(c, m)

    check(res_c, res_m)

    res_c = res_c.view(f_t).zero_().view(c_t)
    res_m = res_m.view(f_t).zero_().view(c_t)

    func(self_c, other_c, out=res_c)
    func(self_m, other_m, out=res_m)
    check(res_c, res_m)


@pytest.mark.parametrize(
    "case",
    [
        {
            "self": [
                torch.randn(1024 * 2).uniform_(-1, 1),
                lambda t: t[:512:2],
            ],
            "other": [1, lambda t: t],
        },
        {
            "self": [1, lambda t: t],
            "other": [
                torch.randn(8, 128 * 2).uniform_(-1, 1),
                lambda t: t[:, :80:2],
            ],
            "alpha": 2,
        },
        {
            "self": [torch.ones(1), lambda t: t],
            "other": [
                torch.randn(1024 * 2).uniform_(-1, 1),
                lambda t: t[500::2],
            ],
            "alpha": -2,
        },
        {
            "self": [
                torch.randn(8, 128 * 2).uniform_(-1, 1),
                lambda t: t[::2, :],
            ],
            "other": [torch.ones(1), lambda t: t],
        },
        {
            "self": [
                torch.randn(128, 128 * 2).uniform_(-1, 1),
                lambda t: t[::2, 1::2],
            ],
            "other": [
                torch.randn(128, 128 * 2).uniform_(-1, 1),
                lambda t: t[1::2, ::2],
            ],
        },
        {
            "self": [
                torch.randn(128, 128 * 2).uniform_(-1, 1),
                lambda t: t[::2, :],
            ],
            "other": [
                torch.randn(64, 128 * 2).uniform_(-1, 1),
                lambda t: t,
            ],
        },
    ],
)
@pytest.mark.parametrize("func", [torch.add, torch.sub])
@pytest.mark.parametrize(
    "dtypes",
    [
        [torch.complex64, torch.float],
    ],
)
def test_complex_add_sub_tensor_uncontig(case, func, dtypes):
    c_t, f_t = dtypes
    self, other = case["self"], case["other"]

    self_dense_cpu, self_f = self
    if isinstance(self_dense_cpu, torch.Tensor) and self_dense_cpu.numel() != 1:
        self_dense_cpu = self_dense_cpu.to(f_t).view(c_t)
        self_dense_musa = self_dense_cpu.musa()
    elif isinstance(self_dense_cpu, torch.Tensor):
        self_dense_musa = self_dense_cpu.musa()
    else:
        self_dense_musa = self_dense_cpu
    self_sparse_cpu = self_f(self_dense_cpu)
    self_sparse_musa = self_f(self_dense_musa)

    other_dense_cpu, other_f = other
    if isinstance(other_dense_cpu, torch.Tensor) and other_dense_cpu.numel() != 1:
        other_dense_cpu = other_dense_cpu.to(f_t).view(c_t)
        other_dense_musa = other_dense_cpu.musa()
    elif isinstance(other_dense_cpu, torch.Tensor):
        other_dense_musa = other_dense_cpu.musa()
    else:
        other_dense_musa = other_dense_cpu
    other_sparse_cpu = other_f(other_dense_cpu)
    other_sparse_musa = other_f(other_dense_musa)

    alpha = case.get("alpha", 1)

    res_c = func(self_sparse_cpu, other_sparse_cpu, alpha=alpha)
    res_m = func(self_sparse_musa, other_sparse_musa, alpha=alpha)

    cmp = testing.DefaultComparator()

    def check(c, m):
        assert c.shape == m.shape
        c = c.view(f_t)
        m = m.view(f_t).cpu()
        assert cmp(c, m)

    check(res_c, res_m)

    res_c = res_c.view(f_t).zero_().view(c_t)
    res_m = res_m.view(f_t).zero_().view(c_t)

    func(self_sparse_cpu, other_sparse_cpu, alpha=alpha, out=res_c)
    func(self_sparse_musa, other_sparse_musa, alpha=alpha, out=res_m)
    check(res_c, res_m)
