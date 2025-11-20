"""Testing distribution operators"""

# pylint: disable=import-outside-toplevel, invalid-name

import math
import numpy as np
import torch
from torch.utils._import_utils import _check_module_exists
import pytest
from torch_musa import testing

torch.manual_seed(42)
torch.musa.manual_seed(42)

DEVICE = "musa"
SHAPE = (8, 16)
CPU_VAL = torch.rand(SHAPE, dtype=torch.float32, device="cpu") * 3.0 + 0.1
CPU_VAL_OUTPUT = torch.rand(SHAPE, dtype=torch.float32, device="cpu")
CPU_TOTAL = CPU_VAL.sum(dim=-1, keepdim=True)

MUSA_VAL = CPU_VAL.to(DEVICE)
MUSA_VAL_OUTPUT = CPU_VAL_OUTPUT.to(DEVICE)
MUSA_TOTAL = CPU_TOTAL.to(DEVICE)

float_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


skip_if_no_scipy = pytest.mark.skipif(
    not _check_module_exists("scipy"), reason="test requires SciPy, but SciPy not found"
)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype_", float_dtypes)
@pytest.mark.parametrize("from_", [-42, 0, 4.2])
@pytest.mark.parametrize("to_", [-4.2, 0, 42])
def test_uniform_kstest(dtype_, from_, to_):
    """Testing uniform random generator by kstest"""
    from scipy import stats

    size = 1000
    if to_ > from_:
        t = torch.empty(size, dtype=dtype_, device="musa").uniform_(from_, to_)
        res = stats.kstest(
            t.cpu().to(torch.double), "uniform", args=(from_, (to_ - from_))
        )
        assert res.statistic < 0.1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
@pytest.mark.parametrize("mean", [-10, 0, 50])
@pytest.mark.parametrize("std", [1, 5, 10])
def test_normal_kstest(dtype, mean, std):
    """Testing normal random generator by kstest"""
    from scipy import stats

    size = 1000
    t = torch.empty(size, dtype=dtype, device="musa").normal_(mean=mean, std=std)
    res = stats.kstest(t.cpu().to(torch.double), "norm", args=(mean, std))
    assert res.statistic < 0.1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", testing.get_all_types())
def test_bernoulli_self(dtype):
    """
    Testing bernoulli.out and bernoulli_.float. If you compile torch_musa
    with MUSA_ARCH=10, beroulli_.float will run on CPU, so we don't test half.
    If you want to run bernoulli_.float on GPU, you should compile torch_musa
    with MUSA_ARCH >= 21 and EBABLE_COMPILE_FP64=ON.
    """

    def isBinary(t):
        return torch.ne(t, 0).mul_(torch.ne(t, 1)).sum().item() == 0

    t = torch.empty(10, 10, dtype=dtype, device="musa")

    t.fill_(2)
    t.bernoulli_(0.5)  # bernoulli_.float
    assert isBinary(t.cpu())

    for p_dtype in float_dtypes:
        p = torch.rand(10, dtype=p_dtype, device="musa").expand(10, 10)
        t.fill_(2)
        t.bernoulli_(p)
        assert isBinary(t.cpu())  # bernoulli_.Tensor

        t.fill_(2)
        torch.bernoulli(torch.rand_like(t, dtype=p_dtype), out=t)
        assert isBinary(t.cpu())

        t.fill_(2)
        t.bernoulli_(torch.rand_like(t, dtype=p_dtype))
        assert isBinary(t.cpu())


@skip_if_no_scipy
class TestDistribution:
    """Test distribution operators"""

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("dtype", testing.get_float_types())
    def test_lognormal_kstest(self, dtype):
        """test lognormal by kstest"""
        from scipy import stats

        size = 4096
        device = torch.musa.current_device()
        for mean in [-3, 0, 7]:
            for std in [1, 5, 7]:
                t = torch.empty(size, dtype=dtype, device=device).log_normal_(
                    mean=mean, std=std
                )
                res = stats.kstest(
                    t.cpu().to(torch.double), "lognorm", args=(std, 0, math.exp(mean))
                )
                if dtype == torch.half:
                    assert res.statistic < 0.3
                else:
                    assert res.statistic < 0.1

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("dtype", testing.get_float_types())
    def test_geometric_kstest(self, dtype):
        """test geometric by kstest"""
        from scipy import stats

        size = 1000
        device = torch.musa.current_device()
        torch.manual_seed(42)
        for p in [0.2, 0.8]:
            t = torch.empty(size, dtype=dtype, device=device).geometric_(p=p)
            actual = np.histogram(t.cpu().to(torch.double), np.arange(1, 100))[0]
            expected = stats.geom(p).pmf(np.arange(1, 99)) * size
            res = stats.chisquare(actual, expected)
            np.testing.assert_allclose(res.pvalue, 1.0, atol=0.1, rtol=0)

    @testing.test_on_nonzero_card_if_multiple_musa_device(1)
    @pytest.mark.parametrize("dtype", testing.get_float_types())
    def test_cauchy_kstest(self, dtype):
        """test cauchy by kstest"""
        from scipy import stats

        sizes = [(4096,), (5, 4096), (2, 3, 5000)]
        device = torch.musa.current_device()
        torch.manual_seed(42)
        for size in sizes:
            for median in [-10, 0, 50]:
                for sigma in [0.5, 1.0, 10.0]:
                    t = torch.empty(size, dtype=dtype, device=device).cauchy_(
                        median=median, sigma=sigma
                    )
                    t_double = t.cpu().to(torch.double)
                    for row in t_double.view(-1, t_double.shape[-1]):
                        res = stats.kstest(row, "cauchy", args=(median, sigma))
                        assert res.statistic < 0.1


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_normal_tensor_float(dtype):
    mean = torch.randn(128, dtype=dtype)
    std = 1.5
    cpu_out = torch.normal(mean, std)
    musa_out = torch.normal(mean.to("musa"), std)
    assert musa_out.shape == cpu_out.shape
    assert musa_out.dtype == mean.dtype
    assert abs(musa_out.mean().cpu() - cpu_out.mean()) < 0.5


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_normal_float_tensor(dtype):
    std = torch.rand(128, dtype=dtype)
    mean = 2.0
    cpu_out = torch.normal(mean, std)
    musa_out = torch.normal(mean, std.to("musa"))
    assert musa_out.shape == cpu_out.shape
    assert musa_out.dtype == std.dtype
    assert abs(musa_out.mean().cpu() - cpu_out.mean()) < 0.5


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_normal_tensor_tensor(dtype):
    mean = torch.randn(128, dtype=dtype)
    std = torch.rand(128, dtype=dtype) + 0.1
    cpu_out = torch.normal(mean, std)
    musa_out = torch.normal(mean.to("musa"), std.to("musa"))
    assert musa_out.shape == cpu_out.shape
    assert musa_out.dtype == mean.dtype
    assert abs(musa_out.mean().cpu() - cpu_out.mean()) < 0.5


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_normal_tensor_tensor_out(dtype):
    """Test normal distribution with tensor-tensor output."""
    mean = torch.randn(128, dtype=dtype)
    std = torch.rand(128, dtype=dtype) + 0.1
    out_cpu = torch.empty_like(mean)
    out_musa = torch.empty_like(mean.to("musa"))

    torch.normal(mean, std, out=out_cpu)
    torch.normal(mean.to("musa"), std.to("musa"), out=out_musa)

    assert out_cpu.shape == out_musa.shape
    assert abs(out_musa.mean().cpu() - out_cpu.mean()) < 0.5


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", float_dtypes)
def test_normal_float_tensor_out(dtype):
    """Test normal distribution with float-tensor output."""
    std = torch.rand(128, dtype=dtype) + 0.1
    mean = 1.5
    out_cpu = torch.empty_like(std)
    out_musa = torch.empty_like(std.to("musa"))

    torch.normal(mean, std, out=out_cpu)
    torch.normal(mean, std.to("musa"), out=out_musa)

    assert out_cpu.shape == out_musa.shape
    assert abs(out_musa.mean().cpu() - out_cpu.mean()) < 0.5


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test__standard_gamma_grad():
    cpu_grad = torch._standard_gamma_grad(CPU_VAL, CPU_VAL_OUTPUT)
    dev_grad = torch._standard_gamma_grad(MUSA_VAL, MUSA_VAL_OUTPUT).cpu()
    assert torch.allclose(cpu_grad, dev_grad, atol=1e-6, rtol=1e-5), \
        "_standard_gamma_grad max diff too large"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test__standard_gamma():
    dev_out = torch._standard_gamma(MUSA_VAL)
    assert torch.any(dev_out > 0), "musa _standard_gamma produced negative values"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test__dirichlet_grad():
    cpu_grad = torch._dirichlet_grad(CPU_VAL, CPU_VAL_OUTPUT, CPU_TOTAL)
    dev_grad = torch._dirichlet_grad(MUSA_VAL, MUSA_VAL_OUTPUT, MUSA_TOTAL).cpu()

    assert torch.allclose(cpu_grad, dev_grad, atol=1e-6, rtol=1e-5), \
        "torch._dirichlet_grad failed on musa"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test__sample_dirichlet():
    dev_out = torch._standard_gamma(MUSA_VAL)
    assert torch.any(dev_out > 0), "musa _sample_dirichlet produced negative values"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_poisson():
    dev_out = torch.poisson(MUSA_VAL)
    assert torch.any(dev_out > 0), "musa poisson produced negative values"


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_binomial():
    total_cpu = torch.randint(1, 10, SHAPE, dtype=torch.float, device="cpu")
    probs_cpu = torch.rand(SHAPE, dtype=torch.float, device="cpu")

    total_dev = total_cpu.to(DEVICE)
    probs_dev = probs_cpu.to(DEVICE)

    dev_out = torch.binomial(total_dev, probs_dev)
    assert torch.any(dev_out > 0), "musa binomial produced negative values"
