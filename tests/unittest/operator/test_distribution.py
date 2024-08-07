"""Testing distribution operators: uniform, normal, bernoulli"""

# pylint: disable=import-outside-toplevel, invalid-name

import torch
import pytest
from torch_musa import testing

float_dtypes = [torch.float32, torch.float16]
if testing.get_musa_arch() >= 22:
    float_dtypes.append(torch.bfloat16)


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
