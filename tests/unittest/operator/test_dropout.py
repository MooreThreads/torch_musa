"""Test dropout operators."""
# pylint: disable=missing-function-docstring, unused-import, C0103
import torch
import numpy as np
import pytest

from torch_musa import testing
from torch_musa.testing import freeze_rng_state

p = [0, 0.2, 0.8, 1.0]
inplace = [False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "shape",
    [
        (1,),
        (128, 128),
        (20, 20, 20, 20),
        (4, 128, 20, 20, 2),
        (2, 2, 3, 4, 5, 6),
        (2, 3, 1, 8, 7, 6, 2),
        (2, 3, 1, 8, 7, 1, 6, 2),
        (0, 3, 1, 8, 7, 1, 6, 2),
        (0, 3, 1, 8, 7, 1, 0, 2),
        (0, 3, 1, 0, 7, 1, 6, 2),
    ],
)
@pytest.mark.parametrize("p_value", p)
@pytest.mark.parametrize("inplace_value", inplace)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("coefficient", [1, -1, 2, -3])
def test_dropout_train(shape, p_value, inplace_value, dtype, coefficient):
    if testing.get_musa_arch() < 22 and dtype == torch.bfloat16:
        return
    device = "musa"
    input_tensor = (
        (torch.rand(shape, dtype=dtype, device=device) + 0.1) * coefficient
    ).requires_grad_()
    module = torch.nn.Dropout(p=p_value, inplace=inplace_value)
    module.train()
    out = module(input_tensor)
    out.backward(torch.ones(input_tensor.shape, dtype=dtype, device=device))
    out_array = out.cpu()
    input_array = input_tensor.cpu()
    input_grad_array = input_tensor.grad.cpu()

    assert torch.count_nonzero(out_array) <= torch.count_nonzero(input_array)
    assert torch.count_nonzero(out_array) == torch.count_nonzero(input_grad_array)

    # Attention: The grad_fn of the output tensor of musa is different from both CPU and CUDA.
    # For musa, it is NativeDropoutBackward, while for both CUDA and CPU, it is MulBackward.

@pytest.mark.parametrize("shape", [(50, 2, 20, 20), (10, 4, 2, 2, 2)])
def test_dropoout_with_seed(shape):
    """
    Testing dropout for murand seed genertor
    """

    torch.musa.manual_seed_all(123)
    x = torch.randn(shape, device="musa")
    dropout = torch.nn.Dropout()
    with freeze_rng_state():
        y1 = dropout(x)
    with freeze_rng_state():
        y2 = dropout(x)

    assert torch.all(y1 == y2)

    torch.musa.manual_seed_all(321)
    with freeze_rng_state():
        y3 = dropout(x)

    assert torch.any(y1 != y3)


def test_empty_dropout():
    x = torch.tensor([]).musa()
    out = torch.nn.functional.dropout(x)
    assert out.size() == x.size()


def test_dropout_uncontiguous():
    x = torch.rand([10, 3, 5, 6], device="musa")[..., ::2]
    out = torch.nn.functional.dropout(x)
    assert torch.count_nonzero(out.cpu()) <= torch.count_nonzero(x.cpu())
