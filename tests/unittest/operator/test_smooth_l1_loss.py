#pylint: disable=redefined-builtin
"Smooth L1 Loss Test"
from itertools import product
import pytest
import torch.nn.functional as F
import torch
from torch_musa import testing

integral_types = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", [(2, 2), (64, 64)])
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
@pytest.mark.parametrize("beta", [0.5, 1.0, 1.5])
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_smoothl1loss(shape, dtype, beta, reduction):
    "Testing smooth l1 loss"
    def _make_test_tensor(shape, contiguous=True):
        if contiguous:
            test_tensor = torch.randn(shape).to(dtype=dtype)
        else:
            # Select every other element in the innermost dimension to
            # make it non-contiguous.
            doubled_shape = list(shape)
            doubled_shape[-1] *= 2
            test_tensor = torch.randn(doubled_shape).to(dtype=dtype)
            test_tensor = test_tensor[..., ::2]
        return test_tensor

    # init smooth l1 loss model
    smooth_l1 = torch.nn.SmoothL1Loss(beta=beta, reduction=reduction)

    # test contiguous input
    input = _make_test_tensor(shape)
    target = _make_test_tensor(shape)
    output_cpu = smooth_l1(input, target)
    output_musa = smooth_l1(input.to("musa"), target.to("musa"))
    testing.DefaultComparator(output_musa, output_cpu)

    # test uncontiguous input
    input = _make_test_tensor(shape, False)
    target = _make_test_tensor(shape, False)
    output_cpu = smooth_l1(input, target)
    output_musa = smooth_l1(input.to("musa"), target.to("musa"))
    testing.DefaultComparator(output_musa, output_cpu)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_smoothl1loss_intergral_target():
    "Testing smooth l1 loss with int target"
    def _input_grad(input, target, reduction):
        output = F.smooth_l1_loss(input, target, reduction=reduction, beta=0.5)
        output.sum().backward()
        return input.grad

    for dtype, reduction in product(integral_types, ("none", "sum", "mean")):
        input = torch.randn(2, 2, device="musa", requires_grad=True)
        target = torch.randint(0, 9, (2, 2), device="musa", dtype=dtype)

        input_grad_with_float_target = _input_grad(input, target.float(), reduction)

        input_grad = _input_grad(
            input.detach().clone().requires_grad_(True), target, reduction
        )
        testing.DefaultComparator(input_grad, input_grad_with_float_target)


def test_smoothl1loss_negative_beta_not_supported():
    "Testing smooth l1 loss with negative beta"
    with pytest.raises(
        RuntimeError, match="smooth_l1_loss does not support negative values for beta."
    ):
        F.smooth_l1_loss(
            torch.randn(2, 2).to("musa"), torch.randn(2, 2).to("musa"), beta=-1.0
        )
