"""Test softmax operators."""
# pylint: disable=missing-function-docstring, unused-import
import pytest
import torch
import torch_musa

from torch_musa import testing

configs = [
    # shape, dim
    [(1024,), 0],
    [(4, 256), 1],
    [(4, 256, 2), 0],
    [(4, 256, 2, 2), 1],
    [(5, 64, 30, 2, 3), 1],
    [(7, 1, 2048, 2, 3, 4), 2],
    [(2, 4, 8, 16, 32, 2, 2), 3],
    [(2, 3, 4, 5, 6, 7, 8, 1), 2]
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_fp32(config, functor):
    shape, dim = config
    if functor == torch.nn.LogSoftmax:
        abs_diff = 1e-5
    else:
        abs_diff = 1e-6
    test = testing.OpTest(func=functor,
                          input_args={"dim": dim},
                          comparators=testing.DefaultComparator(abs_diff=abs_diff))
    input_tensor = torch.randn(shape, dtype=torch.float32, requires_grad=True)
    test.check_result({"input": input_tensor}, train=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_fp16(config, functor):
    shape, dim = config
    test = testing.OpTest(func=functor,
                          input_args={"dim": dim},
                          comparators=testing.DefaultComparator(abs_diff=1e-8))
    input_tensor = torch.randn(shape, dtype=torch.float16).to(torch.float32).requires_grad_()
    test.check_musafp16_vs_musafp32({"input": input_tensor}, train=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("functor", [torch.nn.Softmax, torch.nn.LogSoftmax])
def test_softmax_bf16(config, functor):
    shape, dim = config
    if functor == torch.nn.LogSoftmax:
        abs_diff, rel_diff = 1e-1, 1e-1
    else:
        abs_diff, rel_diff = 1e-3, 1e-8
    test = testing.OpTest(func=functor,
                          input_args={"dim": dim},
                          comparators=testing.DefaultComparator(abs_diff=abs_diff,
                                                                rel_diff=rel_diff))
    input_tensor = torch.randn(shape, dtype=torch.bfloat16).requires_grad_()
    test.check_result({"input": input_tensor}, train=True)
