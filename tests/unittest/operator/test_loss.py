"""Test loss operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import random
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data = testing.get_raw_data()
# dtype of input tensor of mse_loss only support Float32 in muDNN now.
support_dtypes = [torch.float32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data)
@pytest.mark.parametrize("dtype", support_dtypes)
def test_mse_loss(input_data, dtype):
    if input_data.numel() == 0:
        return
    input_data = input_data.to(dtype)
    target_data = torch.rand_like(input_data)
    loss = torch.nn.MSELoss()
    output_cpu = loss(input_data, target_data)
    output_musa = loss(input_data.to("musa"), target_data.to("musa"))
    assert pytest.approx(output_cpu, 1e-6) == output_musa.to("cpu")


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "config",
    [
        # input_shape, target_shape, ignore_index
        [(20,), (1,), -100],
        [(16, 20), (16,), -100],
        [(3, 20, 224, 224), (3, 224, 224), -100],
        [(3, 20, 224, 224), (3, 224, 224), 255],
        [(3, 20, 224, 224), (3, 20, 224, 224)],
    ],
)
# only test torch.float32 cause `log_softmax*/softmax*` not implemented for `Half` on CPU kernel
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
    ],
)
def test_cross_entropy_loss(config, dtype):
    input_data = torch.randn(config[0]).to(dtype)
    if len(config[0]) == len(config[1]) and len(config[0]) != 1:
        # probabilities for each class
        target_data = torch.randn(config[1]).softmax(dim=1).to(dtype)
    else:
        class_dim = 0 if len(config[0]) == 1 else 1
        if class_dim == 0:
            target_data = torch.tensor(random.randint(0, config[0][class_dim] - 1))
        else:
            target_data = torch.empty(config[1], dtype=torch.long).random_(
                config[0][class_dim]
            )
        if len(config) > 2 and config[2] > 0:
            # ignored index will not contribute to the gradient
            mask = torch.rand(config[1]) > 0.5
            target_data[mask] = config[2]
    ce_loss = torch.nn.CrossEntropyLoss
    # use default parameters except `ignore_index`
    ce_loss_args = {
        "ignore_index": config[2] if len(config) > 2 else -100,
    }
    test = testing.OpTest(
        func=ce_loss,
        input_args=ce_loss_args,
        comparators=testing.DefaultComparator(abs_diff=1e-6),
    )
    input_data.requires_grad_(True)
    test.check_result({"input": input_data, "target": target_data}, train=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_nll_loss2d():
    input_data = torch.randn(3, 20, 224, 224, requires_grad=True)
    target_data = torch.randint(0, 19, (3, 224, 224), dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()
    output_cpu = loss(input_data, target_data)
    musa_data = torch.tensor(
        input_data.detach().numpy(), requires_grad=True, device="musa"
    )
    musa_target = torch.tensor(target_data.detach().numpy(), device="musa")
    output_musa = loss(musa_data, musa_target)
    output_cpu.backward()
    output_musa.backward()
    assert pytest.approx(output_cpu.detach(), 1e-6) == output_musa.detach().to("cpu")
    assert pytest.approx(input_data.grad, 1e-6) == musa_data.grad.to("cpu")


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_nll_loss_fp32():
    input_data = torch.randn(3, 20, requires_grad=True)
    target_data = torch.randint(0, 19, (3,), dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()
    output_cpu = loss(input_data, target_data)
    musa_data = torch.tensor(
        input_data.detach().numpy(), requires_grad=True, device="musa"
    )
    musa_target = torch.tensor(target_data.detach().numpy(), device="musa")
    output_musa = loss(musa_data, musa_target)
    output_cpu.backward()
    output_musa.backward()
    assert pytest.approx(output_cpu.detach(), 1e-6) == output_musa.detach().to("cpu")
    assert pytest.approx(input_data.grad, 1e-6) == musa_data.grad.to("cpu")


def test_nll_loss_fp16():
    input_data = torch.randn(3, 20, dtype=torch.float32, requires_grad=True)
    target_data = torch.randint(0, 19, (3,), dtype=torch.int64)
    loss = torch.nn.CrossEntropyLoss()
    output_cpu = loss(input_data, target_data)
    musa_data = torch.tensor(
        input_data.half().detach().numpy(), requires_grad=True, device="musa"
    )
    musa_target = torch.tensor(target_data.detach().numpy(), device="musa")
    output_musa = loss(musa_data, musa_target)
    output_cpu.backward()
    output_musa.backward()
    assert pytest.approx(
        output_cpu.float().detach(), 1e-3
    ) == output_musa.float().detach().to("cpu")
    assert pytest.approx(input_data.grad, 1e-2) == musa_data.grad.float().to("cpu")


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_binary_cross_entropy():
    input_data = torch.randn(3, 20, 224, 224, requires_grad=True)
    target_data = torch.randint(
        0, 2, (3, 20, 224, 224), requires_grad=False, dtype=input_data.dtype
    )
    output_cpu = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(input_data), target_data
    )
    musa_data = torch.tensor(
        input_data.detach().numpy(), requires_grad=True, device="musa"
    )
    musa_target = torch.tensor(target_data.detach().numpy(), device="musa")
    output_musa = torch.nn.functional.binary_cross_entropy(
        torch.sigmoid(musa_data), musa_target
    )
    output_cpu.backward()
    output_musa.backward()
    assert pytest.approx(output_cpu.detach(), 1e-6) == output_musa.detach().to("cpu")
    assert pytest.approx(input_data.grad, 1e-6) == musa_data.grad.to("cpu")
