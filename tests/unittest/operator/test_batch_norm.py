"""Test batch_norm operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import,not-callable, C0301
import torch
import pytest
import torch_musa

from torch_musa import testing

input_data_2d = [
    torch.randn(4, 100, 4, 4),
    torch.randn(64, 100, 16, 16),
    torch.randn(256, 0, 16, 16),
    torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
    torch.randn(256, 100, 16, 16).to(memory_format=torch.channels_last),
    torch.randn(256, 0, 16, 16).to(memory_format=torch.channels_last),
]

input_data_1d = [
    torch.randn(4, 100),
    torch.randn(256, 100),
]

input_data_3d = [
    torch.randn(4, 100, 35, 45, 10),
    torch.randn(256, 100, 55, 65, 20),
]

inputs = [
    (torch.nn.BatchNorm1d, input_data_1d),
    (torch.nn.BatchNorm2d, input_data_2d),
    (torch.nn.BatchNorm3d, input_data_3d),
]

train = [True, False]
affine = [True]

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("train", train)
@pytest.mark.parametrize("affine", affine)
def test_batch_norm(inputs, train, affine):
    torch_op = inputs[0]
    input_data_list = inputs[1]
    for input_data in input_data_list:
        input_data.requires_grad_()
        m = torch_op(100, affine=affine)
        m.train(train)
        output = m(input_data)
        output_musa = m.to("musa")(input_data.to("musa"))
        assert testing.DefaultComparator(abs_diff=1e-5)(output, output_musa.cpu())
        assert output.grad_fn.__class__ == output_musa.grad_fn.__class__


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="fp16 batch_norm supported in QY2 or later",
)
@pytest.mark.parametrize("train", train)
@pytest.mark.parametrize("affine", affine)
def test_batch_norm_2d_fp16(inputs, train, affine):
    torch_op = inputs[0]
    input_data_list = inputs[1]
    for input_data in input_data_list:
        input_data.requires_grad_()
        m = torch_op(100, affine=affine)
        m.train(train)
        output = m(input_data)
        m.half()
        input_data = input_data.half()
        output_musa = m.to("musa")(input_data.to("musa"))
        assert testing.DefaultComparator(abs_diff=1e-2)(
            output, output_musa.cpu().float()
        )
        assert output.grad_fn.__class__ == output_musa.grad_fn.__class__


input_data_2d = [
    torch.randn(4, 100, 4, 4),
    torch.randn(64, 100, 16, 16),
    torch.randn(0, 100, 0, 16),
    torch.randn(4, 100, 4, 0).to(memory_format=torch.channels_last),
    torch.randn(8, 100, 8, 8).to(memory_format=torch.channels_last),
    torch.randn(16, 100, 16, 16).to(memory_format=torch.channels_last),
]

input_data_3d = [
    torch.randn(4, 100, 4, 4, 4),
    torch.randn(8, 100, 8, 8, 8),
    torch.randn(16, 100, 1, 1, 1),
]

input_data_1d = [
    torch.randn(4, 100),
    torch.randn(8, 100),
    torch.randn(16, 100),
]
inputs = [
    (torch.nn.BatchNorm1d, input_data_1d),
    (torch.nn.BatchNorm2d, input_data_2d),
    (torch.nn.BatchNorm3d, input_data_3d),
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("inputs", inputs)
@pytest.mark.parametrize("train", [True])
@pytest.mark.parametrize("affine", affine)
def test_batch_norm_bwd(inputs, train, affine):
    torch_op = inputs[0]
    input_data_list = inputs[1]
    for input_data in input_data_list:
        model = torch_op(100, affine=affine)
        musa_model = torch_op(100).to("musa")
        model.train(train)
        musa_model.train(train)
        output = model(input_data)
        output_musa = musa_model(input_data.to("musa"))
        output.sum().backward()
        output_musa.sum().backward()
        assert testing.DefaultComparator(abs_diff=1e-3)(
            model.weight.grad, musa_model.weight.grad.cpu()
        )
        assert testing.DefaultComparator(abs_diff=1e-3)(
            model.bias.grad, musa_model.bias.grad.cpu()
        )


input_data_legit_no_stats = [
    # [input, weight, bias]
    # 1d
    [torch.randn(256, 100), torch.randn(100), torch.randn(100)],
    [torch.randn(64, 3), torch.randn(3), torch.randn(3)],
    [torch.randn(4, 17), torch.randn(17), torch.randn(17)],
    # 2d
    [torch.randn(256, 100, 16, 16), torch.randn(100), torch.randn(100)],
    [torch.randn(64, 3, 16, 16), torch.randn(3), torch.randn(3)],
    [torch.randn(4, 17, 16, 16), torch.randn(17), torch.randn(17)],
    # 3d
    [torch.randn(256, 100, 16, 16, 45, 10), torch.randn(100), torch.randn(100)],
    [torch.randn(64, 3, 16, 16, 65, 20), torch.randn(3), torch.randn(3)],
    [torch.randn(4, 17, 16, 16, 65, 20), torch.randn(17), torch.randn(17)],
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_data_legit_no_stats)
def test_native_batch_norm_legit(input_data):
    x = input_data[0]
    weight = input_data[1]
    bias = input_data[2]

    output_musa, save_mean_musa, save_invstd_musa = (
        torch.ops.aten._native_batch_norm_legit(
            x.musa(), weight.musa(), bias.musa(), True, 0.1, 1e-5
        )
    )
    output_cpu, save_mean_cpu, save_invstd_cpu = (
        torch.ops.aten._native_batch_norm_legit(x, weight, bias, True, 0.1, 1e-5)
    )

    assert testing.DefaultComparator(abs_diff=1e-3)(
        output_cpu.float(), output_musa.cpu().float()
    )
    assert testing.DefaultComparator(abs_diff=1e-3)(
        save_mean_cpu.float(), save_mean_musa.cpu().float()
    )
    assert testing.DefaultComparator(abs_diff=1e-3)(
        save_invstd_cpu.float(), save_invstd_musa.cpu().float()
    )

    running_mean_cpu = torch.randn(weight.shape)
    running_var_cpu = torch.randn(weight.shape)
    running_mean_musa = running_mean_cpu.musa()
    running_var_musa = running_var_cpu.musa()

    output_musa, save_mean_musa, save_invstd_musa = (
        torch.ops.aten._native_batch_norm_legit(
            x.musa(),
            weight.musa(),
            bias.musa(),
            running_mean_musa,
            running_var_musa,
            True,
            0.1,
            1e-5,
        )
    )
    output_cpu, save_mean_musa, save_invstd_musa = (
        torch.ops.aten._native_batch_norm_legit(
            x, weight, bias, running_mean_cpu, running_var_cpu, True, 0.1, 1e-5
        )
    )
    assert testing.DefaultComparator(abs_diff=1e-3)(
        output_cpu.float(), output_musa.cpu().float()
    )
