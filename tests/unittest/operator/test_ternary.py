"""Test ternary operators."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, C3001, W0622
import random
import copy
import torch
import pytest
import torch_musa

from torch_musa import testing

input_datas = [
    {
        "input": torch.tensor(random.uniform(-10, 10)),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30),
    },
    {
        "input": torch.randn(30),
        "tensor1": torch.tensor(random.uniform(-10, 10)),
        "tensor2": torch.tensor(random.uniform(-10, 10)),
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30),
    },
    {
        "input": torch.randn(30, 1),
        "tensor1": torch.randn(1, 30),
        "tensor2": torch.randn(30, 1),
    },
    {
        "input": torch.randn(0, 0),
        "tensor1": torch.randn(0, 0),
        "tensor2": torch.randn(0, 0),
    },
    {
        "input": torch.tensor(random.uniform(-10, 10)),
        "tensor1": torch.randn(30, 30)[::2, ::2],
        "tensor2": torch.randn(60, 60)[::4, ::4],
    },
    {
        "input": torch.randn(30, 30)[::2, ::2],
        "tensor1": torch.randn(30, 30)[::2, ::2],
        "tensor2": torch.randn(60, 60)[::4, ::4],
    },
    {
        "input": torch.randn(30, 0),
        "tensor1": torch.randn(30, 0),
        "tensor2": torch.randn(60, 0)[::2, :],
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(1, 0, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(1, 0, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(1, 0, 30, 30).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(1, 3, 30, 30),
    },
    {
        "input": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(64, 64, 1, 1),
        "tensor2": torch.randn(64, 64, 1, 1),
    },
    {
        "input": torch.randn(64, 64, 1, 1),
        "tensor1": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(0, 64, 1, 1),
        "tensor1": torch.randn(0, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(0, 64, 1, 1).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(0, 64, 1, 0),
        "tensor1": torch.randn(0, 64, 1, 0).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(0, 64, 1, 0).to(memory_format=torch.channels_last),
    },
]

values = [-1, 0.5, 0, 0.5, 1]
for data in testing.get_raw_data():
    input_datas.append({"input": data, "tensor1": data, "tensor2": data})

all_support_types = testing.get_all_support_types()


def transform_dtype(dtype, value):
    if dtype in (torch.float32, torch.float16, torch.bfloat16):
        return float(value)
    if dtype in (torch.int32, torch.int64):
        return int(value)
    return dtype


all_support_types.extend([torch.float16])


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("value", values)
def test_addcmul(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": input_data["tensor2"].to(dtype),
        "value": transform_dtype(dtype, value),
    }
    test = testing.OpTest(func=torch.addcmul, input_args=input_dict)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    else:
        test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("value", values)
def test_addcmul_bf16(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": input_data["tensor2"].to(dtype),
        "value": transform_dtype(dtype, value),
    }
    test = testing.OpTest(func=torch.addcmul, input_args=input_dict)
    test.check_result()
    test.check_out_ops(bf16=True)
    test.check_grad_fn(bf16=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("value", values)
def test_addcdiv_fp32(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": torch.abs(input_data["tensor2"].to(dtype)) + 0.001,
        "value": transform_dtype(dtype, value),
    }
    comparator = testing.DefaultComparator(abs_diff=1e-5)
    test = testing.OpTest(
        func=torch.addcdiv, input_args=input_dict, comparators=comparator
    )
    test.check_result()
    test.check_out_ops()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_fp16(input_data, value):
    input_dict = {
        "input": input_data["input"].to(torch.float16).to(torch.float32),
        "tensor1": input_data["tensor1"].to(torch.float16).to(torch.float32),
        "tensor2": torch.abs(input_data["tensor2"].to(torch.float16).to(torch.float32))
        + 0.001,
        "value": transform_dtype(torch.float32, value),
    }
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    test = testing.OpTest(
        func=torch.addcdiv, input_args=input_dict, comparators=comparator
    )
    test.check_musafp16_vs_musafp32()
    test.check_out_ops(fp16=True)
    test.check_grad_fn(fp16=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_bf16(input_data, value):
    input_dict = {
        "input": input_data["input"].to(torch.bfloat16),
        "tensor1": input_data["tensor1"].to(torch.bfloat16),
        "tensor2": torch.abs(input_data["tensor2"].to(torch.bfloat16)) + 0.001,
        "value": transform_dtype(torch.float32, value),
    }
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    test = testing.OpTest(
        func=torch.addcdiv, input_args=input_dict, comparators=comparator
    )
    test.check_result()
    test.check_out_ops(bf16=True)
    test.check_grad_fn(bf16=True)


input_datas = [
    {
        "input": torch.randn(30).musa(),
        "tensor1": torch.randn(30).musa(),
        "tensor2": torch.randn(30).musa(),
    },
    {
        "input": torch.randn(30, 30)[::2, ::2].musa(),
        "tensor1": torch.randn(30, 30)[::2, ::2].musa(),
        "tensor2": torch.randn(60, 60)[::4, ::4].musa(),
    },
    {
        "input": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last).musa(),
        "tensor1": torch.randn(2, 3, 30, 30)
        .to(memory_format=torch.channels_last)
        .musa(),
        "tensor2": torch.randn(2, 3, 30, 30)
        .to(memory_format=torch.channels_last)
        .musa(),
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last).musa(),
        "tensor1": torch.randn(1, 3, 30, 30)
        .to(memory_format=torch.channels_last)
        .musa(),
        "tensor2": torch.randn(1, 3, 30, 30).musa(),
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last).musa(),
        "tensor1": torch.randn(1, 3, 30, 30).musa(),
        "tensor2": torch.randn(1, 3, 30, 30).musa(),
    },
    {
        "input": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last).musa(),
        "tensor1": torch.randn(64, 64, 1, 1).musa(),
        "tensor2": torch.randn(64, 64, 1, 1).musa(),
    },
    {
        "input": torch.randn(64, 64, 1, 1).musa(),
        "tensor1": torch.randn(64, 64, 1, 1)
        .to(memory_format=torch.channels_last)
        .musa(),
        "tensor2": torch.randn(64, 64, 1, 1)
        .to(memory_format=torch.channels_last)
        .musa(),
    },
]


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("value", values)
def test_addcdiv_in_place_fp32(input_data, dtype, value):
    comparator = testing.DefaultComparator(abs_diff=2e-4)
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": torch.abs(input_data["tensor2"].to(dtype)) + 0.01,
        "value": transform_dtype(dtype, value),
    }
    input_cpu = input_dict["input"].clone().cpu()
    tensor1_cpu = input_dict["tensor1"].clone().cpu()
    tensor2_cpu = input_dict["tensor2"].clone().cpu()
    ptr1 = input_dict["input"].data_ptr()
    ptr_t1_1 = input_dict["tensor1"].data_ptr()
    ptr_t2_1 = input_dict["tensor2"].data_ptr()

    input_dict["input"].addcdiv_(
        input_dict["tensor1"], input_dict["tensor2"], value=value
    )
    ptr2 = input_dict["input"].data_ptr()
    ptr_t1_2 = input_dict["tensor1"].data_ptr()
    ptr_t2_2 = input_dict["tensor2"].data_ptr()
    input_cpu.addcdiv_(tensor1_cpu, tensor2_cpu, value=value)
    res = comparator(input_cpu, input_dict["input"])
    info_str = ""
    if not res:
        atol, rtol, equal_nan = comparator.get_tolerance()
        mask_t = ~torch.isclose(
            input_dict["input"].cpu(), input_cpu, rtol, atol, equal_nan
        )
        selected = torch.abs(input_dict["input"][mask_t].cpu() - input_cpu[mask_t])
        info_str = f"Max abs error: {selected.max().item()}"
    assert res, info_str
    assert ptr1 == ptr2, "inplace operation should not change data address!"
    assert ptr_t1_1 == ptr_t1_2, "inplace operation should not change data address!"
    assert ptr_t2_1 == ptr_t2_2, "inplace operation should not change data address!"


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_fp16_in_place(input_data, value):
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    input_dict = {
        "input": input_data["input"].to(torch.float16),
        "tensor1": input_data["tensor1"].to(torch.float16),
        "tensor2": torch.abs(input_data["tensor2"].to(torch.float16)) + 0.1,
        "value": value,
    }

    input_cpu = input_dict["input"].clone().cpu().float()
    tensor1_cpu = input_dict["tensor1"].clone().cpu().float()
    tensor2_cpu = input_dict["tensor2"].clone().cpu().float()
    ptr1 = input_dict["input"].data_ptr()
    ptr_t1_1 = input_dict["tensor1"].data_ptr()
    ptr_t2_1 = input_dict["tensor2"].data_ptr()
    input_dict["input"].addcdiv_(
        input_dict["tensor1"], input_dict["tensor2"], value=value
    )
    ptr2 = input_dict["input"].data_ptr()
    ptr_t1_2 = input_dict["tensor1"].data_ptr()
    ptr_t2_2 = input_dict["tensor2"].data_ptr()
    input_cpu.addcdiv_(tensor1_cpu, tensor2_cpu, value=value)
    res = comparator(input_cpu, input_dict["input"].float())
    info_str = ""
    if not res:
        atol, rtol, equal_nan = comparator.get_tolerance()
        mask_t = ~torch.isclose(
            input_dict["input"].cpu().float(), input_cpu, rtol, atol, equal_nan
        )
        selected = torch.abs(
            input_dict["input"][mask_t].cpu().float() - input_cpu[mask_t]
        )
        info_str = f"Max abs error: {selected.max().item()}"
    assert res, info_str
    assert ptr1 == ptr2, "inplace operation should not change data address!"
    assert ptr_t1_1 == ptr_t1_2, "inplace operation should not change data address!"
    assert ptr_t2_1 == ptr_t2_2, "inplace operation should not change data address!"


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_bf16_in_place(input_data, value):
    input_dict = {
        "input": input_data["input"].to(torch.bfloat16),
        "tensor1": input_data["tensor1"].to(torch.bfloat16),
        "tensor2": torch.abs(input_data["tensor2"].to(torch.bfloat16)) + 0.1,
        "value": value,
    }
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    input_cpu = input_dict["input"].clone().cpu()
    tensor1_cpu = input_dict["tensor1"].clone().cpu()
    tensor2_cpu = input_dict["tensor2"].clone().cpu()
    ptr1 = input_dict["input"].data_ptr()
    ptr_t1_1 = input_dict["tensor1"].data_ptr()
    ptr_t2_1 = input_dict["tensor2"].data_ptr()
    input_dict["input"].addcdiv_(
        input_dict["tensor1"], input_dict["tensor2"], value=value
    )
    ptr2 = input_dict["input"].data_ptr()
    ptr_t1_2 = input_dict["tensor1"].data_ptr()
    ptr_t2_2 = input_dict["tensor2"].data_ptr()
    input_cpu.addcdiv_(tensor1_cpu, tensor2_cpu, value=value)
    res = comparator(input_cpu, input_dict["input"])
    info_str = ""
    if not res:
        atol, rtol, equal_nan = comparator.get_tolerance()
        mask_t = ~torch.isclose(
            input_dict["input"].cpu().float(), input_cpu, rtol, atol, equal_nan
        )
        selected = torch.abs(
            input_dict["input"][mask_t].cpu().float() - input_cpu[mask_t]
        )
        info_str = f"Max abs error: {selected.max().item()}"
    assert res, info_str
    assert ptr1 == ptr2, "inplace operation should not change data address!"
    assert ptr_t1_1 == ptr_t1_2, "inplace operation should not change data address!"
    assert ptr_t2_1 == ptr_t2_2, "inplace operation should not change data address!"


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("value", values)
def test_addcdiv_out(input_data, dtype, value):
    comparator = testing.DefaultComparator(abs_diff=2e-4)
    if dtype in (torch.float16, torch.bfloat16):
        comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": torch.abs(input_data["tensor2"].to(dtype)) + 0.1,
        "value": transform_dtype(dtype, value),
    }
    output_musa = torch.empty_like(input_dict["input"], device="musa")

    input_cpu = input_dict["input"].clone().cpu().float()
    tensor1_cpu = input_dict["tensor1"].clone().cpu().float()
    tensor2_cpu = input_dict["tensor2"].clone().cpu().float()
    output_cpu = torch.empty_like(input_cpu, device="cpu").float()

    ptr1 = input_dict["input"].data_ptr()
    ptr_t1_1 = input_dict["tensor1"].data_ptr()
    ptr_t2_1 = input_dict["tensor2"].data_ptr()
    ptr_output_1 = output_musa.data_ptr()

    torch.addcdiv(
        input_dict["input"],
        input_dict["tensor1"],
        input_dict["tensor2"],
        value=value,
        out=output_musa,
    )
    ptr2 = input_dict["input"].data_ptr()
    ptr_t1_2 = input_dict["tensor1"].data_ptr()
    ptr_t2_2 = input_dict["tensor2"].data_ptr()
    ptr_output_2 = output_musa.data_ptr()

    torch.addcdiv(input_cpu, tensor1_cpu, tensor2_cpu, value=value, out=output_cpu)
    output_musa = output_musa.float()
    res = comparator(output_cpu, output_musa)
    info_str = ""
    if not res:
        atol, rtol, equal_nan = comparator.get_tolerance()
        mask_t = ~torch.isclose(output_musa.cpu(), input_cpu, rtol, atol, equal_nan)
        selected = torch.abs(output_musa[mask_t].cpu() - input_cpu[mask_t])
        info_str = f"Max abs error: {selected.max().item()}"
    assert res, info_str
    assert ptr1 == ptr2, "out operation should not change data address!"
    assert ptr_t1_1 == ptr_t1_2, "out operation should not change data address!"
    assert ptr_t2_1 == ptr_t2_2, "out operation should not change data address!"
    assert ptr_output_1 == ptr_output_2, "out operation should not change data address!"


# where not support braodcast in muDNN
input_datas = testing.get_raw_data()
input_datas.append(torch.tensor(random.uniform(-10, 10)))


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_where_with_different_shape(data, dtype):
    input_args = {}
    input_args["condition"] = data > 0.5
    input_args["input"] = torch.tensor(1.0).to(dtype)
    input_args["other"] = torch.tensor(0.0).to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    else:
        test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_where_for_broadcast_feature(dtype):
    input_args = {}
    input_args["condition"] = torch.randn(1, 1, 3, 3) > 0
    input_args["input"] = torch.randn(1, 2, 3, 3).to(dtype)
    input_args["other"] = torch.randn(4, 1, 1, 3, 1).to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    else:
        test.check_result()
    test.check_grad_fn()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_where(data, dtype):
    input_args = {}
    input_args["condition"] = data > 0.5
    input_args["input"] = data.to(dtype)
    input_args["other"] = data.to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    if dtype == torch.float16:
        test.check_musafp16_vs_musafp32()
    else:
        test.check_result()
    test.check_grad_fn()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float16])
def test_where_bf16(data, dtype):
    input_args = {}
    input_args["condition"] = data > 0.5
    input_args["input"] = data.to(dtype)
    input_args["other"] = data.to(dtype)
    test = testing.OpTest(func=torch.where, input_args=input_args)
    test.check_result()
    test.check_grad_fn()


dtypes = [torch.float32, torch.half, torch.uint8, torch.int8, torch.int16, torch.int32]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape", [(3, 2), (200, 100)])
@pytest.mark.parametrize("dtype", dtypes)
def test_naive_where(dtype, shape):
    "Testing naive where : error with int64"
    input = torch.randint(-5, 5, size=shape).to(dtype)
    output_cpu = torch.where(input)
    output_musa = torch.where(input.to("musa"))
    comparator = lambda musa, cpu: (musa.cpu() == cpu).all()
    for musa, cpu in zip(output_musa, output_cpu):
        assert comparator(musa, cpu)


input_datas = [
    {
        "input": torch.randn(30, 30),
        "tensor1": torch.randn(30, 30),
        "tensor2": torch.randn(1, 30),
    },
    {
        "input": torch.randn(30),
        "tensor1": torch.tensor(random.uniform(-10, 10)),
        "tensor2": torch.tensor(random.uniform(-10, 10)),
    },
    {
        "input": torch.randn(30, 30),
        "tensor1": torch.randn(1, 30),
        "tensor2": torch.randn(30, 1),
    },
    {
        "input": torch.randn(0, 0),
        "tensor1": torch.randn(0, 0),
        "tensor2": torch.randn(0, 0),
    },
    {
        "input": torch.randn(30, 30)[::2, ::2],
        "tensor1": torch.randn(30, 30)[::2, ::2],
        "tensor2": torch.randn(60, 60)[::4, ::4],
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(2, 3, 30, 30).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(1, 3, 30, 30).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(1, 3, 30, 30),
    },
    {
        "input": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor1": torch.randn(64, 64, 1, 1),
        "tensor2": torch.randn(64, 64, 1, 1),
    },
    {
        "input": torch.randn(64, 64, 1, 1),
        "tensor1": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(64, 64, 1, 1).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(0, 64, 1, 1),
        "tensor1": torch.randn(0, 64, 1, 1).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(0, 64, 1, 1).to(memory_format=torch.channels_last),
    },
    {
        "input": torch.randn(0, 64, 1, 0),
        "tensor1": torch.randn(0, 64, 1, 0).to(memory_format=torch.channels_last),
        "tensor2": torch.randn(0, 64, 1, 0).to(memory_format=torch.channels_last),
    },
]

values = [-1, 0.5, 0, 0.5, 1]
for data in testing.get_raw_data():
    input_datas.append({"input": data, "tensor1": data, "tensor2": data})

all_support_types = testing.get_all_support_types()

all_support_types.extend([torch.float16])


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", all_support_types)
@pytest.mark.parametrize("value", values)
def test_addcmul_inplace(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": input_data["tensor2"].to(dtype),
        "value": transform_dtype(dtype, value),
    }
    inplace_input = copy.deepcopy(input_dict)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.addcmul.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    if dtype == torch.float16:
        test.check_res(cpu_to_fp32=True)
    else:
        test.check_res()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("value", values)
def test_addcmul_bf16_inplace(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": input_data["tensor2"].to(dtype),
        "value": transform_dtype(dtype, value),
    }
    inplace_input = copy.deepcopy(input_dict)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.addcmul.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("value", values)
def test_addcdiv_fp32_inplace(input_data, dtype, value):
    input_dict = {
        "input": input_data["input"].to(dtype),
        "tensor1": input_data["tensor1"].to(dtype),
        "tensor2": torch.abs(input_data["tensor2"].to(dtype)) + 0.001,
        "value": transform_dtype(dtype, value),
    }
    comparator = testing.DefaultComparator(abs_diff=1e-5)
    inplace_input = copy.deepcopy(input_dict)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.addcdiv.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
        comparators=[comparator],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_fp16_inplace(input_data, value):
    input_dict = {
        "input": input_data["input"]
        .to(torch.float16)
        .to(torch.float32)
        .to(torch.float16),
        "tensor1": input_data["tensor1"]
        .to(torch.float16)
        .to(torch.float32)
        .to(torch.float16),
        "tensor2": torch.abs(
            input_data["tensor2"].to(torch.float16).to(torch.float32).to(torch.float16)
        )
        + 0.001,
        "value": transform_dtype(torch.float32, value),
    }
    comparator = testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3)
    inplace_input = copy.deepcopy(input_dict)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.addcdiv.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
        comparators=[comparator],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(0)
@pytest.mark.parametrize("input_data", input_datas)
@pytest.mark.parametrize("value", values)
def test_addcdiv_bf16_inplace(input_data, value):
    input_dict = {
        "input": input_data["input"].to(torch.bfloat16),
        "tensor1": input_data["tensor1"].to(torch.bfloat16),
        "tensor2": torch.abs(input_data["tensor2"].to(torch.bfloat16)) + 0.001,
        "value": transform_dtype(torch.float32, value),
    }
    inplace_input = copy.deepcopy(input_dict)
    self_tensor = inplace_input["input"]
    inplace_input.pop("input")
    test = testing.InplaceOpChek(
        func_name=torch.addcdiv.__name__ + "_",
        self_tensor=self_tensor,
        input_args=inplace_input,
        comparators=[
            testing.DefaultComparator(abs_diff=5e-2, rel_diff=5e-3, equal_nan=True)
        ],
    )
    test.check_address()
    test.check_res(cpu_to_fp32=True)
