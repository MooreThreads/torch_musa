""" Test complex fill operator. """

import pytest
import torch


real_dtypes = [torch.float16, torch.float32, torch.float64]
complex_dtypes = [torch.complex64, torch.complex128]
DEVICES = "musa"


fill_inputs = [
    torch.randn(8),
    torch.randn(3, 5),
    torch.randn(2, 4, 8),
    torch.randn(2, 4, 6, 8),
]

fill_value = [2, torch.tensor(1.5)]


@pytest.mark.parametrize("cdtype", complex_dtypes)
@pytest.mark.parametrize("dtype", complex_dtypes + real_dtypes)
@pytest.mark.parametrize("input_data", fill_inputs)
@pytest.mark.parametrize("value", fill_value)
def test_complex_fill(cdtype, dtype, input_data, value):
    """
    This unit test verifies the correctness and stability of the in-place fill
    operation on complex tensors under various dtype configurations.
    """
    input_data = input_data.to(cdtype).to(DEVICES)
    orig_shape = input_data.shape
    orig_stride = input_data.stride()
    orig_numel = input_data.numel()
    orig_ptr = input_data.data_ptr()
    orig_contig = input_data.is_contiguous()

    if torch.is_tensor(value):
        v = value.to(dtype).to(DEVICES)
        v_compare = value.item()
    else:
        v = value
        v_compare = value

    input_data.fill_(v)

    fill_shape = input_data.shape
    fill_stride = input_data.stride()
    fill_numel = input_data.numel()
    fill_ptr = input_data.data_ptr()
    fill_contig = input_data.is_contiguous()

    assert fill_shape == orig_shape
    assert fill_stride == orig_stride
    assert fill_numel == orig_numel
    assert fill_ptr == orig_ptr
    assert fill_contig == orig_contig
    assert input_data.dtype == cdtype
    assert input_data.device.type == DEVICES

    assert torch.allclose(
        input_data.cpu(), torch.full(orig_shape, v_compare, dtype=cdtype)
    )


fill_uncontig_cases = [
    {
        "full_shape": [128, 128],
        "uncontig_f": [
            lambda t: t[::2, :],
            lambda t: t[:, :80:2],
        ],
    },
    {
        "full_shape": [1024],
        "uncontig_f": [
            lambda t: t[:512:2],
            lambda t: t[500::2],
        ],
    },
]


@pytest.mark.parametrize("cdtype", complex_dtypes)
@pytest.mark.parametrize("dtype", complex_dtypes + real_dtypes)
@pytest.mark.parametrize("input_case", fill_uncontig_cases)
@pytest.mark.parametrize("value", fill_value)
def test_complex_fill_uncontig(cdtype, dtype, input_case, value):
    """
    Check uncontig cases.
    """
    shape, funcs = input_case["full_shape"], input_case["uncontig_f"]

    if torch.is_tensor(value):
        v_cpu = value.to(dtype)
        v_musa = v_cpu.to(DEVICES)
    else:
        v_cpu = value
        v_musa = value

    for f in funcs:
        dense_cpu = torch.randn(shape).to(cdtype)
        dense_musa = dense_cpu.to(DEVICES)

        sparse_cpu = f(dense_cpu)
        sparse_musa = f(dense_musa)
        assert not sparse_cpu.is_contiguous()
        assert not sparse_musa.is_contiguous()

        sparse_cpu.fill_(v_cpu)
        sparse_musa.fill_(v_musa)

        target = torch.view_as_real(sparse_musa).cpu()
        assert torch.allclose(torch.view_as_complex(target), sparse_cpu)
