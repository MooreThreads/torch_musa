"""Tests for torch.tensor() consuming __cuda_array_interface__.

C++ only parses shape / typestr / data / strides. device and version
in the interface dict are ignored; device comes from torch.tensor(..., device=).
"""

# pylint: disable=missing-function-docstring, missing-class-docstring
import pytest
import torch
from torch_musa import testing


class FakeExternalArray:
    """Stand-in for warp / cupy / numba arrays. Keeps `tensor` alive for the pointer."""

    def __init__(
        self,
        tensor=None,
        *,
        shape=None,
        typestr="<f4",
        data_ptr=None,
        strides=None,
        read_only=False,
    ):
        self._tensor = tensor
        if tensor is not None:
            if shape is None:
                shape = tuple(tensor.shape)
            if data_ptr is None:
                data_ptr = tensor.data_ptr() if tensor.numel() > 0 else 0
        iface = {
            "shape": shape,
            "typestr": typestr,
            "data": (data_ptr, read_only),
        }
        if strides is not None:
            iface["strides"] = strides
        self.__cuda_array_interface__ = iface


@testing.skip_if_musa_unavailable
@pytest.mark.parametrize(
    "dtype,typestr,values",
    [
        (torch.float32, "<f4", [1.0, 2.0, 3.0]),
        (torch.int32, "<i4", [1, 2, 3]),
        (torch.float16, "<f2", [1.0, 2.0]),
    ],
)
def test_tensor_from_cai_explicit_device(dtype, typestr, values):
    source = torch.tensor(values, dtype=dtype, device="musa:0")
    ext = FakeExternalArray(source, typestr=typestr)

    result = torch.tensor(ext, device="musa:0")

    assert result.device == torch.device("musa:0")
    assert result.dtype == dtype
    assert result.shape == source.shape
    # torch.tensor() copies; torch.as_tensor() would share the pointer.
    assert result.data_ptr() != source.data_ptr()
    torch.testing.assert_close(result, source)


@testing.skip_if_musa_unavailable
def test_tensor_from_cai_strided_view():
    source = torch.arange(6, dtype=torch.float32, device="musa:0").reshape(2, 3)
    view = source[:, ::2]  # [[0, 2], [3, 5]]
    byte_strides = tuple(s * view.element_size() for s in view.stride())
    ext = FakeExternalArray(view, typestr="<f4", strides=byte_strides)

    result = torch.tensor(ext, device="musa:0")

    assert result.shape == (2, 2)
    torch.testing.assert_close(result, view)
    # If byte strides were ignored, from_blob would treat the pointer as
    # a contiguous (2, 2) and copy [[0, 1], [2, 3]].
    torch.testing.assert_close(
        result.cpu(), torch.tensor([[0.0, 2.0], [3.0, 5.0]])
    )


@testing.skip_if_musa_unavailable
def test_tensor_from_cai_readonly_raises():
    source = torch.tensor([1.0, 2.0], dtype=torch.float32, device="musa:0")
    ext = FakeExternalArray(source, typestr="<f4", read_only=True)

    with pytest.raises(TypeError, match="read only flag is not supported"):
        torch.tensor(ext, device="musa:0")


@testing.skip_if_musa_unavailable
def test_tensor_from_cai_zero_size():
    # CAI 0-size uses integer 0, not None. None never reaches the hooks path.
    ext = FakeExternalArray(shape=(0,), typestr="<f4", data_ptr=0)

    result = torch.tensor(ext, device="musa:0")

    assert result.device == torch.device("musa:0")
    assert result.shape == (0,)
    assert result.numel() == 0
    assert result.dtype == torch.float32


@testing.skip_if_musa_unavailable
def test_tensor_from_cai_missing_shape_raises():
    source = torch.tensor([1.0], dtype=torch.float32, device="musa:0")
    ext = FakeExternalArray(source, typestr="<f4")
    del ext.__cuda_array_interface__["shape"]

    with pytest.raises(TypeError, match="attribute `shape` must exist"):
        torch.tensor(ext, device="musa:0")


@testing.skip_if_musa_unavailable
def test_tensor_from_cai_invalid_typestr_raises():
    source = torch.tensor([1.0], dtype=torch.float32, device="musa:0")
    ext = FakeExternalArray(source, typestr="not-a-dtype")

    with pytest.raises(ValueError, match="cannot parse `typestr`"):
        torch.tensor(ext, device="musa:0")
