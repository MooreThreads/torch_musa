"""Unittest for tensor shape APIs."""

# pylint: disable=missing-function-docstring, unused-import
import torch
import torch_musa


def test_empty_strided():
    device = torch.device("musa")
    for shape in [(2, 5), (0, 3)]:
        for strides in [(8, 4), (1, 4), (0, 0)]:
            empty_strided = torch.empty_strided(shape, strides, device=device)
            size = empty_strided.storage().size()
            as_strided = torch.empty(size, device=device).as_strided(shape, strides)
            assert empty_strided.shape == as_strided.shape
            assert empty_strided.stride() == as_strided.stride()


def test_resize_enlarge():
    tensor = torch.randn(1).to("musa")
    tensor.resize_(8)

    assert tensor.storage().size() == 8
    assert tensor.shape == (8,)

    tensor = torch.randn(2, 3).to("musa")
    tensor.resize_((4, 6))

    assert tensor.storage().size() == 24
    assert tensor.shape == (4, 6)


def test_resize_shrink():
    tensor = torch.randn(8).to("musa")
    tensor.resize_(4)
    tensor.zero_()

    assert tensor.storage().size() == 8
    assert tensor.shape == (4,)

    tensor = torch.randn(4, 6).to("musa")
    tensor.resize_((2, 3))

    assert tensor.storage().size() == 24
    assert tensor.shape == (2, 3)


def test_resize_zero():
    tensor = torch.randn(10).to("musa")
    tensor.resize_(0)

    assert tensor.storage().size() == 10
    assert tensor.shape == (0,)

    tensor = torch.randn((2, 3)).to("musa")
    tensor.resize_(0)

    assert tensor.storage().size() == 6
    assert tensor.shape == (0,)


def test_resize_empty_storage():
    tensor = torch.empty((2, 3), device="musa")
    tensor.resize_(10)

    assert tensor.storage().size() == 10
    assert tensor.shape == (10,)
