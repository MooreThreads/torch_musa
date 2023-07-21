"""Test uncontiguous sub_."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest
import torch_musa

from torch_musa import testing

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_sub_():
    boxes_c = torch.randn(48010, 4)
    boxes = boxes_c.to("musa")
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes_c[:, :2] -= boxes_c[:, 2:] / 2
    testing.DefaultComparator(boxes, boxes_c)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_sin_():
    c = torch.randn(10, 4)
    m = c.to("musa")
    viewd_c = c.t()
    viewd_m = m.t()
    viewd_c.sin_()
    viewd_m.sin_()
    testing.DefaultComparator(viewd_c, viewd_m)
    testing.DefaultComparator(c, m)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_mul_():
    x = torch.tensor([[1,2,3], [4,5,6]]).to("musa")
    y = torch.tensor([[1,2,3], [4,5,6]])
    x[:,2:] *= torch.tensor((2,))
    y[:,2:] *= torch.tensor((2,))
    testing.DefaultComparator(x, y)

@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_mul_2():
    x = torch.tensor([[1,2,3], [4,5,6]]).to("musa")
    y = torch.tensor([[1,2,3], [4,5,6]])
    x[:,2:] *= torch.tensor((2,)).to("musa")
    y[:,2:] *= torch.tensor((2,))
    testing.DefaultComparator(x, y)
