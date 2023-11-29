"""Test uncontiguous sub_."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, not-callable
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
    assert testing.DefaultComparator()(boxes, boxes_c)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_sin_():
    c = torch.randn(10, 4)
    m = c.to("musa")
    viewd_c = c.t()
    viewd_m = m.t()
    viewd_c.sin_()
    viewd_m.sin_()
    assert testing.DefaultComparator()(viewd_c, viewd_m)
    assert testing.DefaultComparator()(c, m)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_mul_():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("musa")
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x[:, 2:] *= torch.tensor((2,)).to("musa")
    y[:, 2:] *= torch.tensor((2,))
    assert testing.DefaultComparator()(x, y)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_viewd_mul_2():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("musa")
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x[:, 2:] *= torch.tensor((2,)).to("musa")
    y[:, 2:] *= torch.tensor((2,))
    assert testing.DefaultComparator()(x, y)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_index_put():
    x = torch.tensor([[1, 2, 3], [4, 5, 6]]).to("musa")
    y = torch.tensor([[1, 2, 3], [4, 5, 6]])
    x[..., [0, 2]] = 16
    y[..., [0, 2]] = 16
    assert testing.DefaultComparator()(x, y)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_index_tensor():
    i_m = torch.tensor([0] * 12).to("musa")
    a_m = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]]).to("musa")
    i = torch.tensor([0] * 12)
    a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 0]])
    assert testing.DefaultComparator()(a_m[:, 5:][i_m].cpu(), a[:, 5:][i])


dtypes = [torch.float16, torch.float32, torch.int64]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("dtype", dtypes)
def test_index_tensor_empty(dtype):
    a_m = torch.arange(24).reshape(2, 3, 4).to(dtype).to("musa")
    assert testing.DefaultComparator()(
        a_m[:, [0, 2], [1, 3]].cpu(), a_m.cpu()[:, [0, 2], [1, 3]]
    )
    assert testing.DefaultComparator()(a_m[:, [0, 2]].cpu(), a_m.cpu()[:, [0, 2]])
    assert testing.DefaultComparator()(
        a_m[[0, 1], :, [1, 2]].cpu(), a_m.cpu()[[0, 1], :, [1, 2]]
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_empty_cat():
    a = torch.empty((0))
    b = torch.empty((0))
    c = torch.cat((a, b), 0)
    c_m = torch.cat((a.to("musa"), b.to("musa")), 0)
    assert testing.DefaultComparator()(c, c_m.cpu())


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_issue_415():
    x = torch.randn((16, 1, 768)).transpose(1, 2)
    x_mu = x.to("musa")
    func = torch.nn.functional.adaptive_avg_pool1d
    assert testing.DefaultComparator()(func(x, 1), func(x_mu, 1))

    x = torch.randn((16, 1, 16, 32)).to(memory_format=torch.channels_last)
    x_mu = x.to("musa")
    assert testing.DefaultComparator()(torch.mean(x), torch.mean(x_mu))

    x = torch.randn((16, 32, 1, 1)).to(memory_format=torch.channels_last)
    x_mu = x.to("musa")
    assert testing.DefaultComparator()(torch.sum(x), torch.sum(x_mu))
    assert testing.DefaultComparator()(
        torch.sum(x, 0, keepdim=True), torch.sum(x_mu, 0, keepdim=True)
    )

    x = torch.randn((4, 1, 16)).transpose(1, 2).unsqueeze(1)
    x_mu = x.to("musa")
    assert testing.DefaultComparator()(torch.max(x), torch.max(x_mu))
    assert testing.DefaultComparator()(
        torch.max(x, 0, keepdim=True)[0], torch.max(x_mu, 0, keepdim=True)[0]
    )
    assert testing.DefaultComparator()(
        torch.max(x, 0, keepdim=True)[1], torch.max(x_mu, 0, keepdim=True)[1]
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_div_broadcast():
    inp = torch.randn(40, 1)
    out = torch.div(inp, 1, rounding_mode="floor")
    out_musa = torch.div(inp.to("musa"), 1, rounding_mode="floor")
    assert testing.DefaultComparator()(out, out_musa)
    out = torch.div(inp, 1, rounding_mode="trunc")
    out_musa = torch.div(inp.to("musa"), 1, rounding_mode="trunc")
    assert testing.DefaultComparator()(out, out_musa)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_uncontiguous_half_cast():
    x = torch.randn(7, 1, 4096).half()
    x_m = x.to("musa")
    y_m = x_m[..., :5].float()
    y = x[..., :5].float()
    assert testing.DefaultComparator()(y_m, y)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_issue_538():
    c_input = torch.rand((1, 128, 64, 64))
    model = torch.nn.InstanceNorm2d(128, affine=True)
    res = model(c_input)

    musa_input = c_input.to("musa")
    musamodel = model.to("musa")
    musa_res = musamodel(musa_input)
    assert testing.DefaultComparator(abs_diff=1e-5)(res, musa_res)
