"""Test dropout operators."""
# pylint: disable=missing-function-docstring, missing-module-docstring, unused-import
import torch as pt
import numpy as np
import pytest

from torch_musa import testing

inputdata = [
    np.random.randn(1).astype(np.float32),
    np.random.randn(2).astype(np.float32),
    np.random.randn(2, 3).astype(np.float32),
    np.random.randn(2, 3, 4).astype(np.float32),
    np.random.randn(2, 3, 1, 4).astype(np.float32),
    np.random.randn(2, 3, 1, 8, 7).astype(np.float32),
    np.random.randn(2, 3, 1, 8, 7, 2).astype(np.float32),
    np.random.randn(2, 3, 1, 8, 7, 6, 2).astype(np.float32),
    np.random.randn(2, 3, 1, 8, 7, 1, 6, 2).astype(np.float32),
]

p = [0, 0.2, 0.8, 1.0]

inplace = [False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
@pytest.mark.parametrize("p_value", p)
@pytest.mark.parametrize("inplace_value", inplace)
def test_dropout_train(input_data, p_value, inplace_value):
    input_data_shape = input_data.shape
    input_tensor = pt.tensor(input_data, requires_grad=True, device="musa")
    module = pt.nn.Dropout(p=p_value, inplace=inplace_value)
    module.train()
    output = module(input_tensor)
    output.backward(pt.ones(input_data_shape, dtype=pt.float32, device="musa"))
    output_array = output.to("cpu").detach().numpy()
    input_grad_array = input_tensor.grad.to("cpu").detach().numpy()

    assert np.count_nonzero(output_array) <= np.count_nonzero(input_data)
    assert np.count_nonzero(output_array) == np.count_nonzero(input_grad_array)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize(
    "shape",
    [
        (128, 128),
        (20, 20, 20, 20),
        (4, 128, 20, 20, 2),
        (2, 2, 3, 4, 5, 6),
        (2, 3, 1, 8, 7, 6, 2),
        (2, 3, 1, 8, 7, 1, 6, 2),
    ],
)
@pytest.mark.parametrize("p_value", p)
@pytest.mark.parametrize("inplace_value", inplace)
@pytest.mark.parametrize("coefficient", [1, -1, 2, -3])
def test_dropout_fp16_train(shape, p_value, inplace_value, coefficient):
    # the output of dropout is nondetermistic though seeded
    device = "musa"
    t_fp16 = ((
        pt.rand(shape, dtype=pt.float16, device="musa") + 0.1
    ) * coefficient).requires_grad_()
    module = pt.nn.Dropout(p=p_value, inplace=inplace_value)
    module.train()
    out = module(t_fp16)
    out.backward(pt.ones(t_fp16.shape, dtype=pt.float16, device=device))
    out_fp16_array = out.to("cpu").detach().numpy()
    t_fp16_array = t_fp16.to("cpu").detach().numpy()
    t_fp16_grad_array = t_fp16.grad.to("cpu").detach().numpy()

    assert np.count_nonzero(out_fp16_array) <= np.count_nonzero(t_fp16_array)
    assert np.count_nonzero(out_fp16_array) == np.count_nonzero(t_fp16_grad_array)
