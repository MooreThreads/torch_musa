"""Test dropout operators."""
# pylint: disable=missing-function-docstring, unused-import
import torch as pt
import numpy as np
import pytest

from torch_musa import testing

p = [0, 0.2, 0.8, 1.0]
inplace = [False]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("shape",
    [
        (1,),
        (128, 128),
        (20, 20, 20, 20),
        (4, 128, 20, 20, 2),
        (2, 2, 3, 4, 5, 6),
        (2, 3, 1, 8, 7, 6, 2),
        (2, 3, 1, 8, 7, 1, 6, 2),
    ]
)
@pytest.mark.parametrize("p_value", p)
@pytest.mark.parametrize("inplace_value", inplace)
@pytest.mark.parametrize("dtype", [pt.float32, pt.float16, pt.bfloat16])
@pytest.mark.parametrize("coefficient", [1, -1, 2, -3])
def test_dropout_train(shape, p_value, inplace_value, dtype, coefficient):
    if testing.get_musa_arch() < 22 and dtype == pt.bfloat16:
        return
    # the output of dropout is nondetermistic though seeded
    device = "musa"
    input_tensor = ((
        pt.rand(shape, dtype=dtype, device=device) + 0.1
    ) * coefficient).requires_grad_()
    module = pt.nn.Dropout(p=p_value, inplace=inplace_value)
    module.train()
    out = module(input_tensor)
    out.backward(pt.ones(input_tensor.shape, dtype=dtype, device=device))
    out_array = out.to("cpu").detach().type(pt.float32).numpy()
    input_array = input_tensor.to("cpu").detach().type(pt.float32).numpy()
    input_grad_array = input_tensor.grad.to("cpu").detach().type(pt.float32).numpy()

    assert (np.count_nonzero(out_array) <= np.count_nonzero(input_array))
    assert (np.count_nonzero(out_array) == np.count_nonzero(input_grad_array))
