# pylint: disable=missing-function-docstring, missing-module-docstring redefined-outer-name, unused-import, invalid-name, too-many-nested-blocks, unnecessary-comprehension
import torch
import pytest
import torch_musa
from torch_musa import testing


inputdata = [j * torch.rand(i) for i in range(20, 100, 29) for j in range(1, 10, 4)] + [
    j * torch.rand(i, k)
    for i in range(20, 100, 29)
    for k in range(20, 100, 29)
    for j in range(1, 10, 3)
]
n_fft = [i for i in range(4, 20, 3)]
center = [True, False]
pad_mode = ["constant", "reflect", "replicate", "circular"]
return_complex = [False, True]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", inputdata)
@pytest.mark.parametrize("n_fft", n_fft)
@pytest.mark.parametrize("center", center)
@pytest.mark.parametrize("pad_mode", pad_mode)
@pytest.mark.parametrize("return_complex", return_complex)
def test_stft(input_data, n_fft, center, pad_mode, return_complex):
    args = [
        {
            "input": input_data,
            "n_fft": n_fft,
            "return_complex": return_complex,
            "pad_mode": pad_mode,
            "center": center,
        }
    ]
    for arg in args:
        test = testing.OpTest(func=torch.stft, input_args=arg)
    test.check_result()
