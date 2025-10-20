"""Unit tests for FFT operators on MUSA backend (compare with CPU results)."""

import pytest
import torch

complex_types = [torch.complex64, torch.complex128]
float_types = [torch.float32, torch.float64]
norm_modes = [None, "forward", "backward", "ortho"]

fft_inputs_complex = [
    torch.randn(8, dtype=torch.complex64),
    torch.randn(4, 8, dtype=torch.complex64),
    torch.randn(2, 4, 8, dtype=torch.complex64),
]

fft_inputs_real = [
    torch.randn(8),
    torch.randn(4, 8),
    torch.randn(2, 4, 8),
]


@pytest.mark.parametrize("dtype", complex_types)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs_complex + fft_inputs_real)
def test_fft(dtype, norm, input_data):
    """Compare MUSA fft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.fft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.fft(musa_input, norm=norm).cpu()

    assert torch.allclose(
        ref, musa_out, atol=1e-4, rtol=1e-4
    ), f"Mismatch in fft (norm={norm}, dtype={dtype})"


@pytest.mark.parametrize("dtype", complex_types)
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs_complex + fft_inputs_real)
def test_ifft(dtype, norm, input_data):
    """Compare MUSA ifft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.ifft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.ifft(musa_input, norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs_real)
def test_rfft(dtype, norm, input_data):
    """Compare MUSA rfft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    ref = torch.fft.rfft(cpu_input, norm=norm)

    musa_input = cpu_input.to("musa")
    musa_out = torch.fft.rfft(musa_input, norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("dtype", [torch.float32])
@pytest.mark.parametrize("norm", norm_modes)
@pytest.mark.parametrize("input_data", fft_inputs_real)
def test_irfft(dtype, norm, input_data):
    """Compare MUSA irfft result with CPU reference."""
    cpu_input = input_data.to(dtype)
    freq = torch.fft.rfft(cpu_input, norm=norm)
    ref = torch.fft.irfft(freq, n=cpu_input.shape[-1], norm=norm)

    musa_input = freq.to("musa")
    musa_out = torch.fft.irfft(musa_input, n=cpu_input.shape[-1], norm=norm).cpu()

    assert torch.allclose(ref, musa_out, atol=1e-4, rtol=1e-4)
