"""Test GRU forward & backward operator."""

# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import, invalid-name
import torch
import pytest

from torch_musa import testing


seqlens = [5, 16]
batch_sizes = [1, 15]
input_sizes = [8, 16]
hidden_sizes = [16]
num_layers = [2]
support_dtypes = [
    torch.float32,
    torch.float16,
    # torch.bfloat16  # TODO(@ai-infra): BF16 precision
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("seqlen", seqlens)
@pytest.mark.parametrize("batch_size", batch_sizes)
@pytest.mark.parametrize("input_size", input_sizes)
@pytest.mark.parametrize("hidden_size", hidden_sizes)
@pytest.mark.parametrize("num_layers", num_layers)
@pytest.mark.parametrize("batch_first", [False, True])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", support_dtypes)
def test_gru_fwdbwd(
    seqlen, batch_size, input_size, hidden_size, num_layers, batch_first, bias, dtype
):
    gru = torch.nn.GRU
    if batch_first:
        input_data = torch.rand(batch_size, seqlen, input_size, requires_grad=True)
        h0 = torch.rand(num_layers, batch_size, hidden_size)
    else:
        input_data = torch.rand(seqlen, batch_size, input_size, requires_grad=True)
        h0 = torch.rand(num_layers, batch_size, hidden_size)

    gru_args = {
        "input_size": input_size,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "batch_first": batch_first,
        "bias": bias,
    }

    if dtype in [torch.bfloat16, torch.float16]:
        abs_diff, rel_diff = 5e-2, 5e-3
    else:
        abs_diff, rel_diff = 1e-4, 1e-5

    test = testing.OpTest(
        func=gru,
        input_args=gru_args,
        test_dtype=dtype,
        comparators=testing.DefaultComparator(abs_diff=abs_diff, rel_diff=rel_diff),
    )
    test.check_result(
        {
            "input": input_data.to(dtype),
            "hx": h0.to(dtype),
        },
        train=True,
    )
