"""Test ctc forward & backward operator."""

# pylint: disable=missing-function-docstring, C0103
import torch
import pytest

from torch_musa import testing


# [T, N, C, S_min, S]
configs = [
    [50, 1, 20, 10, 30],
    [128, 16, 20, 10, 50],
    [504, 6, 4233, 1, 504],
]


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("blank", [0, 1])
@pytest.mark.parametrize("zero_infinity", [True, False])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("pad", [True, False])
def test_ctc_loss(config, blank, zero_infinity, reduction, pad):
    ctc = torch.nn.CTCLoss
    ctc_args = {"reduction": reduction, "blank": blank, "zero_infinity": zero_infinity}
    test = testing.OpTest(
        func=ctc,
        input_args=ctc_args,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    T, N, C, S_min, S = config
    inputs = torch.randn((T, N, C)).log_softmax(2).requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(S_min, S, size=(N,), dtype=torch.long)
    if pad:
        target_size = (N, S)
    else:
        target_size = (sum(target_lengths),)
    targets = torch.randint(low=1, high=C, size=target_size, dtype=torch.long)

    test.check_result(
        {
            "log_probs": inputs,
            "targets": targets,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
        },
        train=True,
    )

    input_args_cpu = {
        "log_probs": inputs.requires_grad_(),
        "targets": targets,
        "input_lengths": input_lengths,
        "target_lengths": target_lengths,
    }
    input_args_musa = {
        "log_probs": inputs.musa().requires_grad_(),
        "targets": targets.musa(),
        "input_lengths": input_lengths.musa(),
        "target_lengths": target_lengths.musa(),
    }
    assert (
        ctc(**ctc_args)(**input_args_cpu).grad_fn.__class__
        == ctc(**ctc_args)(**input_args_musa).grad_fn.__class__
    )


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("config", configs)
@pytest.mark.parametrize("blank", [0, 1])
@pytest.mark.parametrize("zero_infinity", [True, False])
@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
@pytest.mark.parametrize("pad", [True, False])
def test_ctc_loss_fp16(config, blank, zero_infinity, reduction, pad):
    ctc = torch.nn.CTCLoss
    ctc_args = {"reduction": reduction, "blank": blank, "zero_infinity": zero_infinity}
    test = testing.OpTest(
        func=ctc,
        input_args=ctc_args,
        comparators=testing.DefaultComparator(abs_diff=1e-3),
    )
    T, N, C, S_min, S = config
    inputs = torch.randn((T, N, C)).log_softmax(2).requires_grad_()
    input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.long)
    target_lengths = torch.randint(S_min, S, size=(N,), dtype=torch.long)
    if pad:
        target_size = (N, S)
    else:
        target_size = (sum(target_lengths),)
    targets = torch.randint(low=1, high=C, size=target_size, dtype=torch.long)

    test.check_musafp16_vs_musafp32(
        {
            "log_probs": inputs,
            "targets": targets,
            "input_lengths": input_lengths,
            "target_lengths": target_lengths,
        },
        train=True,
    )
