"""Test ctc forward & backward operator."""
# pylint: disable=missing-function-docstring, redefined-outer-name, unused-import
import torch
import pytest

import torch_musa
from torch_musa import testing

# input, other

def ctc_forward_inputs():
    # log_probs: (T, B, V+1)
    # targets: (B, H), H <= T
    # input_lengths: (B, )
    # target_lengths: (B, )
    return [
        {'log_probs': torch.randn(50, 1, 20).log_softmax(2).detach().requires_grad_(),
        'targets': torch.randint(1, 20, (1, 30), dtype=torch.long),
        'input_lengths': torch.full((1,), 50, dtype=torch.long),
        'target_lengths': torch.randint(10, 30, (1,), dtype=torch.long)},
        {'log_probs': torch.randn(50, 16, 20).log_softmax(2).detach().requires_grad_(),
        'targets': torch.randint(1, 20, (16, 30), dtype=torch.long),
        'input_lengths': torch.full((16,), 50, dtype=torch.long),
        'target_lengths': torch.randint(10, 30, (16,), dtype=torch.long)},
        {'log_probs': torch.randn(128, 16, 20).log_softmax(2).detach().requires_grad_(),
        'targets': torch.randint(1, 20, (16, 50), dtype=torch.long),
        'input_lengths': torch.full((16,), 128, dtype=torch.long),
        'target_lengths': torch.randint(10, 50, (16,), dtype=torch.long)},
        {'log_probs': torch.randn(128, 16, 20).log_softmax(2).detach().requires_grad_(),
        'targets': torch.randint(1, 20, (16, 50), dtype=torch.long),
        'input_lengths': torch.full((16,), 100, dtype=torch.long),
        'target_lengths': torch.randint(10, 50, (16,), dtype=torch.long)},
    ]

reduction = ['mean', 'sum']
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.parametrize("input_data", ctc_forward_inputs())
@pytest.mark.parametrize("reduction", reduction)
def test_ctc_loss(input_data, reduction):
    ctc = torch.nn.CTCLoss
    ctc_args = {
        'reduction': reduction
    }
    test = testing.OpTest(
        func=ctc,
        input_args=ctc_args,
        comparators=testing.DefaultComparator(abs_diff=1e-4)
    )
    test.check_result({
        "log_probs": input_data['log_probs'],
        "targets": input_data['targets'],
        "input_lengths": input_data['input_lengths'],
        "target_lengths": input_data['target_lengths']
        }, train=True)
        