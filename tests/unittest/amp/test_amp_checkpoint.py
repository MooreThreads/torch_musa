# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,not-callable,W0613
import pytest
import torch
from torch.utils.checkpoint import checkpoint
from torch_musa import testing
import torch_musa


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(10, 10)
        self.linear2 = torch.nn.Linear(10, 10)

    def forward(self, x):
        x = checkpoint(self.linear1, x, use_reentrant=False)
        x = self.linear2(x)
        return x


def custom_backward_hook(module, grad_input, grad_output):
    try:
        print(f"Backward hook - grad_input dtype: {grad_input[0].dtype}")
    except AttributeError:
        print("Backward hook - grad_input is None")


def test_amp_checkpoint_fp16():
    device = "musa"
    model = SimpleModel().to(device)
    model = model.to(torch.float16)

    with torch.autocast(device, torch.float16):
        model = SimpleModel().to(device)
        model = model.to(torch.float16)
        inputs = torch.randn(1, 10, dtype=torch.float16, device=device)

        # register backward hook
        model.linear1.register_full_backward_hook(custom_backward_hook)

        # forward
        output = model(inputs)

        # compute loss
        loss = output.sum()
        loss.backward()


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
def test_amp_checkpoint():
    device = "musa"
    model = SimpleModel().to(device)
    model = model.to(torch.bfloat16)

    with torch.autocast(device, torch.bfloat16):
        model = SimpleModel().to(device)
        model = model.to(torch.bfloat16)
        inputs = torch.randn(1, 10, dtype=torch.bfloat16, device=device)

        # register backward hook
        model.linear1.register_full_backward_hook(custom_backward_hook)

        # forward
        output = model(inputs)

        # compute loss
        loss = output.sum()
        loss.backward()

    assert inputs.dtype == output.dtype
