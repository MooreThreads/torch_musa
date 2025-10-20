"""Unit tests for UMM feature, take effects globally"""

# pylint: disable=C0413, W0621, C0103, C0415, C0411
import io
import os

os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "cpu:unified"
import torch
from torchvision.models import resnet18
from torch import nn
from torch import optim


def load_model():
    """Load a ResNet18 model."""
    model = resnet18(pretrained=False)

    state_dict = model.state_dict()
    checkpoint = io.BytesIO()
    torch.save(state_dict, checkpoint)
    checkpoint.seek(0)
    current_allocated = torch.musa.memory_allocated()
    state_dict_loaded = torch.load(checkpoint, map_location="musa")
    max_allocated = torch.musa.max_memory_allocated()
    peak_memory_unified_global = max_allocated - current_allocated
    print("peak_memory", peak_memory_unified_global)
    assert (
        peak_memory_unified_global == 0
    ), "torch.load(map_location='musa') with umm dont apply gpu memory "
    new_model = resnet18(pretrained=False)
    new_model.to("musa")
    new_model.load_state_dict(state_dict_loaded)
    return new_model


def test_torch_load_use_umm_globally():
    """
    Takes effect globally for torch.load(map_location="musa") memory usage reduce
    import os
    os.environ["PYTORCH_MUSA_ALLOC_CONF"] = "cpu:unified"
    import torch
    """
    current_allocated = torch.musa.memory_allocated()

    model = load_model()

    max_allocated = torch.musa.max_memory_allocated()
    peak_memory_unified_global = max_allocated - current_allocated

    print(f"peak_memory_unified_global: {peak_memory_unified_global / 1024**2:.2f} MB")

    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    fake_inputs = torch.randn(8, 3, 224, 224, device="musa")
    fake_labels = torch.randint(0, 10, (8,), device="musa")

    num_steps = 5
    for step in range(num_steps):
        optimizer.zero_grad()
        outputs = model(fake_inputs)
        loss = criterion(outputs, fake_labels)
        loss.backward()
        optimizer.step()
        print(f"Step [{step+1}/{num_steps}], Loss: {loss.item():.4f}")
