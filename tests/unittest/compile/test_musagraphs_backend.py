"""
Unit test for verifying the correctness of Musagraphs backend in torch_musa.
"""

import os
import random
import torch
import pytest
import numpy as np
from torch_musa._inductor import musagraph_trees
from torch_musa import testing


def set_seed(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


N, D_in, H, D_out = 64, 4096, 2048, 1024
DEVICE = "musa"


# simple linear model to test the usability of amp .
class SimpleModel(torch.nn.Module):
    """Simple Model"""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_in, 4 * H)
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(4 * H, 4 * H)
        self.fc3 = torch.nn.Linear(4 * H, 4 * H)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.relu(x)
        return x

    # def __call__(self, x):
    #     return self.forward(x)


@testing.test_on_nonzero_card_if_multiple_musa_device(1)
@pytest.mark.skip(
    reason="Error in MUSA SDK 4.3.2, fixed in 4.3.3"
)  # TODO(@ai-infra): remove this
def test_musagraphs_backend():
    """Test the correctness of the Musagraphs backend for a simple linear model on MUSA."""
    set_seed()
    input_tensors = [torch.randn(N, D_in).to(DEVICE) for _ in range(20)]
    model = SimpleModel().eval().to(DEVICE)
    compile_model = torch.compile(model, backend="musagraphs")

    with torch.inference_mode():
        for input_tensor in input_tensors:
            musagraph_trees.mark_step_begin()
            no_compile_model_output = model(input_tensor)  # pylint: disable=E1102
            compile_model_output = compile_model(input_tensor)

            assert torch.equal(
                no_compile_model_output, compile_model_output
            ), "compile vs eager mismatch"
