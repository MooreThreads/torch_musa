# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,unused-variable,wrong-import-order,not-callable
"""
torch.musa.graph is a simple, versatile context manager that captures MUSA work in its context.
Before capture, warm up the workload to be captured by running a few eager iterations.
Warmup must occur on a side stream. Because the graph reads from and writes to the 
same memory addresses in every replay, you must maintain long-lived references to 
tensors that hold input and output data during capture. To run the graph on new input data, 
copy new data to the capture’s input tensor(s), replay the graph, then read the new output
from the capture’s output tensor(s).

If the entire network is capture safe, one can capture and replay the whole network as in the
following example.
"""

import os
import random
import pytest
import torch
from torch import nn
import numpy as np
import torch_musa
from torch_musa import testing

# import faulthandler
# faulthandler.enable()


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
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_in, 4 * H)
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc2 = torch.nn.Linear(4 * H, D_out)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.relu(x)
        return x

    def __call__(self, x):
        return self.forward(x)


def train_in_musa_graph():
    set_seed()
    real_inputs = [torch.randn(N, D_in) for _ in range(20)]
    real_targets = [torch.randn(N, D_out) for _ in range(20)]
    model = SimpleModel().to(DEVICE)

    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # Placeholders used for capture
    static_input = torch.randn(N, D_in, device="musa")
    static_target = torch.randn(N, D_out, device="musa")
    # capturez
    g = torch.musa.MUSAGraph()
    optimizer.zero_grad(set_to_none=True)
    with torch.musa.graph(g):
        static_y_pred = model(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()

    for data, target in zip(real_inputs, real_targets):
        # Fills the graph's input memory with new data to compute on
        static_input.copy_(data)
        static_target.copy_(target)
        # replay() includes forward, backward, and step.
        # You don't even need to call optimizer.zero_grad() between iterations
        # because the captured backward refills static .grad tensors in place.
        g.replay()
        print("static_loss:", static_loss)
    res = static_loss.cpu()
    return res


def train():
    set_seed()
    real_inputs = [torch.randn(N, D_in) for _ in range(20)]
    real_targets = [torch.randn(N, D_out) for _ in range(20)]
    model = SimpleModel().to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for data, target in zip(real_inputs, real_targets):
        pred = model(data.to(DEVICE))
        loss = loss_fn(pred, target.to(DEVICE))
        loss.backward()
        optimizer.step()
        print("loss:", loss)
    res = loss.cpu()
    return res


@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="MUSAGraph is not supported on arch older than qy2",
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_musa_graph():
    res_in_graph = train_in_musa_graph()
    res_no_graph = train()
    print(res_in_graph, res_no_graph)
    testing.DefaultComparator(res_in_graph, res_no_graph)
