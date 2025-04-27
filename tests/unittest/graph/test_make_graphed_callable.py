# pylint: disable= missing-module-docstring, missing-class-docstring,missing-function-docstring,unused-import,wrong-import-order,unused-variable,not-callable
"""
If some of your network is unsafe to capture
(e.g., due to dynamic control flow, dynamic shapes, CPU syncs, or essential CPU-side logic),
you can run the unsafe part(s) eagerly and use torch.msua.make_graphed_callables to graph
only the capture-safe part(s). This is demonstrated next.

make_graphed_callables accepts callables (functions or nn.Module and returns graphed versions.
By default, callables returned by make_graphed_callables are autograd-aware, and can be used
in the training loop as direct replacements for the functions or nn.Module you passed.
make_graphed_callables internally creates MUSAGraph objects, runs warm up iterations, 
and maintains static inputs and outputs as needed. Therefore, (unlike with torch.msua.graph)
you don’t need to handle those manually.


"""
import os
import random
from torch_musa import testing
import pytest
import torch
from torch import nn
import numpy as np
import torch_musa


def set_seed(seed=1029):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


N, D_in, H, D_out = 640, 4096, 2048, 1024
DEVICE = "musa"


def make_partial_graph():
    set_seed()

    module1 = torch.nn.Linear(D_in, H).musa()
    module2 = torch.nn.Linear(H, D_out).musa()
    module3 = torch.nn.Linear(H, D_out).musa()

    loss_fn = torch.nn.MSELoss()

    # Sample inputs used for capture
    # requires_grad state of sample inputs must match
    # requires_grad state of real inputs each callable will see.
    x = torch.randn(N, D_in, device=DEVICE)
    h = torch.randn(N, H, device=DEVICE, requires_grad=True)

    module1 = torch.musa.make_graphed_callables(module1, (x,))
    module2 = torch.musa.make_graphed_callables(module2, (h,))
    module3 = torch.musa.make_graphed_callables(module3, (h,))

    real_inputs = [torch.rand_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device=DEVICE) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):

        tmp = module1(data)  # forward ops run as a graph

        if tmp.sum().item() > 0:
            tmp = module2(tmp)  # forward ops run as a graph
        else:
            tmp = module3(tmp)  # forward ops run as a graph

        loss = loss_fn(tmp, target)
        # module2's or module3's (whichever was chosen) backward ops,
        # as well as module1's backward ops, run as graphs
        print("----loss:", loss)
    return loss.cpu()


def no_graph():
    set_seed()
    module1 = torch.nn.Linear(D_in, H).musa()
    module2 = torch.nn.Linear(H, D_out).musa()
    module3 = torch.nn.Linear(H, D_out).musa()

    loss_fn = torch.nn.MSELoss()

    x = torch.randn(N, D_in, device=DEVICE)
    h = torch.randn(N, H, device=DEVICE, requires_grad=True)

    real_inputs = [torch.rand_like(x) for _ in range(10)]
    real_targets = [torch.randn(N, D_out, device=DEVICE) for _ in range(10)]

    for data, target in zip(real_inputs, real_targets):

        tmp = module1(data)

        if tmp.sum().item() > 0:
            tmp = module2(tmp)
        else:
            tmp = module3(tmp)

        loss = loss_fn(tmp, target)
    return loss.cpu()


@pytest.mark.skipif(
    testing.get_musa_arch() < 22, reason="bf16 is not supported on arch older than qy2"
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_musa_graph():
    res_in_graph = make_partial_graph()
    res_no_graph = no_graph()
    print(res_in_graph, res_no_graph)
    testing.DefaultComparator(res_in_graph, res_no_graph)
