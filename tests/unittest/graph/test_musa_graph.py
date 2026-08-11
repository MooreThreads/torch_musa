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

import gc
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
    assert testing.DefaultComparator(res_in_graph, res_no_graph)


def test_graph_pool_handle_api():
    pool = torch.musa.graph_pool_handle()
    assert isinstance(pool, tuple)
    assert len(pool) == 2
    assert torch.musa._POOL_HANDLE(pool) == pool


class _FakeStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, *_args):
        return None


class _FakeGraph:
    def capture_begin(self, *args, capture_error_mode="global"):
        self.capture_begin_args = args
        self.capture_error_mode = capture_error_mode

    def capture_end(self):
        self.capture_ended = True


def _fake_graph_context():
    ctx = torch.musa.graph.__new__(torch.musa.graph)
    ctx.pool = ()
    ctx.stream_ctx = _FakeStreamContext()
    ctx.musa_graph = _FakeGraph()
    ctx.capture_error_mode = "global"
    return ctx


def test_graph_gc_respects_compiler_config(monkeypatch):
    num_gc_calls = 0

    def fake_collect():
        nonlocal num_gc_calls
        num_gc_calls += 1

    monkeypatch.setattr(torch.musa, "synchronize", lambda: None)
    monkeypatch.setattr(torch.musa, "empty_cache", lambda: None)
    monkeypatch.setattr(gc, "collect", fake_collect)

    monkeypatch.setattr(torch.compiler.config, "force_cudagraph_gc", False)
    with _fake_graph_context():
        pass
    assert num_gc_calls == 0

    monkeypatch.setattr(torch.compiler.config, "force_cudagraph_gc", True)
    with _fake_graph_context():
        pass
    assert num_gc_calls == 1


def test_profiler_update_skips_uncaptured_graph(monkeypatch):
    captured_graph = torch.musa.MUSAGraph()
    uncaptured_graph = torch.musa.MUSAGraph()
    captured_graph._has_capture = True
    uncaptured_graph._has_capture = False
    reinstantiated = []

    monkeypatch.setattr(
        torch.musa.MUSAGraph,
        "get_latest_instance",
        classmethod(lambda cls: [captured_graph, uncaptured_graph]),
    )
    monkeypatch.setattr(
        captured_graph,
        "reinstantiate_graph",
        lambda: reinstantiated.append(captured_graph),
    )
    monkeypatch.setattr(
        uncaptured_graph,
        "reinstantiate_graph",
        lambda: pytest.fail("uncaptured graph should not be reinstantiated"),
    )

    torch.musa.update_musa_graph_with_profile()
    assert reinstantiated == [captured_graph]


@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="MUSAGraph is not supported on arch older than qy2",
)
@testing.test_on_nonzero_card_if_multiple_musa_device(1)
def test_musa_graph_keep_graph_raw_handles():
    stream = torch.musa.Stream()
    graph = torch.musa.MUSAGraph(keep_graph=True)
    static_input = torch.ones(4, device=DEVICE)
    static_output = torch.empty_like(static_input)

    torch.musa.synchronize()
    with torch.musa.graph(graph, stream=stream):
        static_output.copy_(static_input)
        static_output.add_(1)

    raw_graph = graph.raw_musa_graph()
    assert isinstance(raw_graph, int)
    assert raw_graph != 0

    graph.instantiate()
    raw_graph_exec = graph.raw_musa_graph_exec()
    assert isinstance(raw_graph_exec, int)
    assert raw_graph_exec != 0

    static_input.fill_(3)
    graph.replay()
    torch.musa.synchronize()
    assert testing.DefaultComparator(static_output.cpu(), torch.full((4,), 4.0))


test_musa_graph()
