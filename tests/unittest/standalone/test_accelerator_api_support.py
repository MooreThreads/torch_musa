"""Verify torch.accelerator APIs are supported by MUSA."""

# pylint: disable=C0413, C0411, C0103, W0621

import os
import random
import warnings

import numpy as np
import pytest
import torch

from torch import accelerator

# Every reported dtype must round-trip through the device, so the mask has to
# stay inside the dtypes MUSA actually allocates and casts.
CORE_DTYPES = (
    torch.float32,
    torch.float16,
    torch.bfloat16,
    torch.float64,
    torch.int8,
    torch.int32,
    torch.int64,
    torch.uint8,
    torch.bool,
    torch.complex64,
)


def _call_api(name, fn):
    """Call one accelerator API and return an error string if it is unsupported."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            fn()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return f"{name}: {type(exc).__name__}: {exc}"
    return None


def _check_device_capability():
    caps = accelerator.get_device_capability()
    supported = caps["supported_dtypes"]
    assert isinstance(supported, set) and supported, supported
    assert set(CORE_DTYPES) <= supported, supported - set(CORE_DTYPES)

    reference = torch.zeros(16)
    for dtype in supported:
        # The torch 2.13 contract: a dtype in the set must allocate on the
        # accelerator and round-trip to and from the reference dtype.
        tensor = torch.empty(16, dtype=dtype, device="musa")
        tensor.copy_(reference.to(dtype))
        back = tensor.to(reference.dtype)
        assert back.shape == reference.shape


N, D_IN, H, D_OUT = 32, 128, 64, 32
STEPS = 4


class _TinyModel(torch.nn.Module):
    """Small stand-in for a capture-safe network, in the shape of the
    SimpleModel used by tests/unittest/graph/test_musa_graph.py."""

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_IN, H)
        self.fc2 = torch.nn.Linear(H, D_OUT)
        self.relu = torch.nn.ReLU()

    def forward(self, inputs):
        return self.fc2(self.relu(self.fc1(inputs)))


def _seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.musa.manual_seed_all(seed)


def _training_batch():
    _seed(1234)
    inputs = [torch.randn(N, D_IN) for _ in range(STEPS)]
    targets = [torch.randn(N, D_OUT) for _ in range(STEPS)]
    model = _TinyModel().to("musa")
    return inputs, targets, model


def _train_eager():
    inputs, targets, model = _training_batch()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for data, target in zip(inputs, targets):
        pred = model(data.to("musa"))  # pylint: disable=not-callable
        loss = loss_fn(pred, target.to("musa"))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return (
        loss.detach().cpu(),
        [param.detach().cpu().clone() for param in model.parameters()],
    )


def _check_model_graph():
    """Capture a whole training step with accelerator.Graph.

    Same pattern as test_musa_graph.py: the graph reads and writes fixed
    addresses, so the step runs against long-lived static tensors. Each
    iteration copies fresh data in, replays, and reads the static loss back.
    The captured result must match the identical step run eagerly.
    """
    inputs, targets, model = _training_batch()
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    static_input = torch.randn(N, D_IN, device="musa")
    static_target = torch.randn(N, D_OUT, device="musa")

    # Capture must run on a non-default stream; MUSAGraph rejects the default.
    capture_stream = torch.musa.Stream()
    graph = accelerator.Graph()
    optimizer.zero_grad(set_to_none=True)
    with torch.musa.stream(capture_stream), graph:
        static_pred = model(static_input)  # pylint: disable=not-callable
        static_loss = loss_fn(static_pred, static_target)
        static_loss.backward()
        optimizer.step()

    for data, target in zip(inputs, targets):
        static_input.copy_(data)
        static_target.copy_(target)
        # replay runs the captured forward, backward and step. The captured
        # backward refills the .grad tensors in place, so zero_grad is not
        # called between iterations.
        graph.replay()
    torch.musa.synchronize()

    eager_loss, eager_params = _train_eager()
    torch.testing.assert_close(static_loss.detach().cpu(), eager_loss)
    graph_params = [param.detach().cpu() for param in model.parameters()]
    for graphed, eager in zip(graph_params, eager_params):
        torch.testing.assert_close(graphed, eager)


def _check_graph():
    tensor = torch.zeros(4, device="musa")
    # Capture must run on a non-default stream; MUSAGraph rejects the default.
    capture_stream = torch.musa.Stream()
    graph = accelerator.Graph()
    with torch.musa.stream(capture_stream), graph:
        tensor.add_(1)
    before = tensor.cpu().clone()
    graph.replay()
    torch.musa.synchronize()
    after = tensor.cpu()
    # Captured work is recorded, not executed, so one replay applies it once.
    assert torch.equal(after, before + 1), (before, after)

    # The other capture modes MUSA supports must capture and replay too.
    for mode in ("global", "thread_local", "relaxed"):
        other = torch.zeros(4, device="musa")
        mode_graph = accelerator.Graph(capture_error_mode=mode)
        with torch.musa.stream(capture_stream), mode_graph:
            other.add_(1)
        mode_graph.replay()
        torch.musa.synchronize()
        assert torch.equal(other.cpu(), torch.ones(4)), mode


def test_torch_accelerator_apis_supported_on_musa():
    """All public torch.accelerator APIs should work on MUSA.

    The test probes every API exported by torch.accelerator.__all__.  It collects
    all unsupported APIs first, then fails once with the complete list instead of
    stopping at the first unsupported function.
    """
    assert torch.musa.is_available(), "MUSA must be available to test accelerator APIs"

    # Initialize the MUSA runtime/allocator before probing memory-stat APIs.
    tensor = torch.empty(1, device="musa")
    del tensor

    def probe_current_device_idx():
        accelerator.current_device_idx()

    def probe_set_device_index():
        prev = accelerator.current_device_index()
        try:
            accelerator.set_device_index(0)
        finally:
            accelerator.set_device_index(prev)

    def probe_set_device_idx():
        prev = accelerator.current_device_index()
        try:
            accelerator.set_device_idx(0)
        finally:
            accelerator.set_device_index(prev)

    def probe_device_index_context():
        with accelerator.device_index(0):
            pass

    def probe_set_stream():
        accelerator.set_stream(torch.musa.current_stream())

    cases = {
        "current_accelerator": accelerator.current_accelerator,
        "current_device_idx": probe_current_device_idx,
        "current_device_index": accelerator.current_device_index,
        "get_device_capability": _check_device_capability,
        "current_stream": accelerator.current_stream,
        "device_count": accelerator.device_count,
        "device_index": probe_device_index_context,
        "empty_cache": accelerator.empty_cache,
        "empty_host_cache": accelerator.empty_host_cache,
        "get_memory_info": accelerator.get_memory_info,
        "is_available": accelerator.is_available,
        "max_memory_allocated": accelerator.max_memory_allocated,
        "max_memory_reserved": accelerator.max_memory_reserved,
        "memory_allocated": accelerator.memory_allocated,
        "memory_reserved": accelerator.memory_reserved,
        "memory_stats": accelerator.memory_stats,
        "reset_accumulated_memory_stats": (
            accelerator.reset_accumulated_memory_stats
        ),
        "reset_peak_memory_stats": accelerator.reset_peak_memory_stats,
        "set_device_idx": probe_set_device_idx,
        "set_device_index": probe_set_device_index,
        "set_stream": probe_set_stream,
        "synchronize": accelerator.synchronize,
        "Graph": _check_graph,
        "model_graph_demo": _check_model_graph,
    }

    exported = set(accelerator.__all__)
    # model_graph_demo is a usage example on top of Graph, not an export.
    assert set(cases) - {"model_graph_demo"} == exported, sorted(
        (set(cases) - {"model_graph_demo"}) ^ exported
    )

    unsupported = []
    for name in list(accelerator.__all__) + ["model_graph_demo"]:
        error = _call_api(name, cases[name])
        if error is not None:
            unsupported.append(error)

    if unsupported:
        pytest.fail(
            "Unsupported torch.accelerator APIs on MUSA:\n"
            + "\n".join(f" - {error}" for error in unsupported)
        )
