# pylint: disable=missing-module-docstring,missing-function-docstring
import json

import pytest
import torch
from torch.profiler import ProfilerActivity, profile, schedule

from test_musa_graph import D_in, D_out, DEVICE, N, SimpleModel, set_seed
from torch_musa import testing


def _trace_events(trace_path):
    with open(trace_path, "r", encoding="utf-8") as trace_file:
        return json.load(trace_file)["traceEvents"]


def _correlation_id(event):
    args = event.get("args", {})
    return event.get("correlation") or args.get("correlation")


def _linked_correlation_id(event):
    args = event.get("args", {})
    return event.get("linked_correlation_id") or args.get("linked_correlation_id")


@pytest.mark.skipif(
    testing.get_musa_arch() < 22,
    reason="MUSAGraph is not supported on arch older than qy2",
)
def test_profiler_captures_musa_graph_replay_kernels(tmp_path):
    set_seed()
    real_inputs = [torch.randn(N, D_in) for _ in range(20)]
    real_targets = [torch.randn(N, D_out) for _ in range(20)]
    model = SimpleModel().to(DEVICE)
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    static_input = torch.randn(N, D_in, device=DEVICE)
    static_target = torch.randn(N, D_out, device=DEVICE)

    graph = torch.musa.MUSAGraph()
    capture_stream = torch.musa.Stream()
    optimizer.zero_grad(set_to_none=True)
    with torch.musa.graph(graph, stream=capture_stream):
        static_y_pred = model.forward(static_input)
        static_loss = loss_fn(static_y_pred, static_target)
        static_loss.backward()
        optimizer.step()
    torch.musa.synchronize()

    prof_schedule = schedule(wait=8, warmup=2, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.MUSA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
    ) as prof:
        for data, target in zip(real_inputs, real_targets):
            static_input.copy_(data)
            static_target.copy_(target)
            graph.replay()
            prof.step()
        torch.musa.synchronize()

    trace_path = tmp_path / "musa_graph_replay_trace.json"
    prof.export_chrome_trace(str(trace_path))
    events = _trace_events(trace_path)
    graph_launches = [
        event for event in events if event.get("name") == "musaGraphLaunch"
    ]
    assert graph_launches

    launch_correlation_ids = {_correlation_id(event) for event in graph_launches}
    graph_kernel_events = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and (
            _correlation_id(event) in launch_correlation_ids
            or _linked_correlation_id(event) in launch_correlation_ids
        )
    ]
    assert graph_kernel_events
    assert static_loss is not None
