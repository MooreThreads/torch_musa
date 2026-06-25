# pylint: disable=missing-module-docstring,missing-function-docstring
import json

import torch
from torch.profiler import ProfilerActivity, profile, schedule


DEVICE = "musa"
N, D_IN, H, D_OUT = 64, 4096, 2048, 1024


class TwoLayerTransformer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(D_IN, 4 * H)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(4 * H, D_OUT)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


def _trace_events(trace_path):
    with open(trace_path, "r", encoding="utf-8") as trace_file:
        return json.load(trace_file)["traceEvents"]


def test_profiler_captures_musa_kernel(tmp_path):
    real_inputs = [torch.randn(N, D_IN, device=DEVICE) for _ in range(20)]
    model = TwoLayerTransformer().to(DEVICE)

    prof_schedule = schedule(wait=8, warmup=2, active=3, repeat=1)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.MUSA],
        schedule=prof_schedule,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
        with_stack=True,
    ) as prof:
        for data in real_inputs:
            model.forward(data)
            prof.step()
        torch.musa.synchronize()

    trace_path = tmp_path / "musa_kernel_trace.json"
    prof.export_chrome_trace(str(trace_path))
    events = _trace_events(trace_path)
    assert any(event.get("cat") == "kernel" for event in events)
