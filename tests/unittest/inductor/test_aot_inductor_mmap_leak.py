"""Regression test for the AOTInductor weights mmap leak."""

# Owner(s): ["module: inductor"]
import gc
import tempfile

import pytest
import torch
from torch._inductor.package import load_package
from torch._inductor.test_case import TestCase

import torch_musa  # pylint: disable=unused-import
from torch_musa.testing.base_test_tool import _HAS_TRITON

# pylint: disable=missing-function-docstring,missing-class-docstring

# Large enough that one leaked mapping dwarfs allocator and dlopen noise.
_HIDDEN = 4096
_NUM_LAYERS = 2
_WEIGHTS_BYTES = _NUM_LAYERS * _HIDDEN * _HIDDEN * 4  # ~128MB
_ITERS = 5


class _BigWeights(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(_HIDDEN, _HIDDEN, bias=False) for _ in range(_NUM_LAYERS)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def _vm_size_bytes():
    with open("/proc/self/status", encoding="utf-8") as status:
        for line in status:
            if line.startswith("VmSize:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("VmSize missing from /proc/self/status")


def _weights_mappings():
    count = 0
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            fields = line.split()
            if len(fields) < 6 or fields[1] != "rw-p":
                continue
            pathname = fields[-1]
            if pathname != "(deleted)" and not pathname.endswith(".so"):
                continue
            start, end = (int(v, 16) for v in fields[0].split("-"))
            if end - start >= _WEIGHTS_BYTES:
                count += 1
    return count


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
@pytest.mark.skipif(not torch.musa.is_available(), reason="No MUSA device")
class TestAOTInductorMmapLeak(TestCase):
    device = "musa"

    def _package(self, path):
        model = _BigWeights().to(self.device).eval()
        example_inputs = (torch.randn(8, _HIDDEN, device=self.device),)
        with torch.no_grad():
            exported = torch.export.export(model, example_inputs, strict=False)
        return (
            torch._inductor.aoti_compile_and_package(  # pylint: disable=protected-access
                exported,
                package_path=path,
                inductor_configs={
                    "aot_inductor.package": True,
                    "aot_inductor.force_mmap_weights": True,
                },
            ),
            example_inputs,
        )

    def test_weights_unmapped_on_del(self):
        with tempfile.NamedTemporaryFile(suffix=".pt2") as pkg:
            package_path, example_inputs = self._package(pkg.name)

            def load_run_drop():
                loaded = load_package(package_path)
                with torch.no_grad():
                    loaded(*example_inputs)
                mapped = _weights_mappings()
                del loaded
                gc.collect()
                return mapped

            mapped_while_alive = load_run_drop()
            self.assertGreater(
                mapped_while_alive,
                _weights_mappings(),
                "mapping outlived the model, or USE_MMAP_SELF is off",
            )

            baseline = _vm_size_bytes()
            for _ in range(_ITERS):
                load_run_drop()
            growth = _vm_size_bytes() - baseline

        self.assertLess(
            growth,
            _WEIGHTS_BYTES,
            f"leaked {growth / 2 ** 20:.1f}MB over {_ITERS} cycles",
        )


if __name__ == "__main__":
    from torch._inductor.test_case import (  # pylint: disable=ungrouped-imports
        run_tests,
    )

    if torch.musa.is_available():
        run_tests(needs="filelock")
