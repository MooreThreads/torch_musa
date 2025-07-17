"""Test CodegenTriton"""

# pylint: disable=missing-function-docstring, unused-import
import sys
import subprocess
import pytest
import torch
from torch._inductor.codecache import PyCodeCache
from torch.testing._internal.common_utils import TestCase as TorchTestCase
from torch._inductor.utils import fresh_inductor_cache
import torch_musa
from torch_musa.testing.base_test_tool import _HAS_TRITON


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
class TestCodegenTriton(TorchTestCase):
    """Test class of CodegenTriton"""

    def setup_class(self):
        # ensure the cache is empty
        PyCodeCache.cache.clear()

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.cache.values():
            if hasattr(v, "benchmark_compiled_module"):
                self.assertTrue(
                    compiled_module is None, "Found multiple compiled modules"
                )
                compiled_module = v

        self.assertTrue(compiled_module is not None)
        return compiled_module

    def test_add(self):
        def _add_mul_fused(a, b, c):
            out = (a + b) * c
            return out

        with fresh_inductor_cache():
            opt_add_mul_fused = torch.compile(_add_mul_fused)
            a = torch.randn((8, 8), device="musa")
            b = torch.randn((8, 8), device="musa")
            c = torch.randn((8, 8), device="musa")
            ref = _add_mul_fused(a, b, c)
            result = opt_add_mul_fused(a, b, c)
            self.assertEqual(ref, result)

            compiled_module = self.get_compiled_module()
            # benchmark result
            bench_output = subprocess.check_output(
                f"{sys.executable} {compiled_module.__file__}".split(),
                stderr=subprocess.STDOUT,
            ).decode()

            self.assertTrue(len(bench_output) > 0)
