"""Test CodegenTriton"""

# pylint: disable=missing-function-docstring, unused-import
import sys
import subprocess
import pytest
import torch
from torch._inductor.codecache import PyCodeCache
from torch.testing._internal.common_utils import (
    TestCase as TorchTestCase,
    instantiate_parametrized_tests,
    parametrize,
    subtest,
)
from torch._inductor.utils import fresh_inductor_cache
import torch_musa
from torch_musa.testing.base_test_tool import _HAS_TRITON


@instantiate_parametrized_tests
@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
class TestCodegenTriton(TorchTestCase):
    """Test class of CodegenTriton"""

    def setup_class(self):
        # ensure the cache is empty
        PyCodeCache.cache_clear()

    def get_compiled_module(self):
        compiled_module = None
        for v in PyCodeCache.modules:
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

    def test_sqrt_match_eager(self):
        def _sqrt(x):
            return torch.sqrt(x)

        x = torch.tensor(
            [68.2929458618164, 23.02151870727539, 93.42744445800781],
            device="musa",
            dtype=torch.float32,
        )
        with fresh_inductor_cache():
            compiled = torch.compile(_sqrt, fullgraph=True)
            result = compiled(x)
        self.assertEqual(result, _sqrt(x), rtol=0, atol=0)

    def test_normalize_match_eager(self):
        def _normalize(x):
            return torch.nn.functional.normalize(x, dim=-1)

        x = torch.tensor(
            [
                [3.799999, 0.0, 0.0],
                [68.2929458618164, 1.0, 0.0],
                [-0.2819808, 1.1561252, -0.3455162],
            ],
            device="musa",
            dtype=torch.float32,
        )
        with fresh_inductor_cache():
            compiled = torch.compile(_normalize, fullgraph=True)
            result = compiled(x)
        self.assertEqual(result, _normalize(x), rtol=0, atol=0)

    def test_normalize_norm_leq_one(self):
        def _normalize(x):
            return torch.nn.functional.normalize(x, dim=-1)

        x = torch.tensor([[3.799999, 0.0, 0.0]], device="musa", dtype=torch.float32)
        with fresh_inductor_cache():
            result = torch.compile(_normalize, fullgraph=True)(x)
        self.assertTrue(torch.all(result.norm(dim=-1) <= 1.0))

    @parametrize(
        "pool_kwargs",
        [
            subtest({}, name="default"),
            subtest({"count_include_pad": False}, name="count_exclude_pad"),
            subtest({"divisor_override": 123}, name="divisor_override"),
        ],
    )
    def test_avg_pool_large(self, pool_kwargs):
        def _avg_pool(x):
            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=(32, 32),
                stride=(32, 32),
                ceil_mode=True,
                **pool_kwargs,
            )

        x = torch.randn((1, 32, 9, 9), device="musa", dtype=torch.float16)
        with fresh_inductor_cache():
            result = torch.compile(_avg_pool, fullgraph=True)(x)
        self.assertEqual(result, _avg_pool(x), rtol=0, atol=0)

    @parametrize(
        "pool_kwargs",
        [
            subtest({}, name="default"),
            subtest({"count_include_pad": False}, name="count_exclude_pad"),
            subtest({"divisor_override": 123}, name="divisor_override"),
        ],
    )
    def test_avg_pool_large_backward(self, pool_kwargs):
        def _avg_pool(x):
            return torch.nn.functional.avg_pool2d(
                x,
                kernel_size=(32, 32),
                stride=(32, 32),
                ceil_mode=True,
                **pool_kwargs,
            ).sum()

        x = torch.randn(
            (1, 1, 9, 9), device="musa", dtype=torch.float32, requires_grad=True
        )
        expected_x = x.detach().clone().requires_grad_()
        expected = _avg_pool(expected_x)
        with fresh_inductor_cache():
            result = torch.compile(_avg_pool, fullgraph=True)(x)
        result.backward()
        expected.backward()
        self.assertEqual(result, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(x.grad, expected_x.grad, rtol=0, atol=0)

    def test_avg_pool1d_large_backward(self):
        def _avg_pool(x):
            return torch.nn.functional.avg_pool1d(
                x, kernel_size=32, stride=32, ceil_mode=True
            ).sum()

        x = torch.randn(
            (1, 1, 9), device="musa", dtype=torch.float32, requires_grad=True
        )
        expected_x = x.detach().clone().requires_grad_()
        expected = _avg_pool(expected_x)
        with fresh_inductor_cache():
            result = torch.compile(_avg_pool, fullgraph=True)(x)
        result.backward()
        expected.backward()
        self.assertEqual(result, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(x.grad, expected_x.grad, rtol=0, atol=0)

    def test_avg_pool3d(self):
        def _avg_pool(x):
            return torch.nn.functional.avg_pool3d(
                x,
                kernel_size=(9, 9, 9),
                stride=(9, 9, 9),
                ceil_mode=True,
            )

        x = torch.randn((1, 4, 9, 9, 9), device="musa", dtype=torch.float16)
        with fresh_inductor_cache():
            result = torch.compile(_avg_pool, fullgraph=True)(x)
        self.assertEqual(result, _avg_pool(x), rtol=0, atol=0)
