"""Test the codecache"""

# pylint: disable=C0115, C0116
import tempfile
from unittest.mock import patch
import pytest

import torch
import torch._inductor.config
from torch._dynamo.utils import counters
from torch_musa.testing.base_test_tool import _HAS_TRITON
from torch_musa._inductor.codecache import MUSAAsyncCompile


class LinearModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(8, 16)

    def forward(self, x):
        return self.fc1(x)


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
class TestWorkerStartMethod:
    """Test the start_method of MUSAAsyncCompile"""

    # FIXME: multiprocessing re-initialize MUSA error needs to be fixed
    _COMPILE_THREADS = 1

    def test_codecache_spawn(self):
        with torch._inductor.config.patch(
            worker_start_method="spawn",
            compile_threads=TestWorkerStartMethod._COMPILE_THREADS,
        ):
            MUSAAsyncCompile.warm_pool()

            device = torch.musa.current_device()
            model = LinearModel().to(device)
            model = torch.compile(model)
            x = torch.randn(8, 8).to(device)
            model(x).sum().backward()

    def test_codecache_fork(self):
        with torch._inductor.config.patch(
            worker_start_method="fork",
            compile_threads=TestWorkerStartMethod._COMPILE_THREADS,
        ):
            device = torch.musa.current_device()
            model = LinearModel().to(device)
            model = torch.compile(model)
            x = torch.randn(8, 8).to(device)
            model(x).sum().backward()


class Conv2dModel(torch.nn.Module):
    def __init__(self, dim=512):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        torch._dynamo.graph_break()
        x = self.conv2(x)
        return x


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not enabled")
class TestFXGraphCache:
    def setup_class(self):
        # pylint: disable=consider-using-with
        self.tmpdir = tempfile.TemporaryDirectory()
        self.cache_dir_patch = patch("torch._inductor.codecache.cache_dir")
        self.cache_dir_patch.start().return_value = self.tmpdir.name

    def teardown_class(self):
        self.cache_dir_patch.stop()
        self.tmpdir.cleanup()

    def setup_method(self):
        counters.clear()

    @pytest.mark.parametrize("device", ["musa", "cpu"])
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
        ],
    )
    @pytest.mark.parametrize("dynamic", [True, False])
    def test_cache_load_function(self, device, dtype, dynamic):
        """
        Verify that we can populate and load functions from the cache.
        """
        with torch._inductor.config.patch(fx_graph_cache=True):

            def func(x, y):
                return x * 2 + y @ y

            a = torch.rand(5, 5, dtype=dtype, device=device)
            b = torch.rand(5, 5, dtype=dtype, device=device)
            c = a.view(5, 5)

            compiled_func = torch.compile(func, dynamic=dynamic)

            # a first call should miss in the cache
            assert torch.allclose(func(a, b), compiled_func(a, b))
            assert counters["inductor"]["fxgraph_cache_miss"] == 1
            assert counters["inductor"]["fxgraph_cache_hit"] == 0

            # a second call should hit
            torch._dynamo.reset()
            assert torch.allclose(func(a, b), compiled_func(a, b))
            assert counters["inductor"]["fxgraph_cache_miss"] == 1
            assert counters["inductor"]["fxgraph_cache_hit"] == 1

            # cache miss this time
            torch._dynamo.reset()
            assert torch.allclose(func(a, c), compiled_func(a, c))
            assert counters["inductor"]["fxgraph_cache_miss"] == 2
            assert counters["inductor"]["fxgraph_cache_hit"] == 1

    @pytest.mark.parametrize(
        "device",
        [
            "musa",
            "cpu",
        ],
    )
    @pytest.mark.parametrize(
        "dtype",
        [
            torch.float32,
        ],
    )
    @pytest.mark.parametrize("dynamic", [True, False])
    def test_cache_load_model(self, device, dtype, dynamic):
        """
        Verify that we can populate and load models from the cache.
        """
        with torch._inductor.config.patch(fx_graph_cache=True):

            def func(mod, x):
                mod.zero_grad()
                mod(x).sum().backward()
                return [p.grad for p in mod.parameters()]

            compiled_fn = torch.compile(func, dynamic=dynamic)
            mod = Conv2dModel().to(device=device, dtype=dtype)
            x = torch.randn(2, 3, 16, 16, device=device, dtype=dtype)

            # the first call should see all cache misses.
            counters.clear()
            grads1 = compiled_fn(mod, x)
            assert counters["inductor"]["fxgraph_cache_miss"] > 0
            assert counters["inductor"]["fxgraph_cache_hit"] == 0

            # The second should see all hits. (First reset so in-memory guards
            # don't prevent compilation).
            counters.clear()
            torch._dynamo.reset()
            grads2 = compiled_fn(mod, x)
            assert counters["inductor"]["fxgraph_cache_miss"] == 0
            assert counters["inductor"]["fxgraph_cache_hit"] > 0

            # And the results should be the same.
            for grad1, grad2 in zip(grads1, grads2):
                assert torch.allclose(grad1, grad2)
