"""Test context managers in dynamo tracing"""

# pylint: disable=missing-function-docstring, unused-import
import pytest
import torch
import torch._dynamo.test_case
import torch._dynamo.testing


class TestCtxManager:
    """Test class of Context Manager"""

    @pytest.mark.parametrize("device", ["cpu", "musa"])
    def test_autocast(self, device):
        if device == "musa":
            import torch_musa  # pylint: disable=import-outside-toplevel

        def func_musa(x, y):
            # with autocast:
            with torch.musa.amp.autocast(enabled=True):
                out = torch.matmul(x, y)
            return out

        def func_cpu(x, y):
            with torch.amp.autocast("cpu", enabled=True):
                out = torch.matmul(x, y)
            return out

        x = torch.randn((8, 8), device=device, requires_grad=True)
        y = torch.randn((8, 8), device=device, requires_grad=True)

        func = func_musa if device == "musa" else func_cpu

        ref = func(x, y)
        cnts_backend = torch._dynamo.testing.CompileCounter()
        opt_func = torch._dynamo.optimize(cnts_backend)(func)
        res = opt_func(x, y)

        assert ref.dtype == res.dtype
        assert cnts_backend.op_count == 3  # enter + matmul + exit
        assert cnts_backend.frame_count == 1  # no graph break
