"""Test Inductor Triton MM (Matrix Multiplication)"""

import os
import pytest
import torch
from torch import nn
from torch._inductor import config as inductor_config

from torch_musa.testing.base_test_tool import _HAS_TRITON


class SimpleMMModule(nn.Module):
    """
    Simple matrix multiplication module with batch normalization and activation.
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, in_features, out_features, act=True):
        """
        Initialize MM layer with given parameters.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bn1d = nn.BatchNorm1d(out_features)
        self.act = (
            self.default_act
            if act is True
            else act if isinstance(act, nn.Module) else nn.Identity()
        )

    def forward(self, x):
        # Reshape for batch norm if needed (batch norm expects 2D input)
        # x: (batch, in_features)
        return self.act(self.bn1d(self.linear(x)))


@pytest.mark.skipif(not _HAS_TRITON, reason="Triton not installed")
class TestInductorTritonMM:
    """Test class of InductorTritonMM"""

    def test_inductor_triton_mm(self):
        """
        Test method of InductorTritonMM, this test ensures the inductor
        fused mm model accuracy matches golden.
        """
        device = "musa"
        dtype = torch.float16
        in_features = 1024
        out_features = 512
        batch_size = 32

        model = SimpleMMModule(in_features, out_features)
        # typical input: batch of 32, 1024 features
        x = torch.randn(batch_size, in_features)

        # Tune on triton_musa MMA:
        os.environ["ENABLE_MUSA_MMA"] = "1"

        with torch.no_grad():
            golden = model(x)

            model.to(dtype).to(device)
            x = x.to(dtype).to(device)
            y = model(x).cpu().float()
            torch.musa.synchronize()

            torch.testing.assert_close(golden, y, atol=1e-2, rtol=1e-3)

            with inductor_config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_pointwise": True,
                    "max_autotune_gemm": True,
                    # use Triton mm kernel only:
                    "max_autotune_gemm_backends": "TRITON",
                    "max_autotune_gemm_search_space": "DEFAULT",
                }
            ):
                compiled_model = torch.compile(
                    model,
                    backend="inductor",
                    mode="max-autotune",
                    fullgraph=True,
                    dynamic=False,
                )

                y = compiled_model(x).cpu().float()
                torch.musa.synchronize()

                torch.testing.assert_close(golden, y, atol=1e-2, rtol=1e-3)
