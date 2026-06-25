"""Test RNG grid sizing alignment with NVIDIA GPUs.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
from torch.utils._import_utils import _check_module_exists

from torch_musa.core.random import NV_GPU_REGISTRY

skip_if_no_scipy = pytest.mark.skipif(
    not _check_module_exists("scipy"), reason="test requires SciPy"
)

_rng_align_configs = [
    ("default", None),
] + [(name, name) for name in sorted(NV_GPU_REGISTRY.keys())]


@skip_if_no_scipy
@pytest.mark.parametrize("name,rng_align", _rng_align_configs)
def test_randn_normal_kstest_with_rng_align(name, rng_align):
    """randn output should pass KS normality test under any TORCH_RNG_ALIGN."""
    env = os.environ.copy()
    for key in (
        "TORCH_RNG_ALIGN",
        "TORCH_ARCH_MP_COUNT",
        "TORCH_ARCH_MAXTHREADS_PER_MP",
    ):
        env.pop(key, None)
    if rng_align is not None:
        env["TORCH_RNG_ALIGN"] = rng_align

    code = (
        "import json, torch; import torch_musa; "
        "from scipy import stats; "
        "torch.musa.manual_seed(42); "
        "t = torch.randn(4096, device='musa').cpu().to(torch.double); "
        "res = stats.kstest(t.numpy(), 'norm'); "
        "print(json.dumps({'statistic': res.statistic, 'pvalue': res.pvalue}))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tempfile.gettempdir(),
    )
    assert proc.returncode == 0, f"[{name}] subprocess failed:\n{proc.stderr}"
    result = json.loads(proc.stdout.strip())
    assert (
        result["statistic"] < 0.1
    ), f"[{name}] randn KS statistic={result['statistic']:.4f}, expected <0.1"
