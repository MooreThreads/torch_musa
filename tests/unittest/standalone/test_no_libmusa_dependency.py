"""Regression tests for avoiding eager MUSA driver dependencies."""

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import pytest


_DRIVER_SONAME = "libmusa.so.1"
_TORCH_MUSA_LIBS = (
    "libmusa_python.so",
    "libmusa_kernels.so",
    "lib_ext_musa_kernels.so",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _package_dir(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is not None:
        if spec.submodule_search_locations:
            return Path(next(iter(spec.submodule_search_locations))).resolve()
        if spec.origin is not None:
            return Path(spec.origin).resolve().parent

    fallback = _repo_root() / module_name
    if fallback.exists():
        return fallback.resolve()

    raise AssertionError(f"Cannot locate Python package {module_name!r}")


def _single_library(path: Path, name: str) -> Path:
    candidates = sorted(path.glob(f"{name}*"))
    candidates = [candidate for candidate in candidates if candidate.is_file()]
    if not candidates:
        raise AssertionError(f"Cannot find {name} under {path}")
    return candidates[0]


def _shared_libraries_to_check():
    torch_lib = _package_dir("torch") / "lib" / "libtorch_cpu.so"
    if not torch_lib.exists():
        raise AssertionError(f"Cannot find libtorch_cpu.so at {torch_lib}")

    torch_musa_lib_dir = _package_dir("torch_musa") / "lib"
    return [torch_lib] + [
        _single_library(torch_musa_lib_dir, name) for name in _TORCH_MUSA_LIBS
    ]


def _needed_libraries(path: Path):
    if shutil.which("readelf") is None:
        pytest.skip("readelf is required to inspect shared library dependencies")

    proc = subprocess.run(
        ["readelf", "-d", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return re.findall(r"Shared library: \[(.*?)\]", proc.stdout)


# Driver APIs must be loaded lazily; otherwise these import-time dependency
# checks will fail by detecting a direct DT_NEEDED entry on libmusa.so.1. (@fengzhao.zhang)
@pytest.mark.parametrize(
    "library", _shared_libraries_to_check(), ids=lambda path: path.name
)
def test_shared_libraries_do_not_need_musa_driver(library):
    """Import-time libraries should not have a DT_NEEDED entry for libmusa."""
    needed = _needed_libraries(library)
    assert _DRIVER_SONAME not in needed, f"{library} depends on {_DRIVER_SONAME}"


def test_import_torch_musa_without_visible_devices():
    """Importing the wheel should not require visible MUSA devices or lazy init."""
    env = os.environ.copy()
    env["MUSA_VISIBLE_DEVICES"] = ""
    env["MTHREADS_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sys.path if path)

    code = """
import json
import torch
import torch_musa
print(json.dumps({"lazy_initialized": torch_musa.core._lazy_init.is_initialized()}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tempfile.gettempdir(),
    )
    assert proc.returncode == 0, proc.stderr

    stdout = proc.stdout.strip().splitlines()
    assert stdout, "subprocess did not produce import smoke-test output"
    result = json.loads(stdout[-1])
    assert not result["lazy_initialized"]
