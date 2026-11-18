"""Exercise every public header under ``torch_musa/csrc/include`` from a
downstream extension built with MUSAExtension.
"""

# pylint: disable=missing-function-docstring, redefined-outer-name

import importlib
import os
import subprocess
import sys

import pytest

torch = pytest.importorskip("torch")
torch_musa = pytest.importorskip("torch_musa")


CPP_SOURCE = r"""
#include <c10/core/DeviceType.h>

// Every header under torch_musa/csrc/include/
#include <c10/musa/MUSAAllocatorConfig.h>
#include <c10/musa/MUSACachingAllocator.h>
#include <c10/musa/MUSADeviceAssertionHost.h>
#include <c10/musa/MUSAEvent.h>
#include <c10/musa/MUSAException.h>
#include <c10/musa/MUSAFunctions.h>
#include <c10/musa/MUSAGraphsC10Utils.h>
#include <c10/musa/MUSAGuard.h>
#include <c10/musa/MUSAMiscFunctions.h>
#include <c10/musa/MUSAStream.h>
#include <c10/musa/PeerToPeerAccess.h>
#include <ATen/detail/MUSAHooksInterface.h>

#include <musa_runtime.h>

#include <torch/extension.h>

namespace {

int run_smoke() {
  c10::register_privateuse1_backend("musa");
  at::detail::getMUSAHooks().initMUSA();

  TORCH_CHECK(c10::musa::is_musa_available(), "MUSA is not available");
  const auto n = c10::musa::device_count();
  TORCH_CHECK(n > 0, "no MUSA device available");

  c10::musa::MUSAGuard guard(0);

  musaStream_t raw = nullptr;
  TORCH_CHECK(
      musaStreamCreateWithFlags(&raw, musaStreamNonBlocking) == musaSuccess,
      "musaStreamCreateWithFlags failed");

  {
    auto stream = c10::musa::getStreamFromExternal(raw, 0);
    c10::musa::MUSAStreamGuard stream_guard(stream);
    C10_MUSA_CHECK(musaGetLastError());
    c10::musa::Synchronize();
  }

  {
    c10::musa::MUSAEvent event(musaEventDisableTiming);
    event.record(c10::musa::getStreamFromExternal(raw, 0));
    event.synchronize();
    TORCH_CHECK(event.isCreated(), "event was not created");
    TORCH_CHECK(event.query(), "event did not complete");
  }

  TORCH_CHECK(
      c10::musa::currentStreamCaptureStatusMayInitCtx() ==
      c10::musa::CaptureStatus::None, "unexpected stream capture status");

  TORCH_CHECK(
      c10::musa::get_musa_check_suffix() != nullptr,
      "missing MUSA check suffix");

  TORCH_CHECK(
      c10::musa::get_p2p_access(0, 0), "device 0 cannot access itself");

  TORCH_CHECK(
      c10::musa::MUSACachingAllocator::MUSAAllocatorConfig::pinned_num_register_threads() > 0,
      "unexpected pinned register thread count");

  const auto& launch_registry =
      c10::musa::MUSAKernelLaunchRegistry::get_singleton_ref();
  TORCH_CHECK(
      !launch_registry.has_failed(),
      "unexpected device-side assertion failure");
  TORCH_CHECK(
      C10_MUSA_DSA_ASSERTION_COUNT > 0, "unexpected DSA assertion count");


  c10::musa::MUSACachingAllocator::init(static_cast<int>(n));
  c10::musa::MUSACachingAllocator::setMemoryFraction(0.5, 0);
  c10::musa::MUSACachingAllocator::emptyCache();

  c10::musa::MempoolId_t pool{1, 1};
  c10::musa::MUSACachingAllocator::beginAllocateToPool(
      0, pool, [](musaStream_t) { return true; });
  c10::musa::MUSACachingAllocator::endAllocateToPool(0, pool);
  c10::musa::MUSACachingAllocator::releasePool(0, pool);

  TORCH_CHECK(
      musaStreamDestroy(raw) == musaSuccess, "musaStreamDestroy failed");
  return static_cast<int>(n);
}

} // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run_smoke", &run_smoke, "Exercise the packaged public MUSA headers");
}
"""


SETUP_SOURCE = r"""
from setuptools import setup
from torch_musa.utils.musa_extension import BuildExtension, MUSAExtension

setup(
    name="musa_public_headers_example",
    ext_modules=[
        MUSAExtension(
            "musa_public_headers_example",
            ["binding.cpp"],
            libraries=["musart", "musa"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
"""


pytestmark = pytest.mark.skipif(
    not torch.musa.is_available() or torch.musa.device_count() == 0,
    reason="requires at least one MUSA device",
)


@pytest.fixture(scope="module")
def extension_module(tmp_path_factory):
    work_dir = tmp_path_factory.mktemp("musa_public_headers")
    (work_dir / "binding.cpp").write_text(CPP_SOURCE, encoding="utf-8")
    (work_dir / "setup.py").write_text(SETUP_SOURCE, encoding="utf-8")

    env = os.environ.copy()
    package_root = os.path.dirname(os.path.dirname(torch_musa.__file__))
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [package_root, env.get("PYTHONPATH")])
    )
    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=work_dir,
        env=env,
        check=True,
    )

    sys.path.insert(0, str(work_dir))
    try:
        yield importlib.import_module("musa_public_headers_example")
    finally:
        sys.path.remove(str(work_dir))
        sys.modules.pop("musa_public_headers_example", None)


def test_public_headers_example(extension_module):
    device_count = extension_module.run_smoke()
    assert device_count > 0
