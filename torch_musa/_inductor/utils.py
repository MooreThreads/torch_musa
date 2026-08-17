"""Utilities for inductor"""

__all__ = ["_apply_util_patches"]

# pylint: disable=C0103,C0415,C0116,W0221
import functools
from functools import cached_property
from typing import (
    Union,
)
from typing_extensions import (
    Self,
    Callable,
    Any,
)
import torch
from torch._inductor.runtime.benchmarking import (
    time_and_count,
    Benchmarker,
)
from torch.utils._triton import has_triton_package
from torch._inductor.runtime.hints import DeviceProperties
from torch._inductor.utils import log
from torch._inductor.utils import get_gpu_type

GPU_TYPES = ["musa"]


def is_gpu(device: str):
    assert isinstance(device, str) or device is None, device
    return device in ["cuda", "xpu", torch._C._get_privateuse1_backend_name()]


@functools.lru_cache(None)
def has_triton() -> bool:
    from torch._dynamo.device_interface import get_interface_for_device

    def cuda_extra_check(device_interface):
        return device_interface.Worker.get_device_properties().major >= 7

    def _return_true(_):
        return True

    triton_supported_devices = {
        "cuda": cuda_extra_check,
        "xpu": _return_true,
        torch._C._get_privateuse1_backend_name(): _return_true,
    }

    def is_device_compatible_with_triton():
        for device, extra_check in triton_supported_devices.items():
            device_interface = get_interface_for_device(device)
            if device_interface.is_available() and extra_check(device_interface):
                return True
        return False

    return is_device_compatible_with_triton() and has_triton_package()


class TritonBenchmarker(Benchmarker):
    """TritonBenchmarker

    once do_bench interface aligned with triton upstream, remove this class
    """

    @cached_property
    @time_and_count
    def triton_do_bench(self: Self) -> Callable[..., Any]:
        """Lazily import Triton's `do_bench`."""
        try:
            # pylint: disable=import-outside-toplevel
            from triton.backends.mtgpu.musa_testing import do_bench
        except ImportError as e:
            raise NotImplementedError("requires Triton") from e
        return do_bench

    @time_and_count
    def benchmark_gpu(self: Self, _callable: Callable[[], Any], **kwargs: Any) -> float:
        """Benchmark the GPU callable, `_callable`, and return the runtime, in milliseconds.

        Arguments:
        - _callable: The GPU callable to benchmark.

        Keyword Arguments:
        - quantiles: Optionally, a tuple of floats denoting the requested quantiles.
        - return_mode: Optionally, the requested return mode. Currently, Triton's
        `do_bench` supports min, max, mean, and median return modes.
        - **kwargs: Additional kwargs passed to Triton's `do_bench`.

        Returns:
        - The runtime of `callable`, in milliseconds. If `kwargs["quantiles"]` is specified,
        this is the first requested quantile. Else, if `kwargs["return_mode"]` is specified,
        this is the requested return mode. Otherwise, this is the median.
        """
        kwargs.pop("is_vetted_benchmarking", None)
        if "quantiles" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)[0]
        if "return_mode" in kwargs:
            return self.triton_do_bench(_callable, **kwargs)
        return self.triton_do_bench(_callable, **kwargs, return_mode="median")


# A utility function for easier AOTInductor testing
def aot_inductor_launcher(so_path: str, device: str):
    if device == "musa":
        return f"""
            #include <torch_musa/csrc/inductor/aoti_model_container_runner_musa.h>

            torch::inductor::AOTIModelContainerRunnerMusa runner("{so_path}");

            std::vector<at::Tensor> run(std::vector<at::Tensor>& input_tensors) {{
                return runner.run(input_tensors);
            }}

            std::vector<const char*> get_call_spec() {{
                return runner.get_call_spec();
            }}
        """
    if device == "cpu":
        return f"""
            #include <torch/csrc/inductor/aoti_model_container_runner.h>

            torch::inductor::AOTIModelContainerRunnerCpu runner("{so_path}");

            std::vector<at::Tensor> run(std::vector<at::Tensor>& input_tensors) {{
                return runner.run(input_tensors);
            }}

            std::vector<const char*> get_call_spec() {{
                return runner.get_call_spec();
            }}
        """
    raise RuntimeError(f"Unsupported device: {device}")


@functools.cache
def is_big_gpu(index_or_device: Union[int, torch.device] = 0) -> bool:
    if isinstance(index_or_device, torch.device):
        device = index_or_device
    else:
        device = torch.device(get_gpu_type(), index_or_device)

    prop = DeviceProperties.create(device)

    min_sms = 16 if device.type == "xpu" else 68  # 3080
    try:
        if torch.musa.is_available():
            # Set min_sms to 8 to enable GEMM max_autotune for MUSA M1000 GPUs:
            min_sms = 8  # M1000
    except AttributeError:
        pass

    avail_sms = prop.multi_processor_count
    if avail_sms < min_sms:
        log.warning(
            "Not enough SMs to use max_autotune_gemm mode",
            extra={"min_sms": min_sms, "avail_sms": avail_sms},
        )
        return False
    return True


def _apply_util_patches():
    torch._inductor.utils.is_gpu = is_gpu
    torch._inductor.runtime.benchmarking.benchmarker = TritonBenchmarker()
    torch.utils._triton.has_triton = has_triton
    torch._inductor.utils.GPU_TYPES = GPU_TYPES
    torch._inductor.utils.is_big_gpu = is_big_gpu
