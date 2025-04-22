"""AsyncCompile for MUSA"""
import torch

from torch._inductor import config
from torch._inductor.codecache import (
    _compile_start,
    _worker_compile,
    _load_kernel,
    TritonFuture,
    AsyncCompile,
)
from torch._dynamo.device_interface import get_interface_for_device


# FIXME(mingyuan.wang): just curious why there is always a single process when AsyncCompile on CUDA,
# but multi-process on MUSA, which will lead to a re-initialize MUSA error.
# cause fork is default start method in torch2.2, I hardcode compile_threads to 1 for now.
config.compile_threads = 1

class MUSAAsyncCompile(AsyncCompile):
    """subclass of AsyncCompile that compile triton_musa kernels"""
    def triton(
        self, kernel_name: str, source_code: str, device_str: str = "musa"
    ):
        _compile_start()

        if config.compile_threads > 1:
            device_interface = get_interface_for_device(device_str)
            device = torch.device(device_str, device_interface.current_device())
            cc = device_interface.get_compute_capability(device)  # pylint: disable=invalid-name
            future = self.process_pool().submit(
                _worker_compile, kernel_name, source_code, cc, device
            )
            return TritonFuture(kernel_name, source_code, future)

        return _load_kernel(kernel_name, source_code)
