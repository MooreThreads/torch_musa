"""Implement MUSATritonWrapperCodeGen"""

import dataclasses
from typing import Optional
import sympy

import torch
from torch._inductor import config
from torch.utils._sympy.singleton_int import SingletonInt
from torch._inductor.codegen.common import IndentedBuffer

from torch._inductor.virtualized import V
from torch._inductor.codegen.wrapper import WrapperCodeGen
from torch._inductor.codegen.wrapper import EnterDeviceContextManagerLine


@dataclasses.dataclass
class EnterMUSADeviceContextManagerLine(EnterDeviceContextManagerLine):
    """
    Context manager line for entering MUSA device guard.

    This dataclass represents a step in code generation where the runtime
    must switch to the correct MUSA device or stream. Depending on AOT (ahead-of-time)
    compilation mode and ABI compatibility, it emits different guard statements.
    """

    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer) -> None:
        if V.graph.cpp_wrapper:
            code.writeline("\n")
            if V.graph.aot_mode:
                # In AOT mode, we have a stream provided as a param. A stream is
                # associated with a device, so we never expect the device to change.
                # CUDAStreamGuard sets the stream and the device.
                if self.last_seen_device_guard_index is None:
                    if config.abi_compatible:
                        code.writeline(
                            "AOTIMusaStreamGuard stream_guard(stream, this->device_idx_);"
                        )
                    else:
                        code.writeline(
                            "c10::musa::MUSAStreamGuard stream_guard("
                            + "c10::musa::getStreamFromExternal(stream, this->device_idx_));"
                        )
                else:
                    assert (
                        self.last_seen_device_guard_index == self.device_idx
                    ), "AOTInductor only supports running on one MUSA device"
            else:
                if self.last_seen_device_guard_index is None:
                    code.writeline(
                        f"AOTIMusaGuard device_guard({self.device_idx});"
                        if config.abi_compatible
                        else f"c10::musa::MUSAGuard device_guard({self.device_idx});"
                    )
                else:
                    code.writeline(f"device_guard.set_index({self.device_idx});")
        else:
            # Note _DeviceGuard has less overhead than device, but only accepts
            # integers
            code.writeline(f"with {V.graph.device_ops.device_guard(self.device_idx)}:")
            code.do_indent()
            code.writeline(V.graph.device_ops.set_device(self.device_idx))


class MUSATritonWrapperCodeGen(WrapperCodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def write_header(self):
        """write header

        TODO(mingyuan.wang): maybe we dont need to overwrite this
        if `torch_musa` is imported automatically
        """
        context = torch._guards.TracingContext.try_get()
        aot_config_comment = ""
        if context is not None and context.aot_graph_name is not None:
            aot_config_comment = f"# AOT ID: {context.aot_graph_name}"
        self.imports.splice(
            f"""
                {aot_config_comment}
                from ctypes import c_void_p, c_long, c_int
                import torch
                import torch_musa
                import math
                import random
                import os
                import tempfile
                from math import inf, nan
                from torch._inductor.hooks import run_intermediate_hooks
                from torch._inductor.utils import maybe_profile
                from torch._inductor.codegen.memory_planning import _align as align
                from torch import device, empty_strided
                from torch._inductor.async_compile import AsyncCompile
                from torch._inductor.select_algorithm import extern_kernels
                from torch._inductor.codegen.multi_kernel import MultiKernelCall
            """,
            strip=True,
        )
        self.header.splice(
            """
                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                _quantized = torch.ops._quantized
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
                #empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
                empty_strided_musa = torch_musa._MUSAC._empty_strided_musa
                reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                async_compile = AsyncCompile()
            """,
            strip=True,
        )

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        assert not V.graph.cpp_wrapper, "CppWrapper is not supported yet"
        super().codegen_device_guard_enter(device_idx)

    # def codegen_device_guard_exit(self):
    #     super().codegen_device_guard_exit()

    def benchmark_compiled_module(self, output):
        """benchmark the compiled module

        The only difference with WarpperCodeGen is that we need to specify 'musa' as the device
        argument for the print_performance function.
        """

        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        def add_torchbind_input(name, value):
            import pickle  # pylint:disable=import-outside-toplevel

            output.writeline(f"{name} = pickle.loads({pickle.dumps(value)!r})")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch._inductor.utils import print_performance
                """,
                strip=True,
            )

            for name, value in V.graph.constants.items():
                # all the constants are global variables, that's why we need
                # these 'global var_name' lines
                output.writeline(f"global {name}")
                add_fake_input(
                    name, value.size(), value.stride(), value.device, value.dtype
                )

            if len(V.graph.torchbind_constants) > 0:
                output.writeline("import pickle")
                for name, torchbind_obj in V.graph.torchbind_constants.items():
                    # all the constants are global variables, that's why we need
                    # these 'global var_name' lines
                    output.writeline(f"global {name}")
                    add_torchbind_input(name, torchbind_obj)

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    # TODO: this fallback and those below actually will generate possibly
                    # invalid benchmark code, because it's not guaranteed 42
                    # is actually a valid value for the kernel in question.
                    # See https://github.com/pytorch/pytorch/issues/124686
                    add_expr_input(name, V.graph.sizevars.size_hint(value, fallback=42))
                else:
                    shape = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_size()
                    ]
                    stride = [
                        V.graph.sizevars.size_hint(x, fallback=42)
                        for x in value.get_stride()
                    ]
                    add_fake_input(
                        name,
                        shape,
                        stride,
                        value.get_device(),
                        value.get_dtype(),
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline(
                'return print_performance(fn, times=times, repeat=repeat, device="musa")'
            )

    # def add_benchmark_harness(self, output):
    #     """
    #     Append a benchmark harness to generated code for debugging
    #     """
    #     super().add_benchmark_harness(output)

    def make_allocation(self, name, device, dtype, shape, stride):
        assert device.type == "musa", f"Unexcepted device type: {device.type}"
        # optimized path for faster allocations, saving ~2us versus the empty_strided
        return (
            f"{name} = empty_strided_musa("
            f"{self.codegen_shape_tuple(shape)}, "
            f"{self.codegen_shape_tuple(stride)}, "
            f"{dtype})"
        )
