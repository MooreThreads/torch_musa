"""MUSATritonWrapperCodeGen"""

# pylint: disable=all
import contextlib
import dataclasses
from typing import Optional

import sympy

from torch._dynamo.utils import dynamo_timed

from torch.utils._sympy.singleton_int import SingletonInt

from torch._inductor import config
from torch._inductor.virtualized import V
from torch._inductor.codegen.common import IndentedBuffer
from torch._inductor.codegen.wrapper import (
    MemoryPlanningLine,
    WrapperCodeGen,
)
from torch._inductor.utils import cache_on_self

from ..utils import triton_is_available
from .. import codecache


@dataclasses.dataclass
class EnterMUSADeviceContextManagerLine:
    device_idx: int
    last_seen_device_guard_index: Optional[int]

    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        assert not V.graph.cpp_wrapper

        # Note _DeviceGuard has less overhead than device, but only accepts
        # integers
        code.writeline(f"with torch.musa._DeviceGuard({self.device_idx}):")
        device_cm_stack.enter_context(code.indent())
        code.writeline(
            f"torch.musa.set_device({self.device_idx}) # no-op to ensure context"
        )


class ExitMUSADeviceContextManagerLine:
    def codegen(self, code: IndentedBuffer, device_cm_stack: contextlib.ExitStack):
        if not V.graph.cpp_wrapper:
            device_cm_stack.close()


class MUSATritonWrapperCodeGen(WrapperCodeGen):
    """
    Generate outer wrapper in Python that calls the kernels.
    """

    def write_header(self):
        self.header.splice(
            f"""
                from ctypes import c_void_p, c_long
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

                from torch import device, empty, empty_strided
                from {codecache.__name__} import MUSAAsyncCompile
                from torch._inductor.select_algorithm import extern_kernels

                aten = torch.ops.aten
                inductor_ops = torch.ops.inductor
                assert_size_stride = torch._C._dynamo.guards.assert_size_stride
                alloc_from_pool = torch.ops.inductor._alloc_from_pool
                reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
                async_compile = MUSAAsyncCompile()

            """
        )

    @cache_on_self
    def write_triton_header_once(self):
        if triton_is_available():
            self.header.splice(
                """
                    import triton
                    import triton.language as tl
                    from torch._inductor.triton_heuristics import grid, start_graph, end_graph
                    from torch_musa._MUSAC import _musa_getCurrentRawStream as get_musa_stream
                """
            )

    def write_prefix(self):
        self.prefix.splice(
            """

            async_compile.wait(globals())
            del async_compile

            def call(args):
            """
        )
        with self.prefix.indent():
            if config.triton.debug_sync_graph:
                self.prefix.writeline("torch.musa.synchronize()")
            inp_len = len(V.graph.graph_inputs.keys())
            if inp_len != 0:
                lhs = f"{', '.join(V.graph.graph_inputs.keys())}{'' if inp_len != 1 else ','}"
                self.prefix.writeline(f"{lhs} = args")
                self.prefix.writeline("args.clear()")

            self.codegen_inputs(self.prefix, V.graph.graph_inputs)
            if config.size_asserts:
                self.codegen_input_size_asserts()

    def write_get_raw_stream(self, index):
        self.write_triton_header_once()
        name = f"stream{index}"
        self.writeline(f"{name} = get_musa_stream({index})")
        return name

    def write_get_musa_stream(self, index):
        return self.write_get_raw_stream(index)

    def codegen_device_guard_enter(self, device_idx):
        self.writeline(
            EnterMUSADeviceContextManagerLine(device_idx, self.last_seen_device_guard_index)
        )

    def codegen_device_guard_exit(self):
        self.writeline(ExitMUSADeviceContextManagerLine())

    @dynamo_timed
    def generate(self, is_inference):
        if config.profile_bandwidth:
            self.write_triton_header_once()
        result = IndentedBuffer()
        result.splice(self.header)

        with contextlib.ExitStack() as stack:
            stack.enter_context(self.wrapper_call.indent())
            if config.profiler_mark_wrapper_call:
                self.generate_profiler_mark_wrapper_call(stack)
            if config.profile_bandwidth:
                self.generate_start_graph()

            # We disable planning during training because it presently
            # increases peak memory consumption.
            if is_inference and config.memory_planning:
                self.memory_plan()
            else:
                self.memory_plan_reuse()

            device_cm_stack = contextlib.ExitStack()
            for line in self.lines:
                if isinstance(line, MemoryPlanningLine):
                    line.codegen(self.wrapper_call)
                elif isinstance(
                    line,
                    (
                        EnterMUSADeviceContextManagerLine,
                        ExitMUSADeviceContextManagerLine,
                    ),
                ):
                    line.codegen(self.wrapper_call, device_cm_stack)
                else:
                    self.wrapper_call.writeline(line)
            output_refs = self.get_output_refs()
            self.mark_output_type()
            if config.triton.debug_sync_graph:
                self.wrapper_call.writeline("torch.musa.synchronize()")

            if config.profile_bandwidth:
                self.generate_end_graph()

            self.generate_return(output_refs)

        self.append_precomputed_sizes_to_prefix()
        self.finalize_prefix()
        result.splice(self.prefix)

        with result.indent():
            result.splice(self.wrapper_call)

        self.generate_end(result)

        self.add_benchmark_harness(result)

        return result.getvaluewithlinemap()

    def benchmark_compiled_module(self, output):
        def add_fake_input(name, shape, stride, device, dtype):
            output.writeline(
                f"{name} = rand_strided("
                f"{self.codegen_python_shape_tuple(shape)}, "
                f"{self.codegen_python_shape_tuple(stride)}, "
                f"device='{device}', dtype={dtype})"
            )

        def add_expr_input(name, val):
            output.writeline(f"{name} = {val}")

        output.writelines(
            ["", "", "def benchmark_compiled_module(times=10, repeat=10):"]
        )
        with output.indent():
            output.splice(
                """
                from torch._dynamo.testing import rand_strided
                from torch_musa._inductor.utils import print_performance
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

            for name, value in V.graph.graph_inputs.items():
                if isinstance(value, sympy.Symbol) and isinstance(
                    V.graph.sizevars.var_to_val.get(value, None), SingletonInt
                ):
                    # Inductor should only work with dense -> dense graph, and
                    # SingletonInts belong to metadata that should only live on
                    # the subclass.
                    continue
                if isinstance(value, sympy.Expr):  # Don't need to add symbolic
                    add_expr_input(name, V.graph.sizevars.size_hint(value))
                else:
                    shape = [V.graph.sizevars.size_hint(x) for x in value.get_size()]
                    stride = [V.graph.sizevars.size_hint(x) for x in value.get_stride()]
                    add_fake_input(
                        name, shape, stride, value.get_device(), value.get_dtype()
                    )

            call_str = f"call([{', '.join(V.graph.graph_inputs.keys())}])"
            output.writeline(f"fn = lambda: {call_str}")
            output.writeline("return print_performance(fn, times=times, repeat=repeat)")

    def add_benchmark_harness(self, output):
        """
        Append a benchmark harness to generated code for debugging
        """
        super().add_benchmark_harness(output)
