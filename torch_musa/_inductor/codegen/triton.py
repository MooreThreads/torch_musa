# Borrowed from Torch
# pylint: disable=all
import itertools
from typing import Optional, Set

import sympy
import textwrap

import torch

from torch._inductor.ir import (
    ReductionHint,
    IRNode,
)
from torch._inductor import config, scheduler
from torch._dynamo.utils import counters

from torch._inductor.utils import (
    Placeholder,
    unique,
)

from torch._inductor.virtualized import V

from torch._inductor.codegen.common import (
    IndentedBuffer,
    PythonPrinter,
    SizeArg,
)
from torch._inductor.codegen.triton import (
    signature_of,
    config_of,
    signature_to_meta,
    TritonPrinter,
    TritonOverrides,
    TritonKernel,
    TritonScheduling,
    DisableReduction,
    EnableReduction,
    log,
)

class MUSATritonPrinter(TritonPrinter):
    def __init__(self):
        super().__init__()


texpr = MUSATritonPrinter().doprint
pexpr = PythonPrinter().doprint


class MUSATritonOverrides(TritonOverrides):
    """Map element-wise ops to MUSA Triton backend"""


class MUSATritonKernel(TritonKernel):
    overrides = MUSATritonOverrides
    sexpr = pexpr

    def __init__(
        self,
        *groups,
        index_dtype: str,
        mutations: Optional[Set[str]] = None,
        pid_cache=None,
        reduction_hint=ReductionHint.DEFAULT,
        min_elem_per_thread=0,
    ):
        super().__init__(
            *groups,
            index_dtype=index_dtype,
            mutations=mutations,
            pid_cache=pid_cache,
            reduction_hint=reduction_hint,
            min_elem_per_thread=min_elem_per_thread
        )

    def codegen_kernel_benchmark(self):
        result = IndentedBuffer()
        argdefs, call_args, signature = self.args.python_argdefs()

        result.writelines(["", "", "def get_args():"])
        with result.indent():
            name_cnt = itertools.count()
            var_names = []
            for arg_name, arg_sig in zip(call_args, signature):
                var_name = f"arg_{next(name_cnt)}"
                buf = V.graph.get_buffer(arg_name)
                if buf:
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(buf.get_size())}, {V.graph.sizevars.size_hints(buf.get_stride())}, device='{buf.get_device()}', dtype={buf.get_dtype()})"  # noqa: B950 line too long
                    )
                elif arg_name in V.graph.constants:
                    # note that random seed is put in V.graph.constants
                    const_tensor = V.graph.constants[arg_name]
                    result.writeline(
                        f"{var_name} = rand_strided({V.graph.sizevars.size_hints(const_tensor.size())}, {V.graph.sizevars.size_hints(const_tensor.stride())}, device='{const_tensor.device}', dtype={const_tensor.dtype})"  # noqa: B950 line too long
                    )
                elif isinstance(arg_sig, SizeArg):
                    symval_hint = V.graph.sizevars.size_hint(arg_sig.expr)

                    # Force the seed_offset to be 0 so calls to the same kernel
                    # using different seed offset will have the same benchmark harness.
                    # We can dedup kernel definitions in this case.
                    if "seed_offset" in arg_sig.name:
                        symval_hint = 0
                    result.writeline(f"{var_name} = {symval_hint}")
                else:
                    raise KeyError(
                        f"Don't find the buffer or const tensor for {arg_name}"
                    )
                var_names.append(var_name)
            result.writeline(f"return {', '.join(var_names)},")

        result.writelines(["\n", "\n", "def call(args):"])
        grid = []
        extra_args = []
        extra_args_str = None
        index = V.graph.scheduler.current_device.index
        with result.indent():
            result.writeline(f"with torch.musa._DeviceGuard({index}):")
            with result.indent():
                result.writeline(
                    f"torch.musa.set_device({index})"
                )  # no-op to ensure context
                for tree in self.range_trees:
                    expr = pexpr(V.graph.sizevars.size_hint(tree.numel))
                    if tree.prefix != "r" or self.inside_reduction:
                        extra_args.append(expr)
                    if tree.prefix != "r":
                        grid.append(expr)

                stream_name = f"stream{index}"
                result.writeline(f"{stream_name} = get_musa_stream({index})")

                if self.need_numel_args():
                    extra_args_str = ", ".join(map(str, extra_args)) + ", "
                else:
                    extra_args_str = ""

                result.writeline(
                    f"{str(Placeholder.KERNEL_NAME)}.run(*args, {extra_args_str}grid=grid({', '.join(grid)}), stream={stream_name})"
                )

        # benchmark all configs
        result.writelines(["\n", "\n", "def benchmark_all_configs(args):"])
        with result.indent():
            result.writeline(f"with torch.musa._DeviceGuard({index}):")
            with result.indent():
                result.writeline(
                    f"torch.musa.set_device({index})"
                )  # no-op to ensure context
                result.writeline(
                    f"return {str(Placeholder.KERNEL_NAME)}.benchmark_all_configs(*args, {extra_args_str}grid=grid({', '.join(grid)}))"  # noqa: B950 line too long
                )

        ninplace_args = len(unique(self.args.inplace_buffers.values()))
        result.writelines(["\n", "\n", "if __name__ == '__main__':"])
        with result.indent():
            result.writeline("from torch._inductor.utils import get_num_bytes")
            result.writeline("from triton.testing import do_bench")
            result.writeline("")

            result.writeline("args = get_args()")
            result.writeline(
                "ms = do_bench(lambda: call(args), rep=40, fast_flush=True)"
            )
            result.writeline(
                f"num_gb = get_num_bytes(*args, num_in_out_args={ninplace_args}) / 1e9"
            )
            result.writeline("gb_per_s = num_gb / (ms / 1e3)")
            result.writeline(
                'print(f"{ms:.3f}ms    {num_gb:.3f}GB    {gb_per_s:.2f}GB/s")'
            )

        return result

    def imports_for_benchmark_kernel(self):
        return textwrap.dedent(
            """
            from torch._dynamo.testing import rand_strided
            from torch_musa._MUSAC import _musa_getCurrentRawStream as get_musa_stream
            import torch
            from torch._inductor.triton_heuristics import grid
        """
        )

    def call_kernel(self, name: str, node: Optional[IRNode] = None):
        wrapper = V.graph.wrapper_code
        _, call_args, _ = self.args.python_argdefs()
        # dynamo wraps unspec variable as 0d CPU tensor, need convert to scalar
        for i in range(len(call_args)):
            if V.graph.is_unspec_arg(call_args[i]):
                call_args[i] = call_args[i] + ".item()"
        grid = []
        # TODO(jansel): if there are constants, we shouldn't bother passing them as args
        for tree in self.range_trees:
            if isinstance(tree.numel, (sympy.Integer, sympy.Symbol)):
                expr = tree.numel
            else:
                expr = wrapper.generate_numel_expr(name, tree)

            if tree.prefix != "r" or self.inside_reduction:
                call_args.append(expr)
            if tree.prefix != "r":
                grid.append(expr)

        grid = wrapper.generate_default_grid(name, grid)

        call_args_str = ", ".join(pexpr(item) for item in call_args)
        stream_name = wrapper.write_get_raw_stream(
            V.graph.scheduler.current_device.index
        )
        grid_str = ", ".join(pexpr(item) for item in grid)
        wrapper.writeline(
            f"{name}.run({call_args_str}, grid=grid({grid_str}), stream={stream_name})"
        )

    def codegen_kernel(self, name=None):
        from triton import next_power_of_2

        code = IndentedBuffer()

        size_hints = []
        for numel in self.numels:
            numel_hint = V.graph.sizevars.symbolic_hint(numel)
            if not isinstance(numel_hint, (int, sympy.Integer)):
                # This default heuristic hint was picked carefully: it is
                # large, to ensure that we don't shrink the block size (since
                # if you don't have many elements, it'd be wasteful to pick a
                # large block size).  Since we don't know how many elements we
                # might have, we should be OK with some inefficiency to make
                # sure we handle the large case well.  8192 is the largest
                # block size we support, so we pick that.
                #
                # If we have a better hint for unbacked SymInts (e.g., because
                # a user told us, or we are tracking upper bounds) we could
                # use that here.
                size_hint = 8192
            else:
                size_hint = next_power_of_2(int(numel_hint))
            size_hints.append(size_hint)
        if self.persistent_reduction:
            assert self.inside_reduction
            heuristics = "persistent_reduction"
        elif self.inside_reduction:
            heuristics = "reduction"
        else:
            size_hints.pop()
            heuristics = "pointwise"

        if name is None:
            code.splice(
                f"""
                    import triton
                    import triton.language as tl
                    from torch._inductor.ir import ReductionHint
                    from torch._inductor.ir import TileHint
                    from torch_musa._inductor.triton_heuristics import AutotuneHint, {heuristics}
                    from torch._inductor.utils import instance_descriptor
                    from torch._inductor import triton_helpers
                """
            )
            if config.benchmark_kernel:
                code.splice(self.imports_for_benchmark_kernel())

        argdefs, _, signature = self.args.python_argdefs()
        # maps actual expression to SizeArg if its in sizevars replacements
        for i, arg in enumerate(signature):
            if (
                isinstance(arg, SizeArg)
                and arg.expr in V.graph.sizevars.inv_precomputed_replacements
            ):
                signature[i] = SizeArg(
                    arg.name, V.graph.sizevars.inv_precomputed_replacements[arg.expr]
                )

        mutated_args = set()
        for mutation in self.mutations:
            if mutation in self.args.input_buffers:
                mutated_args.add(self.args.input_buffers[mutation])
            if (
                mutation in self.args.inplace_buffers
                and mutation not in V.graph.removed_buffers
                and mutation not in self.removed_buffers
            ):
                mutated_args.add(self.args.inplace_buffers[mutation].inner_name)
            if mutation in self.args.output_buffers:
                mutated_args.add(self.args.output_buffers[mutation])
        mutated_args = sorted(mutated_args)

        triton_meta_signature = signature_to_meta(
            signature, size_dtype=self.index_dtype
        )
        triton_meta = {
            "signature": triton_meta_signature,
            "device": V.graph.scheduler.current_device.index,
            "device_type": V.graph.scheduler.current_device.type,
            "constants": {},
        }

        inductor_meta = {
            "autotune_hints": set(self.autotune_hints),
            "kernel_name": str(Placeholder.DESCRIPTIVE_NAME),
            "mutated_arg_names": mutated_args,
        }

        for tree in self.range_trees:
            if tree.prefix != "r" or self.inside_reduction:
                sizearg = SizeArg(f"{tree.prefix}numel", tree.numel)
                signature.append(sizearg)
                triton_meta_signature[len(argdefs)] = signature_of(
                    sizearg, size_dtype=self.index_dtype
                )
                argdefs.append(f"{tree.prefix}numel")
                # constexpr version causes issues, see
                # https://github.com/pytorch/torchdynamo/pull/1362
                # triton_meta["constants"][len(argdefs)] = V.graph.sizevars.size_hint(
                #     tree.numel
                # )
                # argdefs.append(f"{tree.prefix}numel: tl.constexpr")
        triton_meta["configs"] = [config_of(signature)]

        for tree in self.range_trees:
            if tree.prefix == "r" and (
                not self.inside_reduction or self.persistent_reduction
            ):
                continue
            if tree.prefix == "x" and self.no_x_dim:
                continue
            argdefs.append(f"{tree.prefix.upper()}BLOCK : tl.constexpr")

        if self.inside_reduction:
            reduction_hint = self.reduction_hint
            heuristics_line = f"""
                @{heuristics}(
                    size_hints={size_hints!r},
                    reduction_hint={reduction_hint},
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r}
                )
                @triton.jit
            """
        else:
            tile_hint = ""
            if len(size_hints) == 2:
                if len(signature) == 4:  # input, output and 2 args
                    tile_hint = "tile_hint=TileHint.SQUARE,"
                else:
                    tile_hint = "tile_hint=TileHint.DEFAULT,"
            heuristics_line = f"""
                @{heuristics}(
                    size_hints={size_hints!r}, {tile_hint}
                    filename=__file__,
                    triton_meta={triton_meta!r},
                    inductor_meta={inductor_meta!r},
                    min_elem_per_thread={self.min_elem_per_thread}
                )
                @triton.jit
            """
        code.splice(heuristics_line)
        code.writeline(
            f"def {name or str(Placeholder.KERNEL_NAME)}({', '.join(argdefs)}):"
        )
        self.codegen_body()
        with code.indent():
            self.codegen_static_numels(code)
            for old, new in self.args.aliases():
                code.writeline(f"{old} = {new}")
            code.splice(self.body)

        if config.benchmark_kernel:
            code.splice(self.codegen_kernel_benchmark())

        return code.getvalue()


class MUSATritonScheduling(TritonScheduling):
    def __init__(self, scheduler):
        super().__init__(scheduler)

    def codegen_node_schedule(self, node_schedule, numel, reduction_numel):
        tiled_groups = self.select_tiling(node_schedule, numel, reduction_numel)
        reduction_hint_val, mutations, index_dtype = self.get_kernel_args(
            node_schedule, numel, reduction_numel
        )

        kernel = MUSATritonKernel(
            *tiled_groups,
            reduction_hint=reduction_hint_val,
            mutations=mutations,
            index_dtype=index_dtype,
        )

        self.codegen_node_schedule_with_kernel(node_schedule, kernel)

        with V.set_kernel_handler(kernel):
            src_code = kernel.codegen_kernel()

            for node in node_schedule:
                if node not in (EnableReduction, DisableReduction):
                    node.mark_run()

        kernel_name = self.define_kernel(src_code, node_schedule)
        log.debug("Generating kernel code with kernel_name: %s", kernel_name)
        self.codegen_comment(node_schedule)
        kernel.call_kernel(kernel_name)
        kernel.codegen_nan_check()
        V.graph.removed_buffers |= kernel.removed_buffers
        V.graph.inplaced_to_remove |= kernel.inplaced_to_remove

        if config.warn_mix_layout:
            kernel.warn_mix_layout(kernel_name)

        if (
            V.graph.wrapper_code.supports_intermediate_hooks
            and config.generate_intermediate_hooks
        ):
            # Not every node in the schedule will actually be live on output;
            # we can't check dead buffers.
            live_outs = kernel.args.live_output_buffers()
            for node in node_schedule:
                if not isinstance(node, scheduler.BaseSchedulerNode):
                    continue
                name = node.get_name()
                if name not in live_outs:
                    continue
                origin_node = node.node.get_origin_node()
                if origin_node is not None:
                    counters["inductor"]["intermediate_hooks"] += 1
                    V.graph.wrapper_code.writeline(
                        f"run_intermediate_hooks({origin_node.name!r}, {name})"
                    )

        self.scheduler.free_buffers()

    def codegen_sync(self):
        V.graph.wrapper_code.writeline("torch.musa.synchronize()")

from torch._dynamo.utils import dynamo_timed
from torch._inductor.scheduler import (
    NopKernelSchedulerNode,
    FusedSchedulerNode,
    SchedulerNode,
)


# seems impossible to override the codegen method of Scheduler through subclass,
# so use monkey-patch instead, which makes generation of device guard related code take effect
@dynamo_timed
def codegen(self):
    for node in self.nodes:
        try:
            log.debug(
                "Generating code for node %s with estimated runtime %f",
                node.get_name(),
                node.get_estimated_runtime(),
            )
        except Exception as e:
            log.debug(
                "Generating code for node %s with estimated runtime 0.0",
                node.get_name(),
            )

        self.enter_context(node)

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if (
                device != self.current_device
                or node.is_extern()
                or node.is_template()
            ):
                self.flush()
            if device != self.current_device:
                if device.type == "musa":
                    if self.current_device and self.current_device.type == "musa":
                        V.graph.wrapper_code.codegen_device_guard_exit()
                    assert device.index is not None, "device should have an index"
                    V.graph.wrapper_code.codegen_device_guard_enter(device.index)
                elif self.current_device and self.current_device.type == "musa":
                    V.graph.wrapper_code.codegen_device_guard_exit()
                self.current_device = device

        self.buffer_names_to_free.update(node.last_usage)

        if node.is_template():
            node, *epilogue = node.get_nodes()
            self.get_backend(device).codegen_template(node, epilogue)
        elif node.is_extern():
            self.codegen_extern_call(node)
        elif node.is_foreach():
            self.get_backend(device).codegen_foreach(node)
        elif isinstance(node, (FusedSchedulerNode, SchedulerNode)):
            self.get_backend(device).codegen_nodes(node.get_nodes())
        else:
            assert isinstance(node, NopKernelSchedulerNode)
            node.allocate()

        if config.debug_check_inf_and_nan:
            V.graph.wrapper_code.generate_inf_and_nan_checker(node)

        if config.triton.debug_sync_kernel:
            self.get_backend(device).codegen_sync()

        self.available_buffer_names.update(node.get_names())

        if not isinstance(node, NopKernelSchedulerNode):
            device = node.get_device()
            if self.get_backend(device).ready_to_flush():
                self.flush()

    self.flush()


def _apply_scheduler_patches():
    torch._inductor.scheduler.Scheduler.codegen = codegen
