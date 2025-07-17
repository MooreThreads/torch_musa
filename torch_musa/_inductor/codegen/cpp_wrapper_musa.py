"""MUSA CPP Wrapper Class"""

# mypy: allow-untyped-defs
import functools
import os
import sys
from itertools import chain, count
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING, Union

import sympy

from torch import dtype as torch_dtype
from torch import bfloat16, float16
from torch._inductor.runtime.triton_heuristics import grid as default_grid
from torch._inductor import config
from torch._inductor.utils import DeferredLineBase
from torch._inductor.virtualized import V
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.utils import ALIGN_BYTES
from torch._inductor.codegen.wrapper import SymbolicCallArg
from torch._inductor.codegen.cpp_utils import cexpr, DTYPE_TO_CPP

from .cpp_wrapper_cpu import CppWrapperMusa
from .codegen_device_driver import musa_kernel_driver, musa_kernel_header
from .codecache import get_cpp_wrapper_mubin_path_name
from .codecache import MusaKernelParamCache
from .wrapper import EnterMUSADeviceContextManagerLine

if TYPE_CHECKING:
    from ..graph import GraphLowering


class DeferredMusaKernelLine(DeferredLineBase):
    """
    When using cpp wrapper, MUSA kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as mubin files, so use a deferred line to backfill those information
    """

    def __init__(
        self,
        kernel_name: str,
        line_template: str,
        keys: Tuple[str, ...],
    ):
        super().__init__(line_template)
        assert not isinstance(line_template, DeferredLineBase)
        self.kernel_name = kernel_name
        self.line_template = line_template
        self.keys = keys

    def __call__(self):
        params = MusaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in MusaKernelParamCache"
        for key in self.keys:
            assert (
                key in params
            ), f"{key} not found in MusaKernelParamCache[{self.kernel_name}]"
            if key == get_cpp_wrapper_mubin_path_name():
                assert os.path.exists(params[key]), f"{params[key]} does not exist"

        return self.line_template % tuple(params[key] for key in self.keys)

    def _new_line(self, line):
        return DeferredMusaKernelLine(self.kernel_name, line, self.keys)


class DeferredMusaDefaultGrid:
    """
    A container for the default grid, which may be used by DeferredMusaGridLine
    """

    def __init__(
        self,
        kernel_name: str,
        grid,
        grid_callable: Optional[Callable[..., Any]] = None,
        **grid_extra_kwargs,
    ):
        self.kernel_name = kernel_name
        self.grid = grid
        self.grid_callable = grid_callable
        self.grid_extra_kwargs = grid_extra_kwargs

    def _process_grid(self, grid: Union[List[Any], Tuple[Any, ...]]):
        if isinstance(grid, (list, tuple)):
            return [self._process_grid(e) for e in grid]
        return grid.inner_expr if isinstance(grid, SymbolicCallArg) else grid

    def __call__(self):
        grid = self.grid
        assert isinstance(grid, (list, tuple)), f"expected {grid=} to be a list"
        grid = self._process_grid(grid)
        grid_callable = self.grid_callable or default_grid
        if not self.grid_extra_kwargs:
            grid_fn = grid_callable(*grid)
        else:
            grid_fn = grid_callable(*grid, **self.grid_extra_kwargs)

        params = MusaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in MusaKernelParamCache"
        block_cfg = {
            "XBLOCK": params["x_block"],
            "YBLOCK": params["y_block"],
            "ZBLOCK": params["z_block"],
        }
        return grid_fn(block_cfg)


class DeferredMusaGridLine(DeferredLineBase):
    """
    When using cpp wrapper, MUSA kernel load and launch needs to wait for Triton kernels
    to be tuned and stored as mubin files, so use a deferred line to backfill those information
    """

    def __init__(
        self,
        kernel_name: str,
        grid_var: str,
        grid,
        autotune_configs,
    ):
        super().__init__("")
        self.kernel_name = kernel_name
        self.grid_var = grid_var
        self.grid = grid
        self.autotune_configs = autotune_configs

    def __call__(self):
        params = MusaKernelParamCache.get(self.kernel_name)
        assert (
            params is not None
        ), f"{self.kernel_name} not found in MusaKernelParamCache"

        if self.autotune_configs is not None:
            # This indicates the Triton kernel is a user-defined one.
            grid = None
            if len(self.grid) == 1:
                grid = self.grid[0]
            else:
                for i, c in enumerate(self.autotune_configs):
                    if all(arg == params["meta"][key] for key, arg in c.kwargs.items()):
                        grid = self.grid[i]
                        break
            assert grid is not None
        elif isinstance(self.grid, DeferredMusaDefaultGrid):
            grid = self.grid()
        else:
            grid = self.grid

        assert len(grid) != 0, "Grid can't be empty"
        grid_args_str = ", ".join(
            [cexpr(V.graph.sizevars.simplify(item)) for item in grid]
        )
        return f"    Grid {self.grid_var} = Grid({grid_args_str});"

    def _new_line(self, line):
        return DeferredMusaGridLine(
            self.kernel_name, self.grid_var, self.grid, self.autotune_configs
        )


class MUSACppWrapper(CppWrapperMusa):
    """
    Generates cpp wrapper for running on GPU and calls MUSA kernels
    """

    def __init__(self) -> None:
        self.device = "musa"
        super().__init__()
        self.grid_id = count()
        self.cuda = True

    def write_header(self):
        if V.graph.is_const_graph:
            # We do not write header for constant graph, it will be written by main module.
            return

        if V.graph.aot_mode:
            for header_cpp_file in ("interface.cpp", "implementation.cpp"):
                with open(
                    os.path.join(
                        os.path.dirname(__file__), "aoti_runtime", header_cpp_file
                    ),
                    encoding="utf-8",
                ) as f:
                    self.header.splice(f.read())
        else:
            self.header.splice(
                """
                import torch
                from torch._inductor.codecache import CppWrapperCodeCache

                cpp_wrapper_src = (
                '''
                """
            )

        if config.abi_compatible:
            self.header.splice(
                f"#include <torch_musa/csrc/inductor/aoti_torch/generated/c_shim_{self.device}.h>"
            )
            self.header.splice(
                """
                #include <torch/csrc/inductor/aoti_runtime/arrayref_tensor.h>
                #include <torch/csrc/inductor/aoti_runtime/thread_local.h>
                #include <torch/csrc/inductor/aoti_runtime/scalar_to_tensor.h>
                """
            )
            if V.graph.aot_mode:
                self.header.splice(
                    """
                    #include <torch/csrc/inductor/aoti_runtime/model.h>
                    """
                )
        else:
            self.header.splice(
                """
                #include <ATen/ATen.h>
                #include <ATen/core/dispatch/Dispatcher.h>
                #include <ATen/native/BinaryOps.h>
                #include <torch/csrc/inductor/aoti_runtime/utils.h>
                #include <torch/csrc/inductor/aoti_torch/tensor_converter.h>
                #include <torch/csrc/inductor/aoti_torch/utils.h>
                #include <torch/csrc/inductor/inductor_ops.h>
                #include <torch/types.h>
                #include <ATen/ops/bernoulli_native.h>

                #define reinterpret_tensor torch::inductor::_reinterpret_tensor
                #define alloc_from_pool torch::inductor::_alloc_from_pool
                """
            )
        enable_kernel_profile = config.cpp.enable_kernel_profile and sys.platform in [
            "linux",
            "win32",
        ]
        if config.profiler_mark_wrapper_call or enable_kernel_profile:
            self.header.splice("#include <ATen/record_function.h>")

        self.header.splice("typedef at::Half half;")
        self.header.splice("typedef at::BFloat16 bfloat16;")
        self.header.splice("#include <c10/util/generic_math.h>")

        if not V.graph.aot_mode:
            self.header.splice(
                """
                #include <pybind11/pybind11.h>

                namespace py = pybind11;
                using namespace torch::aot_inductor;

                class RAIIPyObject {
                public:
                    RAIIPyObject() : obj_(nullptr) {}
                    RAIIPyObject(PyObject* obj) : obj_(obj) {}
                    ~RAIIPyObject() {
                        Py_XDECREF(obj_);
                    }
                    RAIIPyObject& operator=(const RAIIPyObject& other) {
                        if (this != &other) {
                            Py_XDECREF(obj_);
                            obj_ = other.obj_;
                            Py_XINCREF(obj_);
                        }
                        return *this;
                    }
                    operator PyObject*() {
                        return obj_;
                    }
                    PyObject* get() {
                        return obj_;
                    }
                private:
                    PyObject* obj_;
                };
                """
            )

        # Round up to the nearest multiple of ALIGN_BYTES
        # ALIGN_BYTES must be a power of 2
        self.header.splice(
            f"""
            [[maybe_unused]] static int64_t align(int64_t nbytes) {{
              return (nbytes + {ALIGN_BYTES} - 1) & -{ALIGN_BYTES};
            }}
            """
        )

        self.header.splice("#include <filesystem>")
        if config.abi_compatible:
            self.header.splice(
                """
                #include <torch_musa/csrc/inductor/aoti_runtime/utils_musa.h>
                """
            )
        else:
            self.header.splice(musa_kernel_header())
        self.header.splice(musa_kernel_driver())

    def write_get_raw_stream(self, device_idx, graph=None):
        name = f"stream{device_idx}"
        self.writeline(f"musaStream_t {name};")
        self.writeline(
            (
                f"AOTI_TORCH_ERROR_CODE_CHECK("
                f"aoti_torch_get_current_musa_stream({device_idx}, (void**)&{name})"
                f");"
            )
        )
        return name

    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=True
    ):
        if not cuda:
            return super().define_kernel(name, kernel, metadata, cuda)
        return None

    def codegen_device_guard_enter(self, device_idx: int) -> None:
        self.writeline(
            EnterMUSADeviceContextManagerLine(
                device_idx, self.last_seen_device_guard_index
            )
        )
        if config.triton.autotune_at_compile_time:
            # mimic logic of EnterDeviceContextManagerLine.codegen for the autotune code block
            self.write_triton_header_once()
            self.kernel_autotune_calls.writeline(
                f"with {V.graph.device_ops.device_guard(device_idx)}:"
            )
            self.kernel_autotune_calls.do_indent()
            self.kernel_autotune_calls.writeline(
                V.graph.device_ops.set_device(device_idx)
            )
            self.kernel_autotune_calls.writeline(
                f"stream{device_idx} = get_raw_stream({device_idx})"
            )
        self.last_seen_device_guard_index = device_idx

    def generate(self, is_inference):
        self.prefix.writeline("\n")
        if not V.graph.aot_mode:
            for kernel in chain(
                sorted(self.src_to_kernel.values()),
                sorted([entry[0] for entry in self.user_defined_kernel_cache.values()]),
            ):
                self.prefix.writeline(f"static MUfunction {kernel} = nullptr;")
            self.prefix.writeline("\n")
        return super().generate(is_inference)

    def generate_user_defined_triton_kernel(
        self,
        kernel_name: str,
        raw_args: List[Any],
        grid: List[Any],
        configs,
        triton_meta,
        constexprs,
    ):
        # in C++ wrapper, we don't pass constexpr args, as they don't
        # get added as parameters to the PTX code compiled from the
        # user-defined Triton kernel (only non-constexpr args do)
        raw_args = [
            raw_arg for i, raw_arg in enumerate(raw_args) if i not in constexprs
        ]
        args = [self.val_to_arg_str(v) for v in raw_args]
        arg_types = [
            arg.get_dtype() if hasattr(arg, "get_dtype") else type(arg)
            for arg in raw_args
        ]
        self.generate_kernel_call(
            kernel_name,
            args,
            arg_types=arg_types,
            raw_args=raw_args,
            grid=grid,
            cuda=True,
            triton=True,
            triton_meta=triton_meta,
            autotune_configs=configs,
        )

    @functools.lru_cache(None)  # pylint: disable=W1518
    def generate_load_kernel_once(
        self,
        kernel_name: str,
        graph: "GraphLowering",  # for per-graph caching
    ):
        """
        Generates or retrieves a cached load-kernel variable name for a given kernel.
        kernel_name: Name of the kernel to load.
        graph: The GraphLowering instance, used to cache per-graph loading.
        return: The C++ variable name into which the kernel will be loaded.
        """
        _ = id(graph)  # fix pylint warning of graph
        keys = (get_cpp_wrapper_mubin_path_name(), "mangled_name", "shared_mem")
        kernel_var_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
        self.writeline(f"if ({kernel_var_name} == nullptr) {{")
        self.writeline(
            DeferredMusaKernelLine(
                kernel_name,
                (
                    """    """
                    + kernel_var_name
                    + """ = loadKernel("%s", "%s", %s, this->mubin_dir_);"""
                    if V.graph.aot_mode
                    else """    """
                    + kernel_var_name
                    + """ = loadKernel("%s", "%s", %s);"""
                ),
                keys,
            )
        )
        self.writeline("}")
        return kernel_var_name

    def generate_args_decl(self, call_args, arg_types):
        """
        Generate C++ argument declarations for a kernel call, returning comma-separated pointers.
        call_args: list, kernel params list.
        arg_types: list, params type.
        return: string,  local variable.
        """
        new_args = []
        for arg, arg_type in zip(call_args, arg_types):
            var_name = f"var_{next(self.arg_var_id)}"
            if isinstance(arg_type, torch_dtype):
                if arg.endswith(".item()"):
                    # Need to declare a scalar in this case
                    ctype = DTYPE_TO_CPP[arg_type]
                    arg = arg[:-7]
                    if config.abi_compatible:
                        self.codegen_tensor_item(
                            arg_type,
                            arg,
                            var_name,
                        )
                    else:
                        if arg_type in (float16, bfloat16):
                            var_name_tmp = f"{var_name}_tmp"
                            self.writeline(
                                f"{ctype} {var_name_tmp} = {arg}.item<{ctype}>();"
                            )
                            self.writeline(f"float {var_name} = float({var_name_tmp});")
                        else:
                            self.writeline(
                                f"{ctype} {var_name} = {arg}.item<{ctype}>();"
                            )
                else:
                    if config.abi_compatible:
                        self.writeline(
                            maybe_hipify_code_wrapper(f"MUdeviceptr {var_name};")
                        )
                        self.writeline(
                            (
                                f"AOTI_TORCH_ERROR_CODE_CHECK("
                                f"aoti_torch_get_data_ptr({arg},"
                                f"reinterpret_cast<void**>(&{var_name}))"
                                f");"
                            )
                        )
                    else:
                        self.writeline(
                            maybe_hipify_code_wrapper(
                                (
                                    f"MUdeviceptr {var_name} = "
                                    f"reinterpret_cast<MUdeviceptr>({arg}.data_ptr());"
                                )
                            )
                        )
            elif arg_type in (sympy.Integer, int):
                self.writeline(f"int {var_name} = {self.expr_printer(arg)};")
            elif arg_type in (sympy.Float, float):
                self.writeline(f"float {var_name} = {self.expr_printer(arg)};")
            else:
                self.writeline(f"auto {var_name} = {self.expr_printer(arg)};")
            new_args.append(f"&{var_name}")

        return ", ".join(new_args)

    # pylint: disable=W0237
    def generate_default_grid(
        self,
        kernel_name: str,
        grid: List[Any],
        musa: bool = True,
        grid_callable: Optional[Callable[..., Any]] = None,
        **grid_extra_kwargs,
    ):
        """
        Generate grid configs for launching a CUDA kernel using the grid
        function from triton_heuristics. Because its computation needs
        to read kernel config after autotune, it is done in a deferred way
        using DeferredCudaDefaultGrid.
        """
        if not musa:
            return grid
        return DeferredMusaDefaultGrid(
            kernel_name, grid, grid_callable, **grid_extra_kwargs
        )

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        autotune_configs=None,
        grid_extra_kwargs="",
    ):
        assert arg_types is not None and len(call_args) == len(
            arg_types
        ), "call_args and arg_types do not match"

        if not cuda:
            # Even in CppWrapperCuda, we may see cpp kernels
            return super().generate_kernel_call(
                kernel_name,
                call_args,
                grid,
                device_index,
                cuda,
                triton,
                arg_types,
                raw_args,
                grid_fn,
                triton_meta,
                autotune_configs,
                grid_extra_kwargs,
            )

        device_index, call_args = self.prepare_triton_kernel_call(
            device_index, call_args
        )
        kernel_var_name = self.generate_load_kernel_once(kernel_name, V.graph)

        # args with value 1 are added into equal_to_1 and constants
        # in triton_meta (in the Python codegen) which makes them
        # inlined in the PTX and compiled CUBIN
        if (
            triton_meta is not None
            and "configs" in triton_meta
            and triton_meta["configs"]
        ):
            equal_to_1 = triton_meta["configs"][0].equal_to_1
            call_args = [arg for i, arg in enumerate(call_args) if i not in equal_to_1]
            arg_types = [t for i, t in enumerate(arg_types) if i not in equal_to_1]

        call_args_str = self.generate_args_decl(call_args, arg_types)
        kernel_args_var = f"kernel_args_var_{next(self.kernel_callsite_id)}"
        self.writeline(f"void* {kernel_args_var}[] = {{{call_args_str}}};")
        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device_index, V.graph)
        )

        grid_var = f"{kernel_name}_grid_{next(self.grid_id)}"
        self.writeline(
            DeferredMusaGridLine(kernel_name, grid_var, grid, autotune_configs)
        )

        kernel_var_name = f"kernels.{kernel_name}" if V.graph.aot_mode else kernel_name
        # add debug printer code for all triton kernel related calls
        debug_printer_manager = V.graph.wrapper_code.debug_printer
        debug_printer_manager.set_printer_args(call_args, kernel_name, arg_types, None)
        with debug_printer_manager:
            self.writeline(f"if ({grid_var}.is_non_zero()) {{")
            self.writeline(
                DeferredMusaKernelLine(
                    kernel_name,
                    r"    launchKernel({}, {}, {}, {}, %s, %s, {}, {});".format(  # pylint: disable=C0209
                        kernel_var_name,
                        f"{grid_var}.grid_x",
                        f"{grid_var}.grid_y",
                        f"{grid_var}.grid_z",
                        kernel_args_var,
                        stream,
                    ),
                    ("num_warps", "shared_mem"),
                ),
            )
            self.writeline("}")
        return None
