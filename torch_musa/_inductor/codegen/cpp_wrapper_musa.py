# pylint: disable=W0613,W0221,E1120,W0237,E1121,E1123

"""MUSA CPP Wrapper Class"""

# mypy: allow-untyped-defs
import dataclasses
import re
from itertools import count, zip_longest
from typing import Any, Optional, Union

import sympy
from typing_extensions import Self

from torch import dtype as torch_dtype
from torch._inductor import config
from torch._inductor.codegen.aoti_hipify_utils import maybe_hipify_code_wrapper
from torch._inductor.codegen.common import device_op_overrides_dict
from torch._inductor.codegen.cpp_utils import cexpr
from torch._inductor.codegen.multi_kernel import MultiKernelCall
from torch._inductor.codegen.triton_utils import should_unwrap_unspec_arg
from torch._inductor.codegen.wrapper import SymbolicCallArg
from torch._inductor.ir import TensorBox
from torch._inductor.runtime.runtime_utils import dynamo_timed
from torch._inductor.utils import GPU_ALIGN_BYTES, IndentedBuffer, cache_on_self
from torch._inductor.virtualized import V
from torch._inductor.runtime.triton_heuristics import GridExpr

from .codecache import MusaKernelParamCache, get_cpp_wrapper_mubin_path_name
from .cpp_wrapper_cpu import CppWrapperMusa
from .wrapper import MUSATritonWrapperCodeGen


@dataclasses.dataclass
class UnwrapUnspecArg:
    """Marker that we need to call .item() on the tensor."""

    dtype: torch_dtype


_CPP_STRING_LITERAL_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\n": "\\n",
    "\t": "\\t",
    "\r": "\\r",
}
_CPP_STRING_LITERAL_PATTERN = re.compile(r'["\\\n\t\r]')


def cpp_string_literal(s: str) -> str:
    """Escape a Python string into a valid C++ string literal."""
    escaped = _CPP_STRING_LITERAL_PATTERN.sub(
        lambda m: _CPP_STRING_LITERAL_ESCAPES[m.group(0)], s
    )
    return f'"{escaped}"'


@dataclasses.dataclass
class DeferredTritonCallWrapper:
    """
    Defers generating a C++ wrapper for a Triton kernel until autotune finishes
    and the final choice + params are known.
    """

    wrapper_name: str
    kernel_name: str
    arg_types: list[Any]

    def generate(self, wrapper) -> None:
        """Emit the C++ wrapper function and ensure mubin is packaged."""
        prefix = wrapper.prefix
        if self.kernel_name.startswith("multi_kernel_"):
            # MultiKernel picks final kernel after autotune warmup.
            self.kernel_name = MultiKernelCall.lookup_choice(self.kernel_name)

        params = MusaKernelParamCache.get(self.kernel_name)
        assert params, f"MusaKernelParamCache not populated for {self.kernel_name}"
        def_args = params["def_args"]
        arg_types = self.arg_types
        inductor_meta = params["inductor_meta"]

        if "extra_launcher_args" in inductor_meta and len(def_args) > len(arg_types):
            # extra_launcher_args are in def_args already; extend arg_types accordingly.
            assert len(def_args) == len(arg_types) - len(
                inductor_meta["extra_launcher_args"]
            )
            arg_types = arg_types + [SymbolicCallArg] * len(
                inductor_meta["extra_launcher_args"]
            )

        if not V.graph.aot_mode:
            prefix.writeline(
                maybe_hipify_code_wrapper(
                    f"static {wrapper.device_codegen.cpp_kernel_type()} "
                    f"{self.kernel_name} = nullptr;"
                )
            )
            kernel_var_name = self.kernel_name
        else:
            kernel_var_name = f"kernels_.{self.kernel_name}"

        # template types for tensor-handle-like args
        template_types = [
            f"typename {name}_type_"
            for name, a_type in zip(def_args, arg_types)
            if isinstance(a_type, (torch_dtype, UnwrapUnspecArg))
        ]
        if V.graph.aot_mode:
            template_types.append("typename kernels_type_")
        if template_types:
            prefix.writeline(f"template <{', '.join(template_types)}>")

        prefix.writeline(f"static inline void {self.wrapper_name}(")
        with prefix.indent():
            assert len(def_args) == len(arg_types), (def_args, arg_types)
            for name, a_type in zip(def_args, arg_types):
                if isinstance(a_type, (torch_dtype, UnwrapUnspecArg)):
                    prefix.writeline(f"const {name}_type_& {name},")
                elif issubclass(a_type, (SymbolicCallArg, sympy.Expr, int)):
                    prefix.writeline(f"int64_t {name},")
                elif a_type is float:
                    prefix.writeline(f"float {name},")
                elif a_type is bool:
                    prefix.writeline(f"bool {name},")
                else:
                    raise ValueError(f"Unexpected arg type {a_type}")
            prefix.writeline(f"{wrapper.device_codegen.cpp_stream_type()} stream_,")
            if V.graph.aot_mode:
                prefix.writeline("kernels_type_& kernels_,")
            prefix.writeline(
                "const std::optional<std::string>& mubin_dir_ = std::nullopt"
            )
        prefix.writeline("){")
        with prefix.indent():
            self.generate_grid(prefix, inductor_meta, params)
            self.generate_load_kernel(prefix, kernel_var_name, params)
            self.generate_launch_kernel(prefix, wrapper, kernel_var_name, params)
        prefix.writeline("}")

        # Ensure the mubin file is included in the package.
        V.graph.wrapper_code.additional_files.append(
            params[get_cpp_wrapper_mubin_path_name()]
        )

    def generate_grid(
        self,
        prefix: IndentedBuffer,
        inductor_meta: dict[str, Any],
        params: dict[str, Any],
    ) -> None:
        """Emit grid calculation code (x, y, z) for Triton launch."""
        grid = GridExpr.from_meta(inductor_meta, params["config"], mode="cpp")
        for line in grid.prefix:
            prefix.writeline(line)
        prefix.splice(
            f"""
            uint32_t grid_0 = {grid.x_grid};
            uint32_t grid_1 = {grid.y_grid};
            uint32_t grid_2 = {grid.z_grid};
            """
        )
        prefix.writeline("if (grid_0 == 0 || grid_1 == 0 || grid_2 == 0) return;")

    def generate_load_kernel(
        self, prefix: IndentedBuffer, kernel_var_name: str, params: dict[str, Any]
    ) -> None:
        """Emit code to load mubin + resolve function symbol if needed."""
        prefix.writeline(f"if ({kernel_var_name} == nullptr) {{")
        with prefix.indent():
            load_kernel_args = [
                cpp_string_literal(params[get_cpp_wrapper_mubin_path_name()]),
                cpp_string_literal(params["mangled_name"]),
                str(params["shared_mem"]),
                "mubin_dir_",
            ]
            prefix.writeline(
                f"{kernel_var_name} = loadKernel({', '.join(load_kernel_args)});"
            )
        prefix.writeline("}")

    def generate_launch_kernel(
        self,
        prefix: IndentedBuffer,
        wrapper,
        kernel_var_name: str,
        params: dict[str, Any],
    ) -> None:
        """Emit the final launch invocation with proper args and stream."""
        triton_meta = params["triton_meta"]
        assert len(self.arg_types) == len(params["def_args"]), (
            self.arg_types,
            params["def_args"],
        )
        arg_type_lookup = dict(zip(params["def_args"], self.arg_types))
        # C++ wrapper strips constants equal_to_1
        call_args = [
            n for n in params["call_args"] if n not in triton_meta["constants"]
        ]
        arg_types = [arg_type_lookup[n] for n in call_args]
        arg_signatures = [triton_meta["signature"][n] for n in call_args]
        call_args_str = wrapper.generate_args_decl(
            prefix, call_args, arg_types, arg_signatures
        )
        prefix.writeline(f"void* kernel_args_[] = {{{call_args_str}}};")
        launch_kernel_args = [
            kernel_var_name,
            "grid_0",
            "grid_1",
            "grid_2",
            str(params["num_warps"]),
            str(params["shared_mem"]),
            "kernel_args_",
            "stream_",
        ]
        prefix.writeline(f"launchKernel({', '.join(launch_kernel_args)});")


class MUSACppWrapper(CppWrapperMusa):
    """
    Generates C++ wrapper for running on GPU (MUSA) and launching MUSA kernels.
    """

    def __init__(self) -> None:
        self.device = "musa"
        self.device_codegen = device_op_overrides_dict[self.device]
        super().__init__()
        self.grid_id = count()
        self._triton_call_wrappers: dict[str, DeferredTritonCallWrapper] = {}

    @staticmethod
    def create(
        is_subgraph: bool,
        subgraph_name: Optional[str],
        parent_wrapper=None,
        partition_signatures=None,
    ):
        """Factory to create the wrapper (subgraphs not yet supported)."""
        # TODO: support subgraph codegen by lifting functions (see CppWrapperCpu.codegen_subgraph)
        return MUSACppWrapper()

    def codegen_invoke_subgraph(self, *args, **kwargs):  # noqa: D401
        """Override abstract method from CppWrapperMusa; not supported yet."""
        raise NotImplementedError(
            "MUSACppWrapper.codegen_invoke_subgraph not supported"
        )

    def write_header(self) -> None:
        """Emit common header plus MUSA driver helpers."""
        if V.graph.is_const_graph:
            # constant graph header is emitted by the main module
            return
        super().write_header()
        self.header.splice(
            maybe_hipify_code_wrapper(self.device_codegen.kernel_driver())
        )

    @cache_on_self
    def write_tma_desc_helpers(self) -> None:
        """Emit TMA descriptor helper functions (once)."""
        self.header.splice(self.device_codegen.tma_descriptor_helpers())

    def write_get_raw_stream(self, device_idx: int, graph=None) -> str:
        """Fetch raw stream pointer for device index."""
        name = f"stream{device_idx}"
        self.writeline(
            maybe_hipify_code_wrapper(
                f"{self.device_codegen.cpp_stream_type()} {name};"
            )
        )
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK("
            f"{self.device_codegen.aoti_get_stream()}({device_idx}, (void**)&{name}));"
        )
        return name

    def codegen_inputs(self) -> None:
        """
        Copy misaligned inputs to aligned buffers in AOT mode to match JIT Inductor
        behavior. See: Note [Input Alignment handling in Inductor].
        """
        if V.graph.aot_mode and V.graph.inputs_to_check:
            for idx in V.graph.inputs_to_check:
                input_name = V.graph.graph_input_names[idx]
                assert input_name in V.graph.graph_inputs, f"{input_name} not found"
                value = V.graph.graph_inputs[input_name]
                assert isinstance(value, TensorBox), f"{input_name} must be a Tensor"
                warn_msg = (
                    f"Input {idx} was compiled as {GPU_ALIGN_BYTES}-bytes aligned, "
                    "but it is not aligned at run time. Copying to an aligned tensor "
                    "to guarantee correctness, but expect a performance hit."
                )
                self.prefix.splice(
                    f"""
                    if ((long({input_name}.data_ptr()) & ({GPU_ALIGN_BYTES} - 1)) != 0) {{
                        AOTI_TORCH_WARN("{warn_msg}");
                        AtenTensorHandle {input_name}_aligned;
                        aoti_torch_clone_preserve_strides({input_name}, &{input_name}_aligned);
                        {input_name} = std::move(RAIIAtenTensorHandle({input_name}_aligned));
                    }}
                    """
                )
        super().codegen_inputs()

    def define_kernel(
        self,
        kernel_name: str,
        kernel_body: str,
        metadata: Optional[str] = None,
        gpu: bool = True,
        cpp_definition: Optional[str] = None,
    ):
        """Define a kernel; for Triton+GPU, let Python wrapper build autotune block."""
        if gpu:
            if config.triton.autotune_at_compile_time:
                return MUSATritonWrapperCodeGen.define_kernel(
                    self, kernel_name, kernel_body, metadata, gpu, cpp_definition
                )
            return None
        # CPU kernels still go through parent implementation.
        return CppWrapperMusa.define_kernel(
            self, kernel_name, kernel_body, metadata, gpu, cpp_definition
        )

    def generate(self, is_inference):
        """Generate the full C++ wrapper."""
        with dynamo_timed("CppWrapperGpu.generate", log_pt2_compile_event=True):
            return super().generate(is_inference)

    def finalize_prefix(self) -> None:
        """Emit deferred Triton wrappers at the beginning of prefix."""
        old_prefix = self.prefix  # new content goes before existing prefix
        self.prefix = IndentedBuffer()
        super().finalize_prefix()
        for kernel in self._triton_call_wrappers.values():
            self.prefix.writeline("")
            kernel.generate(self)
        self.prefix.writeline("")
        self.prefix.splice(old_prefix)

    def generate_tma_descriptor(self, desc) -> None:
        """Generate a host-side TMA descriptor and its data pointer wiring."""
        self.write_tma_desc_helpers()

        # generate data pointer for the source tensor
        source = self.generate_args_decl(
            code=self,
            call_args=[self.val_to_arg_str(desc.tensor)],
            arg_types=[desc.tensor.get_dtype()],
            arg_signatures=[None],
            # passed to init{1,2}DTMADescriptor, which is NOT a Triton kernel
            is_triton_kernel=False,
        )

        desc_name = desc.name
        self.writeline("alignas(64) MUtensorMap " + desc_name + ";")

        # `source` is in the form of `&var_x`, where `var_x` is CUdeviceptr;
        # dereference and cast to void* for helper function argument.
        ptr = f"reinterpret_cast<void*>(*({source}))"
        dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.dims)
        block_dims = ", ".join(self.val_to_arg_str(dim) for dim in desc.block_dims)
        element_size = self.val_to_arg_str(desc.element_size)
        init_fn = f"init{desc.rank}DTMEDescriptor"
        args = f"&{desc_name}, {ptr}, {dims}, {block_dims}, {element_size}"
        self.writeline(f"{init_fn}({args});")

    def generate_args_decl(
        self,
        code: Union[IndentedBuffer, Self],
        call_args,
        arg_types,
        arg_signatures,
        is_triton_kernel: bool = True,
    ) -> str:
        """
        Declare local variables for kernel args (e.g., auto var_0 = ...;) and
        return a comma-separated string of addresses to pass into the kernel.
        """
        new_args: list[str] = []

        signature2dtype = {
            "i32": "int32_t",
            "i64": "int64_t",
            "fp32": "float",
        }

        def process_args(arg, arg_type, arg_signature=None) -> None:
            var_name = f"var_{next(self.arg_var_id)}"
            # Ignore nvTmaDesc: host-side TMA descriptors are passed by value.
            if isinstance(arg_type, UnwrapUnspecArg) and arg_signature != "mtTmeDesc":
                self.codegen_tensor_item(
                    arg_type.dtype, arg, var_name, indented_buffer=code
                )
            elif isinstance(arg_type, torch_dtype) and arg_signature != "mtTmeDesc":
                device_ptr_type = self.device_codegen.cpp_device_ptr()
                code.writeline(
                    maybe_hipify_code_wrapper(
                        f"{device_ptr_type} {var_name} = "
                        f"reinterpret_cast<{device_ptr_type}>({arg}.data_ptr());"
                    )
                )
            elif arg_type in (sympy.Integer, int):
                code.writeline(f"int {var_name} = {cexpr(arg)};")
            elif arg_type in (sympy.Float, float):
                code.writeline(f"float {var_name} = {cexpr(arg)};")
            # Cast SymbolicCallArg explicitly based on Triton signature.
            elif (
                isinstance(arg_type, type(SymbolicCallArg))
                and arg_signature is not None
                and arg_signature in signature2dtype
            ):
                code.writeline(
                    f"{signature2dtype[arg_signature]} {var_name} = {cexpr(arg)};"
                )
            else:
                code.writeline(f"auto {var_name} = {cexpr(arg)};")
            new_args.append(f"&{var_name}")

        for arg, arg_type, arg_signature in zip_longest(
            call_args, arg_types, arg_signatures
        ):
            process_args(arg, arg_type, arg_signature)

        if is_triton_kernel:
            global_scratch = self.device_codegen.cpp_global_scratch(
                next(self.arg_var_id)
            )
            if global_scratch is not None:
                global_scratch_def, global_scratch_var = global_scratch
                code.writeline(global_scratch_def)
                new_args.append(f"&{global_scratch_var}")

        return ", ".join(new_args)

    def generate_kernel_call(
        self,
        kernel_name: str,
        call_args,
        *,
        device=None,
        triton: bool = True,
        arg_types=None,
        raw_args=None,
        triton_meta=None,
    ) -> None:
        """
        Generate a kernel call (GPU default). CPU kernels still supported.
        """
        device = device or V.graph.get_current_device_or_throw()
        if device.type == "cpu":
            # Even here we may encounter C++ CPU kernels.
            CppWrapperMusa.generate_kernel_call(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_args=raw_args,
                triton_meta=triton_meta,
            )
            return

        if (
            triton
            and config.triton.autotune_at_compile_time
            and kernel_name not in self.kernel_autotune_names
        ):
            MUSATritonWrapperCodeGen.generate_kernel_call(
                self,
                kernel_name,
                call_args,
                device=device,
                triton=triton,
                arg_types=arg_types,
                raw_args=raw_args,
                triton_meta=triton_meta,
            )

        stream = (
            "stream"
            if V.graph.aot_mode
            else self.write_get_raw_stream(device.index, V.graph)
        )

        if triton:
            call_args, arg_types = self.prepare_triton_wrapper_args(
                call_args, arg_types
            )
            wrapper_name = f"call_{kernel_name}"
            if wrapper_name not in self._triton_call_wrappers:
                self._triton_call_wrappers[wrapper_name] = DeferredTritonCallWrapper(
                    wrapper_name, kernel_name, arg_types
                )
            call_args.append(stream)
            if V.graph.aot_mode:
                call_args.append("kernels")
                call_args.append("this->cubin_dir_")

            debug_printer_manager = V.graph.wrapper_code.debug_printer
            debug_printer_manager.set_printer_args(
                call_args[: len(arg_types)], kernel_name, arg_types, None
            )
            with debug_printer_manager:
                self.writeline(f"{wrapper_name}({', '.join(call_args)});")
        else:
            casted = []
            for arg_type, arg in zip(arg_types, call_args):
                new_arg = arg
                if arg_type.endswith("*") and arg != "nullptr":
                    new_arg = f"{arg}.data_ptr()"
                casted.append(f"({arg_type}){cexpr(new_arg)}")
            call_args_str = ", ".join(casted)
            self.writeline(f"kernels.{kernel_name}({call_args_str}, {stream});")

    @staticmethod
    def prepare_triton_wrapper_args(
        call_args: list[Any], arg_types: list[Any]
    ) -> tuple[list[Any], list[Any]]:
        """Normalize arg strings and fix dtypes for unspec scalars."""
        assert len(call_args) == len(arg_types), (call_args, arg_types)
        new_args: list[str] = []
        new_arg_types: list[Any] = []

        for arg, a_type in zip(call_args, arg_types):
            if isinstance(arg, str):
                if isinstance(a_type, torch_dtype) and should_unwrap_unspec_arg(arg):
                    # dynamo wraps unspec as 0d CPU tensor; convert to scalar
                    a_type = UnwrapUnspecArg(dtype=a_type)
                new_args.append(arg)
            elif isinstance(arg, bool):
                new_args.append(str(arg).lower())
            elif isinstance(arg, (int, float, SymbolicCallArg)):
                new_args.append(str(arg))
            else:
                new_args.append(cexpr(V.graph.sizevars.simplify(arg)))
            new_arg_types.append(a_type)

        return new_args, new_arg_types

    def make_zero_buffer(self, name: str) -> str:
        """Emit C++ to zero out an AtenTensorHandle."""
        return f"AOTI_TORCH_ERROR_CODE_CHECK(" f"aoti_torch_zero_({name}.get()));"
