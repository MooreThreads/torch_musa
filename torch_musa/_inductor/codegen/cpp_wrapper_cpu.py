"""MUSA CPP wrapper impl"""

import torch

from torch._inductor import config, ir
from torch._inductor.virtualized import V
from torch._inductor.codegen.cpp_utils import (
    DTYPE_TO_CPP,
)
from torch._inductor import async_compile
from torch._inductor.codegen.cpp_wrapper_cpu import CppWrapperCpu


class CppWrapperMusa(CppWrapperCpu):
    """
    Generates cpp wrapper for running on CPU and calls cpp kernels
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"
        super().__init__()
        self.cuda = False

    # This is used to handle the bug fix
    # where there is no get_raw_stream in the Triton Python implementation.
    def write_kernel_autotune_defs_header(self) -> None:
        self.kernel_autotune_defs.splice(
            f"""
                import torch, torch_musa
                from torch._dynamo.testing import rand_strided
                from torch._dynamo.utils import preserve_rng_state
                from torch._inductor.select_algorithm import AlgorithmSelectorCache
                from {async_compile.__name__} import AsyncCompile
                from torch_musa._MUSAC import _musa_getCurrentRawStream as get_raw_stream

                async_compile = AsyncCompile()
                generate_example_value = AlgorithmSelectorCache.generate_example_value
            """
        )

    def codegen_model_kernels(self):
        self.prefix.writeline("namespace {")
        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        declare_kernel = set(self.src_to_kernel.values())
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        for kernel in sorted(declare_kernel):
            self.prefix.writeline(f"    MUfunction {kernel}{{nullptr}};")
        self.prefix.writeline("};")
        self.prefix.writeline("}  // namespace")

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
        """
        Generates kernel call code.

        musa: Defines whether the backend is GPU. Otherwise the backend is CPU.

        triton: Defines whether the GPU backend uses Triton for codegen.
                Otherwise it uses the MUSA language for codegen.
                Only valid when musa == True.
        """
        if cuda:
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

        if config.abi_compatible:
            assert arg_types is not None and len(call_args) == len(
                arg_types
            ), "Mismatch call_args and arg_types in generate_kernel_call"
            new_args = []
            for idx, arg in enumerate(call_args):
                if "*" in arg_types[idx]:
                    var_name = f"var_{next(self.arg_var_id)}"
                    self.writeline(f"auto* {var_name} = get_data_ptr_wrapper({arg});")
                    new_args.append(f"({arg_types[idx]})({var_name})")
                else:
                    # arg is a scalar
                    new_args.append(arg)
            self.writeline(self.wrap_kernel_call(kernel_name, new_args))
        else:
            self.writeline(self.wrap_kernel_call(kernel_name, call_args))
        return None

    def generate_end(self, result):
        if V.graph.aot_mode:
            if V.graph.is_const_graph:
                result.writeline("} // AOTInductorModel::_const_run_impl")
            else:
                result.writeline("} // namespace aot_inductor")
                result.writeline("} // namespace torch")
            return

        # cpp entry function for JIT with cpp wrapper
        result.writeline("'''\n)")
        result.splice(
            f"""
            inductor_entry = CppWrapperCodeCache.load_pybinding(
                ["std::vector<AtenTensorHandle>"], 
                cpp_wrapper_src,
                {self.musa}, 
                {len(V.graph.graph_outputs)})
            """
        )

        wrapper_body = "input_tensors = \
            [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]"
        if V.graph.constants:
            # Append constants to the input args for cpp wrapper.
            # Python wrapper directly gets the value inside the wrapper call
            # as a global variable passed when calling exec(code, mod.__dict__, mod.__dict__).
            # For cpp wrapper, we need to pass this python value to the inductor_entry_impl function explicitly. # pylint: disable=line-too-long
            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    input_tensors.extend(constants_tensor)
            """
        # Convert vector of at::Tensor to vector of AtenTensorHandle.
        # If we pass at::Tensor, the compilation will be too slow.
        wrapper_body += """
                    input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
        """
        # Release the inputs for memory reuse.
        wrapper_body += """
                    args.clear()
        """

        # unwrap output tensor back to python scalar
        if all(x for x in self.output_is_tensor.values()):
            # If no ShapeAsConstantBuffer in the output, directly return the output as tensors
            outputs_str = "output_tensors"
        else:
            outputs = [
                (
                    f"output_tensors[{i}]"
                    if self.output_is_tensor[i]
                    else f"output_tensors[{i}].item()"
                )
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # Wrap the func to support setting result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, buffer_if_can_stack_allocate=None
    ):
        orig_stride = stride
        device_str = self.codegen_device(device)
        dtype_code = self.codegen_dtype(dtype)
        size = self.codegen_shape_tuple(shape)
        stride = self.codegen_shape_tuple(orig_stride)
        if config.abi_compatible:
            size_array_var = self.codegen_int_array_var(
                size,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(shape),
                graph=self.get_codegened_graph(),
            )
            stride_array_var = self.codegen_int_array_var(
                stride,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(orig_stride),
                graph=self.get_codegened_graph(),
            )
            device_type, device_id = device_str.split(",")
            device_idx = "this->device_idx_" if V.graph.aot_mode else device_id
            if buffer_if_can_stack_allocate is not None:
                self.stack_allocated_buffers[name] = buffer_if_can_stack_allocate
                cpp_type = DTYPE_TO_CPP[dtype]
                numel = buffer_if_can_stack_allocate.get_numel()
                # Note: we don't zero storage because empty_strided doesn't zero either.
                self.wrapper_call.writeline(f"{cpp_type} {name}_storage[{numel}];")
                args = [
                    f"{name}_storage",
                    size_array_var,
                    stride_array_var,
                    device_type,
                    device_idx,
                ]
                return f"ArrayRefTensor<{cpp_type}> {name}({', '.join(args)});"

            args = [
                str(len(shape)),
                size_array_var,
                stride_array_var,
                dtype_code,
                device_type,
                device_idx,
                f"&{name}_handle",
            ]

            self.wrapper_call.writeline(f"AtenTensorHandle {name}_handle;")
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));"
            )

            return f"RAIIAtenTensorHandle {name}({name}_handle);"

        if V.graph.aot_mode and device_str.startswith("c10::Device("):
            tensor_device = f"{device_str.split(',')[0]}, this->device_idx_)"
        else:
            tensor_device = device_str

        if device.type == "musa":
            return (
                f"at::Tensor {name} = at::detail::empty_strided_musa("
                f"{size}, {stride}, {dtype_code}, c10::DeviceType::PrivateUse1);"
            )
        return (
            f"{self.declare}{name} = {self.namespace}empty_strided("
            f"{size}, {stride}, at::TensorOptions({tensor_device}).dtype({dtype_code})){self.ending}"  # pylint: disable=line-too-long
        )

    def codegen_reinterpret_view(
        self, data, size_list, stride_list, offset, writer, dtype=None
    ) -> str:
        dim = str(len(size_list))
        original_offset = offset
        size = self.codegen_shape_tuple(size_list)
        stride = self.codegen_shape_tuple(stride_list)
        offset = self.codegen_sizevar(offset)
        call_strs = []
        if config.abi_compatible:
            final_tmp_name = None
            is_raii_aten_tensor_handle = False

            def create_reinterpret_call():
                tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
                args = [
                    f"{data.get_name()}",
                    dim,
                    self.codegen_int_array_var(
                        size,
                        writer,
                        known_statically=self.is_statically_known_list_of_ints(
                            size_list
                        ),
                        graph=self.get_codegened_graph(),
                    ),
                    self.codegen_int_array_var(
                        stride,
                        writer,
                        known_statically=self.is_statically_known_list_of_ints(
                            stride_list
                        ),
                        graph=self.get_codegened_graph(),
                    ),
                    offset,
                ]
                call_str = (
                    f"auto {tmp_name} = reinterpret_tensor_wrapper({', '.join(args)});"
                )
                return tmp_name, call_str

            def create_dtypeview_call(reinterpret_call):
                tmp_aten_tensor_handle = (
                    f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}"
                )
                call_strs = [f"AtenTensorHandle {tmp_aten_tensor_handle};"]
                dtype_name = str(dtype).rsplit(".", maxsplit=1)[-1]
                if data.layout.device.type == "musa":
                    device_name = "musa"
                else:
                    device_name = "cpu"
                get_dtype_function = f"aoti_torch_dtype_{dtype_name}"
                dtypeview_function = f"aoti_torch_{device_name}_view_dtype"
                call_strs.append(
                    f"AOTI_TORCH_ERROR_CODE_CHECK({dtypeview_function}"
                    f"({reinterpret_call}, {get_dtype_function}(), &{tmp_aten_tensor_handle}));"
                )
                tmp_raii_aten_tensor_handle = (
                    f"tmp_{data.get_name()}_{next(self.tmp_tensor_id)}_handle"
                )
                call_strs.append(
                    f"RAIIAtenTensorHandle {tmp_raii_aten_tensor_handle}({tmp_aten_tensor_handle});"
                )
                return tmp_raii_aten_tensor_handle, call_strs

            if (
                size_list == data.layout.size
                and stride_list == data.layout.stride
                and original_offset == data.layout.offset
            ):
                # pure dtypeview
                if dtype is not None and dtype != data.dtype:
                    tmp_output_name, tmp_call_strs = create_dtypeview_call(
                        data.get_name()
                    )
                    call_strs.extend(tmp_call_strs)
                    final_tmp_name = tmp_output_name
                    is_raii_aten_tensor_handle = True
                else:
                    return f"{data.get_name()}"
            else:
                # firstly create reinterpretview
                final_tmp_name, reinterpret_call = create_reinterpret_call()
                call_strs.append(reinterpret_call)

                if dtype is not None and dtype != data.dtype:
                    # wrap it with dtypeview
                    final_tmp_name, tmp_call_strs = create_dtypeview_call(
                        reinterpret_call
                    )
                    call_strs.extend(tmp_call_strs)
            # Because the memory planning is done in two passes (see the implementation
            # of self.generate), the writeline behavior is different in the two passes.
            if writer is None:
                writer = self
            writer.writelines(call_strs)
            if (
                self.can_stack_allocate_buffer(data)
                and self.is_statically_known_list_of_ints(size_list)
                and self.is_statically_known_list_of_ints(stride_list)
                and ir.is_contiguous_strides_for_shape(stride_list, size_list)
            ):
                return final_tmp_name

            if not is_raii_aten_tensor_handle:
                return f"wrap_with_raii_handle_if_needed({final_tmp_name})"

            return final_tmp_name

        args = [data.get_name(), size, stride, offset]
        return f"reinterpret_tensor({', '.join(args)})"
