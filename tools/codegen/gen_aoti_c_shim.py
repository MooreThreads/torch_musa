from __future__ import annotations

import difflib
import os
import textwrap
from dataclasses import dataclass
from typing import Dict, Sequence, Optional

from torchgen.aoti.fallback_ops import aten_shimified_ops
from torchgen.api.types import DispatcherSignature
from torchgen.model import (
    BackendIndex,
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.utils import FileManager, mapMaybe
from torchgen.context import method_with_native_function
from torchgen.gen_aoti_c_shim import (
    get_backend_index_for_aoti,
    get_fallback_op_name,
    gen_static_dispatch_backend_call_signature,
    gen_declaration_and_definition,
)

from torchgen.aoti.fallback_ops import inductor_fallback_ops
from .model import is_musa_dispatch_key


def gen_static_dispatch_backend_call(
    f: NativeFunction,
    backend_index: BackendIndex,
) -> str:
    sig = DispatcherSignature.from_schema(f.func)
    cpp_sig = gen_static_dispatch_backend_call_signature(sig, f)
    if backend_index.dispatch_key.lower() == "privateuse1":
        return f"at::musa::{cpp_sig.name()}"
    return f"at::{backend_index.dispatch_key.lower()}::{cpp_sig.name()}"


def get_header_for_aoti(
    func: NativeFunction,
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None:
        return None
    else:
        if backend_index.dispatch_key.lower() == "privateuse1":
            return f"#include <ATen/ops/{func.root_name}_musa_dispatch.h>"
        else:
            return f"#include <ATen/ops/{func.root_name}_{backend_index.dispatch_key.lower()}_dispatch.h>"


def gen_c_shim(
    func: NativeFunction,
    version_info: dict[str, list[str]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: DispatchKey,
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
) -> str | None:
    backend_index = get_backend_index_for_aoti(
        func, func_group_mapping, dispatch_key, backend_indices, extend_aoti_c_shim
    )
    if backend_index is None and dispatch_key is not None:
        return None

    schema = func.func
    device = "musa" if dispatch_key.lower() == "privateuse1" else dispatch_key.lower()
    backend_call = gen_static_dispatch_backend_call(
        func,
        backend_index,
    )

    try:
        if header:
            declaration, _ = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return declaration
        else:
            _, definition = gen_declaration_and_definition(
                schema, device, backend_call, version_info
            )
            return definition

    except NotImplementedError:
        return None


@dataclass(frozen=True)
class ShimGenerator:
    inductor_fallback_ops: dict[str, dict[str, list[str]]]
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup]
    dispatch_key: Optional[DispatchKey]
    backend_indices: dict[DispatchKey, BackendIndex]
    header: bool  # True to generate .h and False to generate .cpp
    extend_aoti_c_shim: bool

    @method_with_native_function
    def __call__(
        self,
        func: NativeFunction,
    ) -> str | None:
        version_info = self.inductor_fallback_ops[get_fallback_op_name(func)]
        return gen_c_shim(
            func,
            version_info,
            self.func_group_mapping,
            self.dispatch_key,
            self.backend_indices,
            self.header,
            self.extend_aoti_c_shim,
        )


def gen_aoti_c_shim(
    native_functions: Sequence[NativeFunction],
    inductor_fallback_ops: dict[str, dict[str, list[str]]],
    func_group_mapping: dict[OperatorName, NativeFunctionsGroup],
    dispatch_key: Optional[DispatchKey],
    backend_indices: dict[DispatchKey, BackendIndex],
    header: bool,
    extend_aoti_c_shim: bool,
    includes: str = "",
) -> str:
    body = "\n".join(
        list(
            mapMaybe(
                ShimGenerator(
                    inductor_fallback_ops,
                    func_group_mapping,
                    dispatch_key,
                    backend_indices,
                    header,
                    extend_aoti_c_shim,
                ),
                native_functions,
            )
        )
    )
    device = "musa" if dispatch_key.lower() == "privateuse1" else dispatch_key.lower()
    include_device_functions = (
        "#include <ATen/Functions.h>"
        if dispatch_key is None
        else f"#include <ATen/{str(dispatch_key)}Functions.h>"
    )
    aten_warning = (
        (
            "\n\n// This file corresponds to the aten_shimified_ops list in torchgen/aoti/fallback_ops.py\n"
        )
        if dispatch_key is None
        else ""
    )
    warning = """

// WARNING: THIS FILE IS AUTOGENERATED BY torchgen. DO NOT MODIFY BY HAND.
// See https://github.com/pytorch/pytorch/blob/7e86a7c0155295539996e0cf422883571126073e/torchgen/gen.py#L2424-L2436 for details"""

    if header:
        return (
            warning
            + aten_warning
            + textwrap.dedent(
                """

            #pragma once

            #include <torch/csrc/inductor/aoti_torch/c/shim.h>

            #ifdef __cplusplus
            extern "C" {
            #endif

            """
            )
            + body
            + textwrap.dedent(
                """

            #ifdef __cplusplus
            } // extern "C"
            #endif
            """
            )
        )
    else:
        return (
            warning
            + aten_warning
            + textwrap.dedent(
                f"""

            #include <torch/csrc/inductor/aoti_torch/generated/{"extend/" if extend_aoti_c_shim else ""}c_shim_{device}.h>
            #include <torch_musa/csrc/inductor/aoti_torch/generated/c_shim_musa.h>
            #include <torch/csrc/inductor/aoti_torch/utils.h>

            #ifndef AT_PER_OPERATOR_HEADERS
            {include_device_functions}
            #include <torch_musa/share/torch_musa_codegen/ATen/MUSAFunctions.h>
            #include <ATen/CompositeExplicitAutogradFunctions.h>
            #include <ATen/CompositeExplicitAutogradNonFunctionalFunctions.h>
            #include <ATen/CompositeImplicitAutogradFunctions.h>
            #else
            """
            )
            + includes
            + textwrap.dedent(
                """
            #endif // AT_PER_OPERATOR_HEADERS

            using namespace torch::aot_inductor;

            """
            )
            + body
        )


def gen_aoti_c_shim_files(
    aoti_fm: FileManager,
    aoti_backends: set[Optional[DispatchKey]],
    native_functions: Sequence[NativeFunction],
    backend_indices: dict[DispatchKey, BackendIndex],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    extra_musa_headers: str,
    extend_aoti_c_shim: bool,
    update_aoti_c_shim: bool,
) -> None:
    structured_func_group_dict = {}
    for func_group in structured_native_functions:
        for func in func_group.functions():
            if func.structured_delegate is not None:
                structured_func_group_dict[func.structured_delegate] = func_group
                break

    for dispatch_key in aoti_backends:
        # Use aten_shimified_ops for the aten backend, inductor_fallback_ops for others
        fallback_ops_dict = (
            aten_shimified_ops if dispatch_key is None else inductor_fallback_ops
        )
        fallbacks: Dict[str, NativeFunction] = {}
        for func in native_functions:
            op_name = get_fallback_op_name(func)
            if op_name in fallback_ops_dict:
                fallbacks[op_name] = func
        fallback_native_functions = tuple(
            value for _, value in sorted(fallbacks.items())
        )

        # Use "musa" instead of "aten" as the device name, which is different from torch
        device_name = (
            "musa" if dispatch_key.lower() == "privateuse1" else dispatch_key.lower()
        )

        # header files were checked in for ABI-compatiblilty checking
        header_file_name = f"c_shim_{device_name}.h"
        new_header = gen_aoti_c_shim(
            fallback_native_functions,
            fallback_ops_dict,
            structured_func_group_dict,
            dispatch_key,
            backend_indices,
            header=True,
            extend_aoti_c_shim=extend_aoti_c_shim,
            includes="#include <torch_musa/csrc/inductor/aoti_torch/generated/c_shim_musa.h>",
        )
        if update_aoti_c_shim:
            aoti_fm.write(
                header_file_name,
                lambda: new_header,
            )
        else:
            try:
                with open(
                    os.path.join(aoti_fm.install_dir, header_file_name)
                ) as old_file:
                    old_header = old_file.read()

                    if old_header != new_header:
                        diff = "\n".join(
                            difflib.unified_diff(
                                old_header.splitlines(),
                                new_header.splitlines(),
                                fromfile="expected",
                                tofile="actual",
                                lineterm="",
                            )
                        )

                        raise RuntimeError(
                            f"""
The generated AOTInductor C shim header files have unexpectedly changed. This
indicates an AOTInductor fallback operator ABI backward compatibility breakage!!!
Only in a limited number of situations, this is allowed:

1. You added a fallback op to the inductor_fallback_ops list in torchgen/aoti/fallback_ops.py.
If that's the case, run `python torchgen/gen.py --update-aoti-c-shim` to add a new entry to
existing C shim header files.

2. You added a new default argument to an existing fallback op. This is clearly a BC breaking
change in the AOTInductor land. You need to annotate the new default argument in
torchgen/aoti/fallback_ops.py, and then run `python torchgen/gen.py --update-aoti-c-shim` to
update the C shim header files by creating different versions of the fallback op. See
https://github.com/pytorch/pytorch/pull/154848 as an example.

{diff}
                    """
                        )
            except FileNotFoundError:
                print(
                    f"{os.path.join(aoti_fm.install_dir, header_file_name)} not found"
                )

        # cpp files are always generated on-the-fly
        def headers_for_aoti() -> str:
            headers = []
            for func in fallback_native_functions:
                header = get_header_for_aoti(
                    func,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    extend_aoti_c_shim=extend_aoti_c_shim,
                )
                if header is not None:
                    headers.append(header)
            return "\n".join(sorted(set(headers)))

        extra_headers = (
            extra_musa_headers
            if dispatch_key is not None and is_musa_dispatch_key(dispatch_key)
            else ""
        )

        aoti_fm.write(
            f"c_shim_{device_name}.cpp",
            lambda: gen_aoti_c_shim(
                fallback_native_functions,
                fallback_ops_dict,
                structured_func_group_dict,
                dispatch_key,
                backend_indices,
                header=False,
                extend_aoti_c_shim=extend_aoti_c_shim,
                includes=headers_for_aoti() + "\n" + extra_headers,
            ),
        )
