"""
torch_musa codegen, basically borrowed from pytorch/codegen,
with some adaptions to our torch_musa's requestments
"""

# pylint: disable=C0116,C0103,C0413,W0602,W0640,C0412
import argparse
from copy import deepcopy
from os.path import abspath, dirname, join
import pathlib
import re
import os
import sys
from collections import defaultdict
from typing import (
    Dict,
    List,
    Sequence,
    Set,
    Union,
    Optional,
    Tuple,
    Any,
)

import yaml

sys.path.append(dirname(dirname(abspath(__file__))))
# Overrides of internal related modules of torchgen must be done
# before importing it firstly.
from codegen import init_for_musa_codegen

init_for_musa_codegen()

from torchgen.gen import (
    parse_tags_yaml,
    LineLoader,
    ParsedYaml,
    error_check_native_functions,
    get_grouped_native_functions,
    compute_meta_function_declaration,
    get_native_function_declarations,
)
from torchgen.model import (
    BackendIndex,
    BackendMetadata,
    DispatchKey,
    FunctionSchema,
    Location,
    NativeFunction,
    NativeFunctionsGroup,
    OperatorName,
)
from torchgen.selective_build.selector import SelectiveBuilder
from torchgen.utils import (
    concatMap,
    context,
    FileManager,
    make_file_manager,
    NamespaceHelper,
    Target,
    mapMaybe,
)
from torchgen.dest import (
    compute_native_function_declaration,
    gen_registration_helpers,
)

from codegen.gen_aoti_c_shim import (
    gen_aoti_c_shim,
    get_fallback_op_name,
    get_header_for_aoti,
)
from codegen.fallback_ops import inductor_fallback_ops

from codegen.model import (
    FunctionImplKind,
    FunctionExtraInfo,
    musa_add_func_extra_info,
    musa_get_func_extra_info,
    MUSA_STRUCTURED_DISPATCH_KEY,
    MUSA_DISPATCH_KEYS,
    is_musa_dispatch_key,
    MUSA_DEFAULT_KERNEL_NAMESPACE,
    get_musa_dispatch_namespace,
    get_musa_dispatch_file_identifier,
    related_torch_dispatch_keys,
)
from codegen.native_function_generation import (
    autogen_signature,
)
from codegen.dest import (
    musa_create_register_dispatch_key,
    gen_registration_headers,
)
from codegen.utils import (
    get_pytorch_source_directory,
    TorchOpsHeadersMigrator,
    flatten_dispatch,
)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                         HELPER FUNCTIONS
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


_GLOBAL_PARSE_MUSA_YAML_CACHE: Dict[str, ParsedYaml] = {}


def musa_override_native_function_info(
    torch_op: dict,
    musa_op: dict,
) -> Tuple[dict, FunctionExtraInfo]:
    processed_op = deepcopy(torch_op)
    impl_kinds: Dict[DispatchKey, FunctionImplKind] = {}
    torch_part_of_structured_group: bool

    def pop_one(key: str) -> Any:
        return processed_op.pop(key, None)

    def pop_many(*keys) -> None:
        for key in keys:
            pop_one(key)

    def copy_override(key: str, value: Any) -> None:
        processed_op[key] = deepcopy(value)

    def override_no_check(key: str) -> None:
        assert key in musa_op
        copy_override(key, musa_op.get(key))

    override_no_check("__line__")
    pop_one("autogen")
    torch_legacy_ufunc = pop_one("ufunc_inner_loop")
    torch_dispatch = pop_one("dispatch")
    cuda_dispatch_kernel: Optional[str] = None

    if torch_dispatch is not None:
        torch_dispatch = flatten_dispatch(torch_dispatch)

    musa_dispatch = musa_op.get("dispatch", None)
    assert musa_dispatch is None or isinstance(musa_dispatch, dict)
    old_musa_dispatch: Optional[Dict[str, str]] = None
    if musa_dispatch is not None:
        old_musa_dispatch = flatten_dispatch(musa_dispatch)

    torch_structured = processed_op.get("structured", False)
    musa_structured = musa_op.get("structured", None)
    assert musa_structured is None or isinstance(musa_structured, bool)

    if torch_legacy_ufunc is not None:
        assert (
            (torch_structured is True)
            and (old_musa_dispatch is not None)
            and (str(MUSA_STRUCTURED_DISPATCH_KEY) in old_musa_dispatch)
        )

    if torch_structured is True and torch_legacy_ufunc is None:
        assert torch_dispatch is not None
        for k, v in torch_dispatch.items():
            if DispatchKey.parse(k) == DispatchKey.CUDA:
                cuda_dispatch_kernel = v
                break
        assert isinstance(cuda_dispatch_kernel, str) and cuda_dispatch_kernel != ""

    torch_structured_delegate = processed_op.get("structured_delegate", None)
    musa_structured_delegate = musa_op.get("structured_delegate", None)
    assert musa_structured_delegate is None or isinstance(musa_structured_delegate, str)

    torch_part_of_structured_group = (
        torch_structured is True or torch_structured_delegate is not None
    )

    musa_structured_inherits = musa_op.get("structured_inherits", None)
    assert musa_structured_inherits is None or isinstance(musa_structured_inherits, str)

    def share_unstructured_native_kernel(dk: DispatchKey, musa_kernel: str) -> bool:
        if torch_dispatch is None:
            return False
        for torch_dk in related_torch_dispatch_keys(dk):
            torch_dk_str = str(torch_dk)
            if torch_dk_str not in torch_dispatch:
                continue
            torch_kernel = torch_dispatch[torch_dk_str]
            if torch_kernel == musa_kernel:
                return True
        return False

    new_musa_dispatch: Dict[str, str] = {}
    if old_musa_dispatch is not None:
        for dk_str, musa_kernel in old_musa_dispatch.items():
            dk = DispatchKey.parse(dk_str)
            assert is_musa_dispatch_key(dk)
            if dk is MUSA_STRUCTURED_DISPATCH_KEY:
                continue
            new_musa_dispatch[dk_str] = deepcopy(musa_kernel)

            if share_unstructured_native_kernel(dk, musa_kernel):
                impl_kinds[dk] = FunctionImplKind.Legacy
            else:
                impl_kinds[dk] = FunctionImplKind.Customized

        for musa_dk in MUSA_DISPATCH_KEYS:
            if musa_dk is not MUSA_STRUCTURED_DISPATCH_KEY:
                old_musa_dispatch.pop(str(musa_dk), None)

        dklen = len(old_musa_dispatch)
        assert dklen in (0, 1)
        if dklen == 0:
            old_musa_dispatch = None
        else:
            assert str(MUSA_STRUCTURED_DISPATCH_KEY) in old_musa_dispatch

    musa_precomputed = musa_op.get("precomputed", None)
    assert musa_precomputed is None or (
        torch_structured is True
        and (musa_structured is None or musa_structured is True)
        and old_musa_dispatch is not None
    )

    impl_kind: Optional[FunctionImplKind] = None
    if old_musa_dispatch is None:
        assert musa_structured_delegate is None
        assert (musa_structured is None) or (musa_structured == torch_structured)
        if (torch_structured_delegate is not None) or (torch_structured is True):
            if torch_structured is True:
                pop_one("structured_delegate")
                if (musa_structured_inherits is None) and (musa_precomputed is None):
                    impl_kind = FunctionImplKind.Legacy
                else:
                    impl_kind = FunctionImplKind.Customized
                    copy_override("structured_inherits", musa_structured_inherits)
                    copy_override("precomputed", musa_precomputed)
                new_musa_dispatch[str(MUSA_STRUCTURED_DISPATCH_KEY)] = (
                    cuda_dispatch_kernel
                )
            else:
                assert musa_structured_inherits is None and musa_precomputed is None
                pop_many("structured_inherits", "precomputed")
                impl_kind = FunctionImplKind.Legacy
        else:
            assert musa_structured_inherits is None and musa_precomputed is None
            pop_many(
                "structured_inherits",
                "precomputed",
                "structured",
                "structured_delegate",
            )
    else:
        new_musa_dispatch.update(old_musa_dispatch)
        assert musa_structured_delegate is None
        pop_one("structured_delegate")

        if torch_structured is True:
            if musa_structured is None or musa_structured is True:
                if (musa_structured_inherits is None) and (musa_precomputed is None):
                    impl_kind = FunctionImplKind.LegacyMeta
                else:
                    impl_kind = FunctionImplKind.Customized
                    copy_override("structured_inherits", musa_structured_inherits)
                    copy_override("precomputed", musa_precomputed)
            else:
                assert musa_structured_inherits is None and musa_precomputed is None
                impl_kind = FunctionImplKind.Customized
                pop_many("structured", "structured_inherits", "precomputed")
        else:
            assert (
                (musa_structured is None or musa_structured is False)
                and musa_structured_inherits is None
                and musa_precomputed is None
            )
            if share_unstructured_native_kernel(
                MUSA_STRUCTURED_DISPATCH_KEY,
                old_musa_dispatch[str(MUSA_STRUCTURED_DISPATCH_KEY)],
            ):
                impl_kind = FunctionImplKind.Legacy
            else:
                impl_kind = FunctionImplKind.Customized
            pop_many("structured", "structured_inherits", "precomputed")

    copy_override("dispatch", new_musa_dispatch)
    if impl_kind is not None:
        assert isinstance(impl_kind, FunctionImplKind)
        impl_kinds[MUSA_STRUCTURED_DISPATCH_KEY] = impl_kind
    assert musa_op.get("manual_kernel_registration", None) is None

    musa_structured = processed_op.get("structured", False)
    musa_structured_delegate = processed_op.get("structured_delegate", None)

    torch_device_check: str = processed_op.get("device_check", "ExactSame")
    torch_device_guard: bool = processed_op.get("device_guard", True)

    musa_device_check: Optional[str]
    musa_device_guard: Optional[bool]
    if torch_part_of_structured_group and (
        musa_structured is True or musa_structured_delegate is not None
    ):
        musa_device_check = musa_op.get("device_check", None)
        musa_device_guard = musa_op.get("device_guard", None)
    else:
        musa_device_check = musa_op.get("device_check", "NoCheck")
        musa_device_guard = musa_op.get("device_guard", False)

    same_device_behaviour: bool = (
        musa_device_check is None or musa_device_check == torch_device_check
    ) and (musa_device_guard is None or musa_device_guard == torch_device_guard)
    for ds_str in new_musa_dispatch:
        dk = DispatchKey.parse(ds_str)
        if dk is MUSA_STRUCTURED_DISPATCH_KEY:
            continue
        assert dk in impl_kinds
        assert (
            impl_kinds[dk] != FunctionImplKind.Legacy or same_device_behaviour is True
        ), f"{musa_op['func']} info mismatch! \
        Op kind: {impl_kinds[dk]}, \n\
        musa_device_check: {musa_device_check}, musa_device_guard: {musa_device_guard}\n\
        torch_device_check: {torch_device_check}, torch_device_guard: {torch_device_guard})"

    if musa_device_check is not None:
        pop_one("device_check")
        copy_override("device_check", musa_device_check)
    if musa_device_guard is not None:
        pop_one("device_guard")
        copy_override("device_guard", musa_device_guard)

    return processed_op, FunctionExtraInfo(
        impl_kinds=impl_kinds,
        torch_part_of_structured_group=torch_part_of_structured_group,
    )


def musa_handle_autogen_native_function_info(
    musa_op: dict,
) -> Tuple[dict, FunctionExtraInfo]:
    processed_op = deepcopy(musa_op)
    impl_kinds: Dict[DispatchKey, FunctionImplKind] = {}

    def value_of(key: str, default: Any = None) -> Any:
        return processed_op.get(key, default)

    def pop_one(key: str) -> Any:
        return processed_op.pop(key, None)

    def set_default(key: str, default: Any) -> None:
        processed_op[key] = deepcopy(default)

    assert len(value_of("cpp_no_default_args", [])) == 0
    assert value_of("variants", "function") == "function"
    assert value_of("manual_kernel_registration", False) is False
    assert value_of("manual_cpp_binding", False) is False

    assert pop_one("structured") is None
    assert pop_one("structured_delegate") is None
    assert pop_one("structured_inherits") is None
    assert pop_one("precomputed") is None

    assert value_of("python_module") is None
    assert value_of("category_override") is None

    assert pop_one("autogen") is None
    assert pop_one("ufunc_inner_loop") is None

    dispatch = value_of("dispatch", None)
    assert isinstance(dispatch, dict)
    dispatch = flatten_dispatch(dispatch)
    for dk_str in dispatch.keys():
        dk = DispatchKey.parse(dk_str)
        assert is_musa_dispatch_key(dk)
        impl_kinds[dk] = FunctionImplKind.Customized
    set_default("dispatch", dispatch)

    device_check = value_of("device_check", "NoCheck")
    set_default("device_check", device_check)

    device_guard = value_of("device_guard", False)
    set_default("device_guard", device_guard)

    return processed_op, FunctionExtraInfo(
        impl_kinds=impl_kinds,
        torch_part_of_structured_group=False,
    )


def musa_override_native_function_backend(
    backend_index: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]],
    extra_info: FunctionExtraInfo,
) -> Dict[DispatchKey, Dict[OperatorName, BackendMetadata]]:
    processed: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(
        dict
    )
    for backend, index in backend_index.items():
        assert is_musa_dispatch_key(backend)
        for op_name, metadata in index.items():
            processed[backend][op_name] = BackendMetadata(
                kernel=metadata.kernel,
                structured=metadata.structured,
                cpp_namespace=extra_info.native_function_namespace(backend),
            )
    return processed


_GLOBAL_TORCH_FUNC_NAME_RE = re.compile(r"^((?:\w+::)*[^\(]+)\(.+$")
_GLOBAL_MUSA_FUNC_NAME_RE = re.compile(r"^((?:\w+::)*[^\(]+)(.*)$")


def parse_native_yaml_struct(
    torch_ops: List[dict],
    torch_valid_tags: Set[str],
    musa_ops: List[dict],
    path: str = "<stdin>",
) -> ParsedYaml:
    assert isinstance(torch_ops, list)
    assert isinstance(musa_ops, list)
    global _GLOBAL_TORCH_FUNC_NAME_RE, _GLOBAL_MUSA_FUNC_NAME_RE

    torch_ops_map: Dict[str, dict] = {}
    torch_autogen_ops: Dict[str, Dict[str, FunctionSchema]] = defaultdict(dict)
    for torch_op in torch_ops:
        funcs = torch_op.get("func")
        decls = _GLOBAL_TORCH_FUNC_NAME_RE.findall(funcs.strip())
        func_name_with_ns = decls[0]
        torch_ops_map[func_name_with_ns] = torch_op

        autogen_str: str = torch_op.get("autogen", "")
        if autogen_str != "":
            ns_helper = NamespaceHelper.from_namespaced_entity(
                namespaced_entity=funcs,
                max_level=1,
            )
            namespace = ns_helper.get_cpp_namespace(default="aten")
            schema = FunctionSchema.parse(ns_helper.entity_name)
            for op_name in autogen_str.split(", "):
                op_schema = autogen_signature(schema, op_name)
                torch_autogen_ops[namespace][op_name] = op_schema

    rs: List[NativeFunction] = []
    bs: Dict[DispatchKey, Dict[OperatorName, BackendMetadata]] = defaultdict(dict)

    for musa_op in musa_ops:
        assert isinstance(musa_op.get("__line__"), int), musa_op
        loc = Location(path, musa_op["__line__"])

        funcs = musa_op.get("func")
        assert isinstance(funcs, str), f"not a str: {funcs}"
        decls = _GLOBAL_MUSA_FUNC_NAME_RE.findall(funcs.strip())
        func_name_with_ns, func_signature = decls[0]
        assert func_name_with_ns != "", f"empty func name at {loc}"

        torch_op = torch_ops_map.get(func_name_with_ns, None)
        processed_op: dict
        extra_info: FunctionExtraInfo

        if torch_op is not None:
            torch_funcs = torch_op.get("func")
            assert (
                func_signature == "" or funcs == torch_funcs
            ), f"musa func `{funcs}` mismatch existed torch func `{torch_funcs}`"
            processed_op, extra_info = musa_override_native_function_info(
                torch_op, musa_op
            )
        else:
            ns_helper = NamespaceHelper.from_namespaced_entity(
                namespaced_entity=func_name_with_ns,
                max_level=1,
            )
            namespace = ns_helper.get_cpp_namespace(default="aten")
            op_name = ns_helper.entity_name
            if op_name in torch_autogen_ops[namespace]:
                op_schema = torch_autogen_ops[namespace][op_name]
                complete_funcs = str(op_schema)
                namespace = ns_helper.get_cpp_namespace()
                if namespace != "":
                    complete_funcs = f"{namespace}::{complete_funcs}"
                if func_signature == "":
                    musa_op["func"] = complete_funcs
                else:
                    assert funcs == complete_funcs
                processed_op, extra_info = musa_handle_autogen_native_function_info(
                    musa_op
                )
            else:
                raise AssertionError(
                    f"musa newly added kernel(at {loc}) is not supported yet"
                )

        with context(lambda: f"in {loc}:\n  {funcs}"):
            func, m = NativeFunction.from_yaml(processed_op, loc, torch_valid_tags)
            rs.append(func)
            processed_m = musa_override_native_function_backend(m, extra_info)
            BackendIndex.grow_index(bs, processed_m)
            musa_add_func_extra_info(func, extra_info)

    # add pytorch native_funcitonal.yaml convolution ops for aoti
    keep = {
        "convolution",
    }
    for func_name in keep:
        if func_name in torch_ops_map:
            torch_op = torch_ops_map[func_name]

            func, m = NativeFunction.from_yaml(torch_op, loc, torch_valid_tags)
            rs.append(func)

            extra_info = FunctionExtraInfo(
                impl_kinds={
                    DispatchKey.CompositeExplicitAutograd: FunctionImplKind.Legacy
                },
                torch_part_of_structured_group=False,
            )

            musa_add_func_extra_info(func, extra_info)

            m_musa_only = {k: v for k, v in m.items() if is_musa_dispatch_key(k)}
            processed_m = musa_override_native_function_backend(m_musa_only, extra_info)
            BackendIndex.grow_index(bs, processed_m)

    del torch_ops_map

    error_check_native_functions(rs)
    # Default dict is to prevent the codegen from barfing when we have a dispatch key that
    # has no kernels yet.
    indices: Dict[DispatchKey, BackendIndex] = defaultdict(
        lambda: BackendIndex(
            dispatch_key=DispatchKey.Undefined,
            use_out_as_primary=True,
            external=False,
            device_guard=False,
            # I'm actually not sure about this; undefined could be hit on
            # empty TensorList, hypothetically that could have sizes in it
            index={},
        )
    )
    for k, v in bs.items():
        # All structured in-tree operators are implemented in terms of their out operator.
        indices[k] = BackendIndex(
            dispatch_key=k,
            use_out_as_primary=True,
            external=False,
            # Only cuda-like devices in tree require device guards
            device_guard=is_musa_dispatch_key(k),
            index=v,
        )
    return ParsedYaml(rs, indices)


def parse_register_func_name(impl_str: str) -> List[str]:
    match = re.findall(r'["(]([^")]+)[")]', impl_str)
    return match


def parse_native_yaml(
    native_yaml_path: str,
    tags_yaml_path: str,
    musa_yaml_path: str,
) -> ParsedYaml:
    global _GLOBAL_PARSE_MUSA_YAML_CACHE

    if musa_yaml_path not in _GLOBAL_PARSE_MUSA_YAML_CACHE:
        torch_valid_tags = parse_tags_yaml(tags_yaml_path)
        with open(native_yaml_path, "r", encoding="utf-8") as f:
            torch_ops = yaml.load(f, Loader=LineLoader)
        with open(musa_yaml_path, "r", encoding="utf-8") as f:
            musa_ops = yaml.load(f, Loader=LineLoader)
        parsed_yaml = parse_native_yaml_struct(
            torch_ops,
            torch_valid_tags,
            musa_ops,
            path=musa_yaml_path,
        )
        _GLOBAL_PARSE_MUSA_YAML_CACHE[musa_yaml_path] = parsed_yaml

    return _GLOBAL_PARSE_MUSA_YAML_CACHE[musa_yaml_path]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#
#                           RUN IT ALL
#
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def get_kernel_namespace(
    *, f: Union[NativeFunction, NativeFunctionsGroup], backend_idx: BackendIndex
) -> str:
    backend_metadata = backend_idx.get_kernel(f)
    func_name = f.func.name if isinstance(f, NativeFunction) else f.functional.func.name
    cpp_ns: str
    if not backend_metadata:
        cpp_ns = MUSA_DEFAULT_KERNEL_NAMESPACE
    else:
        cpp_ns = backend_metadata.cpp_namespace
    assert cpp_ns.endswith("::native") or cpp_ns.endswith("::musa"), (
        f"The kernel for function {func_name} "
        f"with dispatch key {backend_idx.dispatch_key}"
        f" has a namespace {cpp_ns} and it's not ending with '::native' or equal 'at::musa'."
    )
    return cpp_ns


# Return native function definitions grouped by dispatch key and custom namespace.
# Used in RegisterDispatchKey.cpp and etc.
def get_native_function_definitions(
    *,
    fm: FileManager,
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    dispatch_key: DispatchKey,
    backend_idx: BackendIndex,
    selector: SelectiveBuilder,
    symint: bool,
) -> List[str]:
    definitions: List[str] = []
    ns_definitions: Dict[str, List[str]] = defaultdict(list)
    anonymous_definitions: Dict[str, List[str]] = defaultdict(list)
    registrations: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
    newline = "\n"
    ns_gen = musa_create_register_dispatch_key(
        backend_idx,
        Target.NAMESPACED_DEFINITION,
        selector,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
    )
    anonymous_gen = musa_create_register_dispatch_key(
        backend_idx,
        Target.ANONYMOUS_DEFINITION,
        selector,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
    )
    reg_gen = musa_create_register_dispatch_key(
        backend_idx,
        Target.REGISTRATION,
        selector,
        symint=symint,
        class_method_name=None,
        skip_dispatcher_op_registration=False,
    )
    for f in grouped_native_functions:
        kernel_namespace = (
            get_kernel_namespace(f=f, backend_idx=backend_idx)
            .replace("::native", "")
            .replace("::musa", "")
        )

        ns_definitions[kernel_namespace].extend(
            ns_gen(f),
        )
        anonymous_definitions[kernel_namespace].extend(
            anonymous_gen(f),
        )
        namespace = (
            f.namespace if isinstance(f, NativeFunction) else f.functional.namespace
        )
        if namespace not in registrations[kernel_namespace]:
            registrations[kernel_namespace] = defaultdict(list)
        registrations[kernel_namespace][namespace].extend(
            reg_gen(f),
        )

    for kernel_namespace in ns_definitions:
        if len(ns_definitions[kernel_namespace]) == 0:
            continue
        ns_helper = NamespaceHelper(namespace_str=kernel_namespace)
        registration_body = ""
        for namespace in registrations[kernel_namespace]:
            if not registrations[kernel_namespace][namespace]:
                continue
            for register_func in registrations[kernel_namespace][namespace]:
                torch_func_name, impl_func_name = parse_register_func_name(
                    register_func
                )
                registration_body += f"""
ADVANCED_REGISTER({namespace}, {dispatch_key}, "{torch_func_name}", {impl_func_name})\n"""
        definitions.extend(
            fm.substitute_with_template(
                "RegisterDispatchDefinitions.ini",
                lambda: {
                    "ns_prologue": ns_helper.prologue,
                    "ns_epilogue": ns_helper.epilogue,
                    "dispatch_helpers": gen_registration_helpers(backend_idx),
                    "dispatch_anonymous_definitions": anonymous_definitions[
                        kernel_namespace
                    ],
                    "static_init_dispatch_registrations": registration_body,
                    "deferred_dispatch_registrations": "",
                    "dispatch_namespace": get_musa_dispatch_namespace(dispatch_key),
                    "dispatch_namespaced_definitions": ns_definitions[kernel_namespace],
                },
            ).split(newline)
        )

    return definitions


def gen_source_files(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    structured_native_functions: Sequence[NativeFunctionsGroup],
    backend_indices: Dict[DispatchKey, BackendIndex],
    selector: SelectiveBuilder,
    musa_fm: FileManager,
    aoti_fm: FileManager,
    dispatch_keys: List[DispatchKey],
    functions_keys: Set[DispatchKey],
) -> None:
    extra_musa_headers = """\
#include <torch_musa/csrc/aten/utils/Utils.h>
#include <torch_musa/csrc/core/MUSAGuard.h>
#include <torch_musa/csrc/aten/musa/MUSAContext.h>
#include <torch_musa/csrc/utils/register_wrapper.h>"""

    fm = musa_fm
    for dispatch_key in dispatch_keys:
        backend_index = backend_indices[dispatch_key]
        if backend_index.dispatch_key == DispatchKey.Undefined:
            continue
        dispatch_namespace = get_musa_dispatch_namespace(dispatch_key)
        dispatch_file_identifier = get_musa_dispatch_file_identifier(dispatch_key)

        def operator_headers() -> List[str]:
            headers = []
            for g in grouped_native_functions:
                is_registered = False
                if backend_index.has_kernel(g):
                    is_registered = True
                elif isinstance(g, NativeFunctionsGroup) and any(
                    backend_index.has_kernel(fn) for fn in g.functions()
                ):
                    is_registered = True
                if not is_registered:
                    continue

                headers.append(f"#include <ATen/ops/{g.root_name}_native.h>")
                if dispatch_key in functions_keys:
                    headers.append(
                        f"#include <ATen/ops/{g.root_name}_{dispatch_namespace}_dispatch.h>"
                    )

            return sorted(set(headers))

        dispatch_definitions = get_native_function_definitions(
            fm=fm,
            grouped_native_functions=grouped_native_functions,
            dispatch_key=dispatch_key,
            backend_idx=backend_index,
            selector=selector,
            symint=True,
        )
        comment = (
            "@generated by torch_musa/codegen/gen_torch_musa.py from RegisterMUSA.cpp"
        )
        fm.write_with_template(
            f"Register{dispatch_file_identifier}.cpp",
            "RegisterDispatchKey.cpp",
            lambda: {
                "generated_comment": comment,
                "extra_musa_headers": extra_musa_headers,
                "external_backend_headers": "",
                "dispatch_headers": gen_registration_headers(backend_index),
                "ops_headers": operator_headers(),
                "dispatch_helpers": "",
                "dispatch_definitions": dispatch_definitions,
            },
        )
        # ========== Generate AOT Inductor C shim for MUSA ==========
        structured_func_group_dict = {}
        for func_group in structured_native_functions:
            for func in func_group.functions():
                if func.structured_delegate is not None:
                    structured_func_group_dict[func.structured_delegate] = func_group
                    break
        if dispatch_key in (DispatchKey.PrivateUse1,):
            fallbacks = {}
            for func in native_functions:
                op_name = get_fallback_op_name(func)
                if op_name in inductor_fallback_ops:
                    fallbacks[op_name] = func
            fallback_native_functions = tuple(
                value for _, value in sorted(fallbacks.items())
            )

            header_file_name = "c_shim_musa.h"
            new_header = gen_aoti_c_shim(
                fallback_native_functions,
                structured_func_group_dict,
                dispatch_key,
                backend_indices,
                header=True,
                includes="#include <torch_musa/csrc/inductor/aoti_torch/c/shim.h>",
            )

            aoti_fm.write(
                header_file_name,
                lambda: gen_aoti_c_shim(
                    fallback_native_functions,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    header=True,
                    includes="#include <torch_musa/csrc/inductor/aoti_torch/c/shim.h>",
                ),
            )

            # cpp files are always generated on-the-fly
            def headers_for_aoti() -> str:
                headers = []
                for func in fallback_native_functions:
                    header = get_header_for_aoti(
                        func, structured_func_group_dict, dispatch_key, backend_indices
                    )
                    if header is not None:
                        headers.append(header)
                return "\n".join(sorted(set(headers)))

            extra_headers = (
                extra_musa_headers if is_musa_dispatch_key(dispatch_key) else ""
            )

            cpp_name = "c_shim_musa.cpp"
            aoti_fm.write(
                cpp_name,
                lambda: gen_aoti_c_shim(
                    fallback_native_functions,
                    structured_func_group_dict,
                    dispatch_key,
                    backend_indices,
                    header=False,
                    includes=headers_for_aoti() + "\n" + extra_headers,
                ),
            )

    del fm


def gen_per_operator_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    musa_fm: FileManager,
    ops_fm: FileManager,
    dispatch_keys: List[DispatchKey],
    functions_keys: Set[DispatchKey],
) -> None:
    functions_by_root_name: Dict[str, List[NativeFunction]] = defaultdict(list)
    for fn in native_functions:
        functions_by_root_name[fn.root_name].append(fn)

    grouped_functions_by_root_name: Dict[
        str, List[Union[NativeFunction, NativeFunctionsGroup]]
    ] = defaultdict(list)
    for group in grouped_native_functions:
        name = group.root_name
        grouped_functions_by_root_name[name].append(group)

    wrap_prefix: str = "torch_"
    ops_migrator = TorchOpsHeadersMigrator(
        wrap_prefix=wrap_prefix,
        functions_keys=functions_keys,
    )
    for name, _ in functions_by_root_name.items():
        grouped_functions = grouped_functions_by_root_name.get(name, [])

        customized_meta_functions = [
            g
            for g in grouped_functions
            if isinstance(g, NativeFunctionsGroup)
            and g.structured
            and musa_get_func_extra_info(g).gen_musa_meta_declarations() is True
        ]

        if len(customized_meta_functions) > 0:
            ops_migrator.add_musa_meta_root_name(name)
            common_meta_namespace: Optional[str] = None

            for g in customized_meta_functions:
                extra_info = musa_get_func_extra_info(g)
                meta_namespace = extra_info.meta_class_namespace()
                if common_meta_namespace is None:
                    common_meta_namespace = meta_namespace
                else:
                    assert common_meta_namespace == meta_namespace
            meta_function_declarations = list(
                mapMaybe(
                    compute_meta_function_declaration,
                    customized_meta_functions,
                )
            )
            meta_extra_headers = []
            meta_inline_headers = ops_fm.substitute_with_template(
                "MusaNativeMetaFunction.ini",
                lambda: {
                    "meta_extra_headers": meta_extra_headers,
                    "meta_namespace": common_meta_namespace,
                    "meta_function_declarations": meta_function_declarations,
                },
            )
            ops_fm.write_with_template(
                f"{name}_meta.h",
                "MusaNativeMetaFunction.h",
                lambda: {
                    "upper_root_name": name.upper(),
                    "wrap_prefix": wrap_prefix,
                    "root_name": name,
                    "inline_headers": meta_inline_headers,
                },
            )

        legacy_backend_indices: Dict[
            DispatchKey, Dict["OperatorName", BackendMetadata]
        ] = {}
        need_native_declarations: bool = False
        for dk, backend in backend_indices.items():
            legacy_backend_indices[dk] = {}

            def gen_native_declarations(
                g: Union[NativeFunction, NativeFunctionsGroup],
            ) -> bool:
                return musa_get_func_extra_info(
                    g
                ).gen_musa_native_function_declarations(dk)

            def move_legacy_function(fn: NativeFunction) -> None:
                op_name = fn.func.name
                legacy_backend_indices[dk][op_name] = backend.index.pop(op_name)

            for g in grouped_functions:
                if isinstance(g, NativeFunction):
                    if not backend.has_kernel(g):
                        continue
                    if not gen_native_declarations(g):
                        move_legacy_function(g)
                    else:
                        need_native_declarations = True
                else:
                    assert isinstance(g, NativeFunctionsGroup)
                    if g.structured:
                        if not backend.has_kernel(g):
                            continue
                        if gen_native_declarations(g):
                            need_native_declarations = True
                            continue
                    for fn in g.functions():
                        if not backend.has_kernel(fn):
                            continue
                        if not gen_native_declarations(fn):
                            move_legacy_function(fn)
                        else:
                            need_native_declarations = True

        if need_native_declarations:
            ops_migrator.add_musa_native_root_name(name)

            native_function_declarations = get_native_function_declarations(
                grouped_native_functions=grouped_functions,
                backend_indices=backend_indices,
                native_function_decl_gen=compute_native_function_declaration,
            )
            native_inline_headers = ops_fm.substitute_with_template(
                "MusaNativeFunction.ini",
                lambda: {
                    "native_function_declarations": native_function_declarations,
                },
            )
            ops_fm.write_with_template(
                f"{name}_native.h",
                "MusaNativeFunction.h",
                lambda: {
                    "upper_root_name": name.upper(),
                    "wrap_prefix": wrap_prefix,
                    "root_name": name,
                    "inline_headers": native_inline_headers,
                },
            )

        for dk in backend_indices:
            backend_indices[dk].index.update(legacy_backend_indices[dk])

    ops_migrator.migrate_op_headers()

    for dispatch_key in dispatch_keys:
        if dispatch_key not in functions_keys:
            continue

        dispatch_namespace = get_musa_dispatch_namespace(dispatch_key)
        dispatch_file_identifier = get_musa_dispatch_file_identifier(dispatch_key)
        dispatch_names = []

        for name, _ in functions_by_root_name.items():
            grouped_functions = grouped_functions_by_root_name.get(name, [])
            declarations = list(
                concatMap(
                    musa_create_register_dispatch_key(
                        backend_indices[dispatch_key],
                        Target.NAMESPACED_DECLARATION,
                        selector,
                        symint=True,
                        class_method_name=None,
                        skip_dispatcher_op_registration=False,
                    ),
                    grouped_functions,
                )
            )

            if len(declarations) == 0:
                continue

            dispatch_names.append(name)
            ops_fm.write_with_template(
                f"{name}_{dispatch_namespace}_dispatch.h",
                "DispatchKeyFunction.h",
                lambda: {
                    "upper_root_name": name.upper(),
                    "upper_dispatch_namespace": dispatch_namespace.upper(),
                    "dispatch_namespace": dispatch_namespace,
                    "dispatch_namespaced_declarations": declarations,
                },
            )

        inl_headers = f"#include <ATen/{dispatch_file_identifier}Functions_inl.h>"

        musa_fm.write_with_template(
            f"{dispatch_file_identifier}Functions.h",
            "DispatchKeyFunctions.h",
            lambda: {
                "inline_headers": inl_headers,
            },
        )
        musa_fm.write_with_template(
            f"{dispatch_file_identifier}Functions_inl.h",
            "DispatchKeyFunctions_inl.h",
            lambda: {
                "dispatch_namespace": dispatch_namespace,
                "DispatchKeyFunctions_inl_includes": [
                    f"#include <ATen/ops/{name}_{dispatch_namespace}_dispatch.h>"
                    for name in sorted(dispatch_names)
                ],
                "dispatch_namespaced_declarations": [],
            },
        )
    del musa_fm


def gen_headers(
    *,
    native_functions: Sequence[NativeFunction],
    grouped_native_functions: Sequence[Union[NativeFunction, NativeFunctionsGroup]],
    selector: SelectiveBuilder,
    backend_indices: Dict[DispatchKey, BackendIndex],
    musa_fm: FileManager,
    ops_fm: FileManager,
    dispatch_keys: List[DispatchKey],
    functions_keys: Set[DispatchKey],
) -> None:
    gen_per_operator_headers(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        selector=selector,
        backend_indices=backend_indices,
        musa_fm=musa_fm,
        ops_fm=ops_fm,
        dispatch_keys=dispatch_keys,
        functions_keys=functions_keys,
    )


def codegen() -> None:
    parser = argparse.ArgumentParser(description="Generate ATen source files")
    parser.add_argument(
        "-s",
        "--source-path",
        help="path to source directory for ATen",
        default="torch_musa/csrc/aten/",
    )
    parser.add_argument(
        "-o",
        "--output-dependencies",
        help="output a list of dependencies into the given file and exit",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run without writing any files (still updates outputs)",
    )
    parser.add_argument(
        "-d",
        "--install-dir",
        "--install_dir",
        help="output directory",
        default="build/torch_musa_codegen/ATen",
    )
    parser.add_argument(
        "--aoti-install-dir",
        help="output directory for AOT shim",
        default="torch_musa/csrc/inductor/aoti_torch/generated",
    )
    options = parser.parse_args()
    selector = SelectiveBuilder.get_nop_selector()

    # find .yaml files path
    pytorch_path = get_pytorch_source_directory()
    native_yaml_path = join(pytorch_path, "aten/src/ATen/native/native_functions.yaml")
    tags_yaml_path = join(pytorch_path, "aten/src/ATen/native/tags.yaml")
    musa_yaml_path = join(options.source_path, "ops/musa_functions.yaml")

    # load pytorch native_functions.yaml and tags.yaml
    parsed_yaml = parse_native_yaml(native_yaml_path, tags_yaml_path, musa_yaml_path)
    # valid_tags = parse_tags_yaml(tags_yaml_path)

    native_functions, backend_indices = (
        parsed_yaml.native_functions,
        parsed_yaml.backend_indices,
    )
    grouped_native_functions = get_grouped_native_functions(native_functions)

    structured_native_functions = [
        g for g in grouped_native_functions if isinstance(g, NativeFunctionsGroup)
    ]

    functions_keys: Set[DispatchKey] = {
        DispatchKey.CPU,
        MUSA_STRUCTURED_DISPATCH_KEY,
        DispatchKey.CompositeImplicitAutograd,
        DispatchKey.CompositeImplicitAutogradNestedTensor,
        DispatchKey.CompositeExplicitAutograd,
        DispatchKey.CompositeExplicitAutogradNonFunctional,
        DispatchKey.Meta,
    }

    ops_install_dir = f"{options.install_dir}/ops"
    pathlib.Path(ops_install_dir).mkdir(parents=True, exist_ok=True)
    ops_fm = make_file_manager(options=options, install_dir=ops_install_dir)

    musa_fm = make_file_manager(options=options)

    aoti_install_dir = options.aoti_install_dir
    pathlib.Path(aoti_install_dir).mkdir(parents=True, exist_ok=True)
    aoti_fm = make_file_manager(options=options, install_dir=aoti_install_dir)

    gen_headers(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        selector=selector,
        backend_indices=backend_indices,
        musa_fm=musa_fm,
        ops_fm=ops_fm,
        dispatch_keys=MUSA_DISPATCH_KEYS,
        functions_keys=functions_keys,
    )

    gen_source_files(
        native_functions=native_functions,
        grouped_native_functions=grouped_native_functions,
        structured_native_functions=structured_native_functions,
        backend_indices=backend_indices,
        selector=selector,
        musa_fm=musa_fm,
        aoti_fm=aoti_fm,
        dispatch_keys=MUSA_DISPATCH_KEYS,
        functions_keys=functions_keys,
    )


if __name__ == "__main__":
    codegen()
