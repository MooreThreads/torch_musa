"""Models extension based on pytorch for musa codegen"""

# pylint: disable=C0103,W0602
from collections import defaultdict
from dataclasses import dataclass
from copy import deepcopy
from enum import auto, Enum
from typing import Dict, Union, Iterator

from torchgen.model import (
    DispatchKey,
    NativeFunction,
    NativeFunctionsGroup,
)
from torchgen.utils import assert_never

MUSA_STRUCTURED_DISPATCH_KEY = DispatchKey.PrivateUse1
MUSA_DISPATCH_KEYS = [
    MUSA_STRUCTURED_DISPATCH_KEY,
    DispatchKey.QuantizedPrivateUse1,
    DispatchKey.SparsePrivateUse1,
    DispatchKey.NestedTensorPrivateUse1,
    DispatchKey.AutogradPrivateUse1,
]
MUSA_DEFAULT_KERNEL_NAMESPACE = "at::musa"


def is_musa_dispatch_key(dk: DispatchKey) -> bool:
    return dk in MUSA_DISPATCH_KEYS


def get_musa_dispatch_namespace(dk: DispatchKey) -> str:
    assert is_musa_dispatch_key(dk)
    return str(dk).lower().replace("privateuse1", "musa")


def get_musa_dispatch_file_identifier(dk: DispatchKey) -> str:
    assert is_musa_dispatch_key(dk)
    return str(dk).replace("PrivateUse1", "MUSA")


def related_torch_dispatch_keys(dk: DispatchKey) -> Iterator[DispatchKey]:
    """
    Maybe musa dispatch key and it's related torch dispatch keys share the same native kernel.
    """
    assert is_musa_dispatch_key(dk)
    if dk is MUSA_STRUCTURED_DISPATCH_KEY:
        yield DispatchKey.CPU
        yield DispatchKey.CUDA
    elif dk is DispatchKey.QuantizedPrivateUse1:
        yield DispatchKey.QuantizedCPU
        yield DispatchKey.QuantizedCUDA
    elif dk is DispatchKey.SparsePrivateUse1:
        yield DispatchKey.SparseCPU
        yield DispatchKey.SparseCUDA
    elif dk is DispatchKey.NestedTensorPrivateUse1:
        yield DispatchKey.NestedTensorCPU
        yield DispatchKey.NestedTensorCUDA
    elif dk is DispatchKey.AutogradPrivateUse1:
        yield DispatchKey.AutogradCPU
        yield DispatchKey.AutogradCUDA
    else:
        assert_never(dk)


def init_for_musa_codegen() -> None:
    """This function should be done before executing codegen process"""
    from torchgen import model  # pylint: disable=C0415

    new_dispatch_keys = deepcopy(model.dispatch_keys)
    new_dispatch_keys.extend(MUSA_DISPATCH_KEYS)
    setattr(model, "dispatch_keys", new_dispatch_keys)

    new_structured_dispatch_keys = deepcopy(model.STRUCTURED_DISPATCH_KEYS)
    new_structured_dispatch_keys.add(MUSA_STRUCTURED_DISPATCH_KEY)
    setattr(model, "STRUCTURED_DISPATCH_KEYS", new_structured_dispatch_keys)

    setattr(model, "DEFAULT_KERNEL_NAMESPACE", MUSA_DEFAULT_KERNEL_NAMESPACE)


class FunctionImplKind(Enum):
    Legacy = auto()
    LegacyMeta = auto()
    Customized = auto()


@dataclass(frozen=True)
class FunctionExtraInfo:
    """
    Record additional features for each native function, which is useful
    for subsequent musa codegen.
    """

    impl_kinds: Dict[DispatchKey, FunctionImplKind]
    torch_part_of_structured_group: bool
    device_lazy_init: bool

    def __post_init__(self) -> None:
        assert isinstance(self.impl_kinds, dict) and len(self.impl_kinds) > 0
        assert isinstance(self.torch_part_of_structured_group, bool)
        assert isinstance(self.device_lazy_init, bool)

    def native_function_namespace(self, dk: DispatchKey) -> str:
        assert is_musa_dispatch_key(dk) and dk in self.impl_kinds
        impl_kind: FunctionImplKind = self.impl_kinds[dk]
        namespaces = ["at"]
        if impl_kind == FunctionImplKind.Legacy:
            namespaces.append("native")
        else:
            namespaces.append("musa")
        return "::".join(namespaces)

    def meta_class_namespace(self) -> str:
        assert MUSA_STRUCTURED_DISPATCH_KEY in self.impl_kinds
        if self.impl_kinds[MUSA_STRUCTURED_DISPATCH_KEY] == FunctionImplKind.Customized:
            return "musa"
        return "meta"

    def gen_musa_native_function_declarations(self, dk: DispatchKey) -> bool:
        assert is_musa_dispatch_key(dk) and dk in self.impl_kinds
        impl_kind: FunctionImplKind = self.impl_kinds[dk]
        if impl_kind in [
            FunctionImplKind.LegacyMeta,
            FunctionImplKind.Customized,
        ]:
            return True
        return False

    def gen_musa_meta_declarations(self) -> bool:
        assert MUSA_STRUCTURED_DISPATCH_KEY in self.impl_kinds
        return (
            self.impl_kinds[MUSA_STRUCTURED_DISPATCH_KEY] == FunctionImplKind.Customized
        )


_MUSA_FUNC_EXTRA_INFO_MAP = Dict[str, Dict[str, FunctionExtraInfo]]
_GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE: Dict[str, _MUSA_FUNC_EXTRA_INFO_MAP] = (
    defaultdict(lambda: defaultdict(dict))
)


def primary_function(g: Union[NativeFunction, NativeFunctionsGroup]) -> NativeFunction:
    f: NativeFunction
    if isinstance(g, NativeFunction):
        f = g
    elif isinstance(g, NativeFunctionsGroup):
        f = g.out
    else:
        assert_never(g)
    return f


def musa_add_func_extra_info(
    f: NativeFunction,
    extra_info: FunctionExtraInfo,
) -> None:
    """Record extra features of the native function"""
    global _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE

    file = f.loc.file
    namespace = f.namespace
    func_name = str(f.func.name)
    assert func_name not in _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE[file][namespace]
    _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE[file][namespace][func_name] = extra_info


def musa_get_func_extra_info(
    g: Union[NativeFunction, NativeFunctionsGroup]
) -> FunctionExtraInfo:
    """Visit extra features of the native function/function_group"""
    global _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE

    f: NativeFunction = primary_function(g)
    file = f.loc.file
    namespace = f.namespace
    func_name = str(f.func.name)
    assert func_name in _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE[file][namespace]
    return _GLOBAL_PARSE_MUSA_FUNC_EXTRA_INFO_CACHE[file][namespace][func_name]
