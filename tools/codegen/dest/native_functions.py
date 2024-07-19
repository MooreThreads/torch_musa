"""torchgen.dest.native_functions extension"""

# pylint: disable=C0415
from typing import List

from torchgen.model import (
    BackendIndex,
    NativeFunctionsGroup,
)

from codegen.model import musa_get_func_extra_info


def musa_gen_structured(
    g: NativeFunctionsGroup, backend_index: BackendIndex
) -> List[str]:
    """
    Structured kernels may be implemented by custom metaclass in musa, so the
    namespace of metaclass should be corrected by native function's `impl_kind`.
    """
    from torchgen.dest.native_functions import torch_gen_structured

    old_result = torch_gen_structured(g, backend_index)
    assert isinstance(old_result, list) and len(old_result) == 1
    extra_info = musa_get_func_extra_info(g)
    new_result = [
        text.replace(
            " : public at::meta::",
            f" : public at::{extra_info.meta_class_namespace()}::",
        )
        for text in old_result
    ]
    return new_result


def init_for_musa_codegen() -> None:
    """This function should be done before executing codegen process"""
    from torchgen.dest import native_functions

    torch_gen_structured = getattr(native_functions, "gen_structured")
    setattr(native_functions, "torch_gen_structured", torch_gen_structured)
    setattr(native_functions, "gen_structured", musa_gen_structured)
