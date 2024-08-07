"""Contexts extension based on pytorch for musa codegen"""

import contextlib
from typing import Iterator, Union

from torchgen import local
from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context as utils_context

from codegen.model import musa_get_func_extra_info


@contextlib.contextmanager
def musa_native_function_manager(
    g: Union[NativeFunctionsGroup, NativeFunctionsViewGroup, NativeFunction]
) -> Iterator[None]:
    """`use_ilistref_for_tensor_lists` flag should keep the same as pytorch, not musa"""
    if isinstance(g, NativeFunctionsGroup):
        f = g.out
    elif isinstance(g, NativeFunctionsViewGroup):
        f = g.view
    else:
        f = g
    with utils_context(lambda: f"in musa_functions.yaml line {f.loc}:\n  {f.func}"):
        with local.parametrize(
            use_const_ref_for_mutable_tensors=f.use_const_ref_for_mutable_tensors,
            use_ilistref_for_tensor_lists=musa_get_func_extra_info(
                f
            ).torch_part_of_structured_group,
        ):
            yield


def init_for_musa_codegen() -> None:
    """This function should be done before executing codegen process"""
    from torchgen import context as torchgen_context  # pylint: disable=C0415

    setattr(torchgen_context, "native_function_manager", musa_native_function_manager)
