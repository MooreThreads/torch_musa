"""Contexts extension based on pytorch for musa codegen"""

import contextlib
import functools
from typing import Callable, Iterator, TypeVar, Union

from torchgen import local
from torchgen.model import (
    NativeFunction,
    NativeFunctionsGroup,
    NativeFunctionsViewGroup,
)
from torchgen.utils import context as utils_context

from codegen.model import musa_get_func_extra_info


F = TypeVar("F")


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


def method_with_musa_native_function(
    func: Callable[[object, F], object],
) -> Callable[[object, F], object]:
    """Decorate a MUSA codegen method with metadata from musa_functions.yaml."""

    @functools.wraps(func)
    def wrapper(slf: object, f: F) -> object:
        with musa_native_function_manager(f):
            return func(slf, f)

    return wrapper


def init_for_musa_codegen() -> None:
    """Keep torchgen's native-function context manager unchanged."""
