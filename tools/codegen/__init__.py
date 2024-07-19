"""
torch_musa codegen
This module contains codegeneration utilities for torch_musa.
"""

# pylint: disable=wrong-import-position

from .model import (
    init_for_musa_codegen as model_init,
)

from .context import (
    init_for_musa_codegen as context_init,
)

from .dest.native_functions import (
    init_for_musa_codegen as dest_native_functions_init,
)

from .dest.register_dispatch_key import (
    init_for_musa_codegen as dest_register_dispatch_key_init,
)


def init_for_musa_codegen() -> None:
    model_init()
    context_init()
    dest_native_functions_init()
    dest_register_dispatch_key_init()
