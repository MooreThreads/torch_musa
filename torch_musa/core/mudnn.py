"""muDNN related functions and attributes"""

import sys
from contextlib import contextmanager

from torch.backends import __allow_nonbracketed_mutation, ContextProp, PropModule
import torch_musa


def version():
    """Get the mudnn version."""
    return torch_musa._MUSAC._mudnn_version()


def set_flags(_allow_tf32: bool):
    orig_flags = (torch_musa._MUSAC._get_allow_tf32(),)
    torch_musa._MUSAC._set_allow_tf32(_allow_tf32)
    return orig_flags


@contextmanager
def flags(allow_tf32=True):
    """Some useful flags to setup. We only have allow_tf32 for now."""
    with __allow_nonbracketed_mutation():
        orig_flags = set_flags(allow_tf32)
    try:
        yield
    finally:
        # Recover the previous values
        with __allow_nonbracketed_mutation():
            set_flags(*orig_flags)


class MudnnModule(PropModule):
    """A helper module that enables the hack for the frontend to get mudnn attributes."""
    __all__ = list(set(vars().keys()) - {"__module__", "__qualname__"})

    allow_tf32 = ContextProp(
        torch_musa._MUSAC._get_allow_tf32, torch_musa._MUSAC._set_allow_tf32
    )


# This plays some tricks to inject the funcitons to `torch.backends` so that it keeps consistent to
# the cuda APIs.
sys.modules[__name__] = MudnnModule(sys.modules[__name__], __name__)
