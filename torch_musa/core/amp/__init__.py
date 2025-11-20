# pylint: disable=missing-module-docstring
from .autocast_mode import autocast, custom_fwd, custom_bwd  # noqa: F401
from .grad_scaler import GradScaler  # noqa: F401

from .common import _hook_autocast_common

_hook_autocast_common()
