"""extend supported_ctx_manager_classes / constant-foldable functions"""

import torch  # pylint:disable=import-self
from torch._dynamo.utils import common_constant_types
from torch._dynamo.variables.torch import (
    supported_ctx_manager_classes,
    constant_fold_functions,
    constant_fold_functions_need_guards,
)
import torch_musa


_musa_supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.musa.core.amp.autocast_mode.autocast,
    ]
)

supported_ctx_manager_classes.update(_musa_supported_ctx_manager_classes)

common_constant_types.add(torch_musa._MUSAC._MusaDeviceProperties)

_musa_constant_fold_functions_need_guards = [
    torch.musa.current_device,
    torch.musa.is_initialized,
]

_musa_constant_fold_functions = [
    torch.musa.get_device_properties,
    torch.musa.is_available,
] + _musa_constant_fold_functions_need_guards

constant_fold_functions_need_guards.update(
    dict.fromkeys(_musa_constant_fold_functions_need_guards)
)
constant_fold_functions.update(dict.fromkeys(_musa_constant_fold_functions))
