"""extend supported_ctx_manager_classes"""

import torch  # pylint:disable=import-self
from torch._dynamo.variables.torch import supported_ctx_manager_classes


_musa_supported_ctx_manager_classes = dict.fromkeys(
    [
        torch.musa.core.amp.autocast_mode.autocast,
    ]
)

supported_ctx_manager_classes.update(_musa_supported_ctx_manager_classes)
