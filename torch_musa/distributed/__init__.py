"""apply distributed patch"""

# pylint: disable=import-outside-toplevel
from .tensor import *


# NOTE: DO NOT change the import order, otherwise it may cause unexpected runtime errors
# TODO: reconsider how to apply patches after FSDP testing done
def _apply_distributed_patch():
    from .device_mesh import _apply_device_mesh_patch

    _apply_device_mesh_patch()

    from .fsdp._init_utils import _apply_init_utils_patch

    _apply_init_utils_patch()

    from .fsdp._runtime_utils import _apply_runtime_utils_patch

    _apply_runtime_utils_patch()

    from .dtensor_patches import _apply_dtensor_patches

    _apply_dtensor_patches()

    from ._composable.fsdp.patch import _apply_fsdp2_patches

    _apply_fsdp2_patches()

    from ._state_dict_utils import _apply_state_dict_utils_patch

    _apply_state_dict_utils_patch()

    from .dist_wrapper_patch import _apply_dist_wrapper_patch

    _apply_dist_wrapper_patch()
