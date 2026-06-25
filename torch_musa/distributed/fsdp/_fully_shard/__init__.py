"""apply lazy_fsdp for FSDP2 patch"""

from ._fsdp_param_group import _apply_fsdp_param_group_patch
from ._fsdp_state import _apply_fsdp_state_patch
from ._fsdp_collectives import _apply_fsdp_collectives_patch
from ._fully_shard import _apply_fully_shard_class_patch

def _apply_fully_shard_patch():
    _apply_fsdp_param_group_patch()
    _apply_fsdp_state_patch()
    _apply_fsdp_collectives_patch()
    _apply_fully_shard_class_patch()
