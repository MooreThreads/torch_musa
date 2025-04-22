"""
Map torch_musa objects to their tracing rules (Dynamo variables),
we refer to the mapping rules of torch.cuda ideally.

See specific meanings of Dynamo variables in torch/_dynamo/trace_rules.py
"""

from torch._dynamo.variables import (
    SkipFilesVariable,
    TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)
from torch._dynamo.trace_rules import torch_name_rule_map


_manual_torch_musa_name_rule_map = {
    "torch.musa.get_rng_state": SkipFilesVariable,
    "torch.musa.set_rng_state": SkipFilesVariable,
}

_torch_musa_ctx_manager_classes = {
    k: TorchCtxManagerClassVariable
    for k in [
        "torch_musa.core.amp.autocast_base.AutocastBase",  # torch.amp.autocast_mode.autocast
        "torch_musa.core.amp.autocast_mode.autocast",
    ]
}

_torch_musa_non_c_binding_in_graph_functions = {
    k: TorchInGraphFunctionVariable
    for k in [
        "torch.musa._get_rng_state_offset",
        "torch.musa._set_rng_state_offset",
        "torch.musa.random.get_rng_state_all",
        "torch.musa.random.set_rng_state_all",
    ]
}

_torch_musa_name_rule_map = {
    **_manual_torch_musa_name_rule_map,
    **_torch_musa_ctx_manager_classes,
    **_torch_musa_non_c_binding_in_graph_functions,
}

# NOTE: torch_name_rule_map should be updated before get_torch_obj_rule_map is invoked
# TODO(mt-ai): implement dynamo logic on torch_musa APIs if needed
torch_name_rule_map.update(_torch_musa_name_rule_map)
