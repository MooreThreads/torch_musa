"""
Map torch_musa objects to their tracing rules (Dynamo variables),
we refer to the mapping rules of torch.cuda ideally.

See specific meanings of Dynamo variables in torch/_dynamo/trace_rules.py
"""

__all__ = ["_apply_dynamo_trace_rules_patches"]

import importlib
import torch
from torch._dynamo.variables import (
    # SkipFilesVariable,
    # TorchCtxManagerClassVariable,
    TorchInGraphFunctionVariable,
)
from torch._dynamo.trace_rules import torch_name_rule_map


_torch_musa_non_c_binding_in_graph_functions = dict.fromkeys(
    [
        "torch.musa.current_stream",
        "torch.musa.default_stream",
        "torch.musa.stream",
    ],
    TorchInGraphFunctionVariable,
)

# NOTE: before updating the torch_name_rule_map, clear the lru_cache
torch._dynamo.trace_rules.get_torch_obj_rule_map.cache_clear()
torch_name_rule_map[2].update(_torch_musa_non_c_binding_in_graph_functions)


def _load_obj_from_str(fully_qualified_name):
    module, obj_name = fully_qualified_name.rsplit(".", maxsplit=1)
    if module == "torch.musa":
        module = "torch_musa"
    return getattr(importlib.import_module(module), obj_name)


def _apply_dynamo_trace_rules_patches():
    torch._dynamo.trace_rules._load_obj_from_str = _load_obj_from_str
