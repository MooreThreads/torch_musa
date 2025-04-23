# pylint: disable=all
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)
import torch
import torch.nn as nn
from torch.distributed.fsdp._dynamo_utils import _annotate_modules_for_dynamo
from torch.distributed.fsdp._wrap_utils import _auto_wrap
from torch.distributed.fsdp._init_utils import (
    _check_orig_params_flattened,
    _init_buffer_state,
    _init_core_state,
    _init_device_handle,
    _init_extension,
    _init_ignored_module_states,
    _init_param_handle_from_module,
    _init_prefetching_state,
    _init_process_group_state,
    _init_runtime_state,
    _init_state_dict_state,
    HYBRID_SHARDING_STRATEGIES,
    ProcessGroupType,
)

from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)

from torch.distributed.fsdp.wrap import CustomPolicy, ModuleWrapPolicy
from torch.distributed._tensor import DeviceMesh
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel
from torch.distributed.fsdp._state_dict_utils import _register_all_state_dict_hooks
from torch.distributed.fsdp._unshard_param_utils import _register_flat_param


# remove this patch When PyTorch version >= 2.3
def __init__(
        self,
        module: nn.Module,
        process_group: ProcessGroupType = None,
        sharding_strategy: Optional[ShardingStrategy] = None,
        cpu_offload: Optional[CPUOffload] = None,
        auto_wrap_policy: Optional[
            Union[Callable, ModuleWrapPolicy, CustomPolicy]
        ] = None,
        backward_prefetch: Optional[BackwardPrefetch] = BackwardPrefetch.BACKWARD_PRE,
        mixed_precision: Optional[MixedPrecision] = None,
        ignored_modules: Optional[Iterable[torch.nn.Module]] = None,
        param_init_fn: Optional[Callable[[nn.Module], None]] = None,
        device_id: Optional[Union[int, torch.device]] = None,
        sync_module_states: bool = False,
        forward_prefetch: bool = False,
        limit_all_gathers: bool = True,
        use_orig_params: bool = False,
        ignored_states: Union[
            Optional[Iterable[torch.nn.Parameter]], Optional[Iterable[torch.nn.Module]]
        ] = None,
        device_mesh: Optional[DeviceMesh] = None,
    ):
        torch._C._log_api_usage_once("torch.distributed.fsdp")
        super(FullyShardedDataParallel, self).__init__()
        _init_ignored_module_states(self, module, ignored_modules, ignored_states)
        _init_device_handle(self, module, self._ignored_params, device_id)

        # Add module annotations for Dynamo support (see function for details)
        _annotate_modules_for_dynamo(module, self._ignored_modules, use_orig_params)

        # Initializes self.process_group, along with rank and world size. This will
        # also set another attribute, _inter_node_pg, to control the process group
        # over which sharding occurs, if sharding_strategy is {HYBRID_SHARD, _HYBRID_SHARD_ZERO2}.
        # Note that this is done before auto_wrapping, so that child FSDP modules simply pick up
        # the same process group state as the root FSDP module.
        self._device_mesh = device_mesh
        _init_process_group_state(
            self,
            process_group,
            sharding_strategy,
            auto_wrap_policy,
            device_mesh,
        )
        if auto_wrap_policy is not None:
            root_kwargs = {
                "process_group": process_group,
                "sharding_strategy": sharding_strategy,
                "cpu_offload": cpu_offload,
                "backward_prefetch": backward_prefetch,
                "mixed_precision": mixed_precision,
                "param_init_fn": param_init_fn,
                "device_id": device_id,
                "sync_module_states": sync_module_states,
                "forward_prefetch": forward_prefetch,
                "limit_all_gathers": limit_all_gathers,
                "use_orig_params": use_orig_params,
                "ignored_states": self._ignored_params,
                "device_mesh": device_mesh,
            }
            if sharding_strategy in HYBRID_SHARDING_STRATEGIES and device_mesh is None:
                # Share root process groups with children to maintain
                # the invariant that all FSDP modules will have the same
                # process groups.
                root_kwargs["process_group"] = (self.process_group, self._inter_node_pg)

            _auto_wrap(
                module,
                auto_wrap_policy,
                self._ignored_modules,
                self._ignored_params,
                root_kwargs,
                FullyShardedDataParallel,
            )

        backward_prefetch_limit = 1
        forward_prefetch_limit = 1
        _init_core_state(
            self,
            sharding_strategy,
            mixed_precision,
            cpu_offload,
            limit_all_gathers,
            use_orig_params,
            backward_prefetch_limit,
            forward_prefetch_limit,
        )
        _init_runtime_state(self)
        _init_prefetching_state(self, backward_prefetch, forward_prefetch)
        _init_buffer_state(self, module)
        # extension needs to be set before `_init_param_handle_from_module()`
        _init_extension(self, device_mesh)
        _init_param_handle_from_module(
            self,
            module,
            device_id,
            param_init_fn,
            sync_module_states,
        )
        self._fsdp_wrapped_module = module
        if not use_orig_params:
            _check_orig_params_flattened(self, self._ignored_params)
            _register_flat_param(self, self)

        # `_state_dict_type` controls the `state_dict()` behavior, which is
        # implemented using post-save and pre-load hooks
        _init_state_dict_state(self)
        _register_all_state_dict_hooks(self)


def _apply_fsdp_patch():
    FullyShardedDataParallel.__init__ = __init__
