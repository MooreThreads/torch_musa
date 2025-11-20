# pylint: disable=all
import weakref
from typing import (
    Any,
    cast,
    Dict,
    Optional,
)

import torch
import torch.distributed as dist
from torch.distributed._state_dict_utils import (
    _iterate_state_dict,
    _identity_func,
)


def _create_cpu_state_dict(
    state_dict: Dict[str, Any], pin_memory: bool = False, share_memory: bool = False
) -> Dict[str, Any]:
    """
    Given a state_dict, create another state_dict with the same structure and elements.
    However, all tensors in the returned state_dict are new tensors on CPU. These
    tensors can be placed on pin_memory or share_memory based on the provided arguments.

    .. warning::
        Setting both `pin_memory` and `share_memory` to True significantly increases the
        latency of this method because of the nuances which require us to register memory
        as pinned directly as opposed to relying on the pin_memory cache allocator. This
        option should only be used for long lived tensors which are required to be shared.
        This is not the case as long as at least one of `pin_memory` or `share_memory` is
         set to False.

    """

    def tensor_func(
        obj: torch.Tensor,
        pg: Optional[dist.ProcessGroup],
        device: Optional[torch.device],
        _: Any,
    ) -> torch.Tensor:
        if len(obj.size()) == 0:
            return torch.tensor(0, dtype=obj.dtype)

        if share_memory:
            t = torch.empty(*tuple(obj.size()), dtype=obj.dtype)
            t = t.share_memory_()
            if pin_memory:

                def unpin_memory(t):
                    succ = int(torch.musa.musart().musaHostUnregister(t.data_ptr()))
                    assert (
                        succ == 0
                    ), f"Unpinning shared memory failed with error-code: {succ}"

                weakref.finalize(t, unpin_memory, t)
                succ = int(
                    torch.musa.musart().musaHostRegister(
                        t.data_ptr(),
                        t.numel() * t.element_size(),
                        1,  # lines up with 'cudaHostRegisterPortable'
                    )
                )
                assert (
                    succ == 0
                ), f"Pinning shared memory failed with error-code: {succ}"
            return t
        elif pin_memory:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype).pin_memory()
        else:
            return torch.empty(*tuple(obj.size()), dtype=obj.dtype)

    ret = _iterate_state_dict(
        state_dict,
        _identity_func,
        _identity_func,
        tensor_func,
        pg=None,
        device=None,
        cpu_offload=False,
        ranks_only=(),
        type_check=False,
    )
    return ret


def _apply_state_dict_utils_patch() -> None:
    """
    Apply the state_dict_utils patch.
    """
    from torch.distributed import _state_dict_utils

    _state_dict_utils._create_cpu_state_dict = _create_cpu_state_dict
