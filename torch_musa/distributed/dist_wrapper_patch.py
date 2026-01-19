"""Patch DistWrapper reduce_scatter for torch musa."""

from typing import Callable, cast, Optional, Union
from torch.distributed.checkpoint.utils import T, R, _DistWrapper, _get_failure_dict

from torch.distributed.checkpoint.api import (
    _wrap_exception,
    CheckpointException,
    WRAPPED_EXCEPTION,
)


def reduce_scatter(
    self,
    step: str,
    map_fun: Callable[[], T],
    reduce_fun: Callable[[list[T]], list[R]],
) -> R:
    """
    Compute a value on each rank, then do centralized reduce on a single rank,
    followed by a scatter.

    This method operates in the following way:
        Run ``map_fun`` on all ranks
        Gather results on rank 0
        Call ``reduce_fun`` on all those values
        Scatter to each rank part of the result.
    """
    local_data: Union[WRAPPED_EXCEPTION, T]
    try:
        local_data = map_fun()
    except BaseException as exc:  ## pylint: disable=W0718
        local_data = _wrap_exception(exc)

    all_data = self.all_gather_object(local_data)
    all_results: Optional[list[Union[R, CheckpointException]]] = None

    assert all_data is not None
    node_failures = _get_failure_dict(all_data)

    if len(node_failures) == 0:
        try:
            # N.B. why can't mypy cast List[R] to List[Union[R, WRAPPED_EXCEPTION]]?
            all_results = cast(
                list[Union[R, CheckpointException]],
                reduce_fun(cast(list[T], all_data)),
            )
        except BaseException as exc:  # pylint: disable=W0718
            node_failures[self.rank] = _wrap_exception(exc)

    if len(node_failures) > 0:
        all_results = [CheckpointException(step, node_failures)] * self.get_world_size()

    result = all_results[self.rank]

    if isinstance(result, CheckpointException):
        raise result
    return result


def _apply_dist_wrapper_patch() -> None:
    _DistWrapper.reduce_scatter = reduce_scatter
