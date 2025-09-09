# pylint: disable=attr-rgx,function-rgx,invalid-name,unused-argument,missing-type-doc

"""MUSA Graph utilities for TorchInductor backend."""
from __future__ import annotations

import dataclasses
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch._dynamo.utils import counters
from torch._inductor.utils import InputType


perf_hint_log = torch._logging.getArtifactLogger(__name__, "perf_hints")
static_inputs_log = torch._logging.getArtifactLogger(
    __name__, "musagraph_static_inputs"
)


OutputType = List[Optional[Union[int, torch.Tensor]]]
ModelType = Callable[[List[InputType]], OutputType]


@dataclasses.dataclass(frozen=True)
class FunctionID:
    "Unique counter of a function wrapped in musagraphify_impl"
    id: int


@dataclasses.dataclass(frozen=True)
class PlaceholderInfo:
    """
    A serializable version of torch.fx.Node that contains information
    pertinent to placeholder stack traces. We use these in logging and error messages
    related to musagraphs, and will cache these results.
    """

    name: str
    stack_trace: Optional[str]
    # This field is recursive, but never cyclic (since a node never uses itself)
    users: List[PlaceholderInfo]
    mutating_use_stack_trace: Optional[str]


@dataclasses.dataclass(frozen=True)
class WrappedFunction:
    """
    Represents a function that you want to record for MUSA graph replay,
    with a little more metadata so we can identify if we have an applicable
    MUSA graph in our MUSA graph tree for it.
    """

    model: Callable[..., Any]
    static_input_idxs: Sequence[int]
    id: FunctionID
    constants: Tuple[torch.Tensor, ...]
    placeholders: Sequence[PlaceholderInfo]
    mutated_input_idxs: Sequence[int]


def get_mutating_use_stack_trace_from_node(
    placeholder_node: torch.fx.Node,
) -> Optional[str]:
    """
    Retrieve the stack trace from a mutating use of a given FX node.

    This function inspects the users of the provided placeholder_node to find
    any mutating operations, specifically looking for uses of the
    `torch.ops.aten.copy_.default` operator. If such a use is found, it attempts
    to return the associated stack trace metadata.

    If the node has exactly one user, it returns that user's stack trace
    directly if available.

    Args:
        placeholder_node (torch.fx.Node): The FX node whose mutating uses are inspected.

    Returns:
        Optional[str]: The stack trace string associated with a mutating use of the
        node if found; otherwise, None.
    """

    # reinplaced uses might have a single, non-copy_ use
    if len(placeholder_node.users) == 1:
        return next(iter(placeholder_node.users)).meta.get("stack_trace", None)

    for use in placeholder_node.users:
        if use.target == torch.ops.aten.copy_.default:
            if stack_trace := use.meta.get("stack_trace", None):
                return stack_trace

    return None


def get_mutating_use_stack_trace(placeholder_info: PlaceholderInfo) -> Optional[str]:
    return placeholder_info.mutating_use_stack_trace


def to_placeholder_info(placeholder_node: torch.fx.Node) -> PlaceholderInfo:
    """
    Convert a FX placeholder node into a structured PlaceholderInfo object.

    This function extracts the name and stack trace metadata from the given
    placeholder_node. If the node's operation type is "placeholder", it recursively
    collects information from its user nodes and retrieves any mutating use stack trace.

    Args:
        placeholder_node (torch.fx.Node): The FX placeholder node to convert.

    Returns:
        PlaceholderInfo: An object containing the node's name, stack trace, user
        information (recursively), and any mutating use stack trace.
    """

    name = placeholder_node.name
    stack_trace = placeholder_node.meta.get("stack_trace", None)
    users = []
    mutating_use_stack_trace = None
    # Only recurse to users once, since we only care about user's stack traces
    if placeholder_node.op == "placeholder":
        users = [to_placeholder_info(i) for i in placeholder_node.users]
        mutating_use_stack_trace = get_mutating_use_stack_trace_from_node(
            placeholder_node
        )

    return PlaceholderInfo(name, stack_trace, users, mutating_use_stack_trace)


def get_placeholder_info(graph: torch.fx.Graph) -> List[PlaceholderInfo]:
    return [
        to_placeholder_info(node) for node in graph.nodes if node.op == "placeholder"
    ]


def format_default_skip_message(reason: str) -> str:
    return f"skipping musagraphs due to {reason}"


def get_mutation_stack_trace(
    placeholders: Sequence[PlaceholderInfo], mutation_indices: Sequence[int]
) -> str:
    """
    Retrieve the stack trace for mutations from a list of placeholder infos based on mutation
    indices.

    Iterates over the provided mutation indices, attempts to get the mutating use stack trace
    for each corresponding placeholder. Returns the first found stack trace along with
    a formatted message indicating mutated inputs. If no stack trace is found, returns
    only the formatted message.

    Args:
        placeholders (Sequence[PlaceholderInfo]): A sequence of PlaceholderInfo objects.
        mutation_indices (Sequence[int]): Indices indicating which placeholders are mutated.

    Returns:
        str: A message describing the mutated inputs and their associated stack trace if found.
    """

    stack_trace: Optional[str] = ""

    for idx in mutation_indices:
        placeholder = placeholders[idx]
        if stack_trace := get_mutating_use_stack_trace(placeholder):
            break

    msg = format_default_skip_message(
        f"mutated inputs ({len(mutation_indices)} instances)"
    )
    if stack_trace:
        return f"{msg}. Found from : \n {stack_trace}"

    return msg


def check_for_mutation(
    func: WrappedFunction,
    inputs: List[InputType],
    is_musa_graph_recorded_tensor: Callable[[torch.Tensor], bool],
) -> Optional[str]:
    """
    Check for mutations in function inputs while filtering out static or recorded tensors.

    This function identifies indices of inputs that are mutated but are neither static inputs
    nor tensors recorded by the MUSA graph. The check behavior differs depending on whether
    `torch._inductor.config.triton.cudagraph_trees` is enabled:
      - If enabled, it filters mutated inputs to exclude static inputs and MUSA graph tensors.
      - Otherwise, it considers all mutated input indices.

    It then logs the static input indices and the filtered mutation indices, and
    returns the mutation stack trace if any mutation indices remain.

    Args:
        func (WrappedFunction): The wrapped function object containing metadata like mutated
                                and static input indices.
        inputs (List[InputType]): The list of inputs passed to the function.
        is_musa_graph_recorded_tensor (Callable[[torch.Tensor], bool]): A callable to determine
            if a tensor is recorded by the MUSA graph.

    Returns:
        Optional[str]: The stack trace message for the mutation if mutations are detected;
                        otherwise, None.
    """

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        # checking if mutation is only on parameters/static inputs
        mutation_indices: Sequence[int] = [
            idx
            for idx in func.mutated_input_idxs
            if not (
                idx in func.static_input_idxs
                or is_musa_graph_recorded_tensor(inputs[idx])  # type: ignore[arg-type]
            )
        ]
    else:
        mutation_indices = func.mutated_input_idxs

    static_inputs_log.debug(
        "check mutation static input indices: %s", func.static_input_idxs
    )
    static_inputs_log.debug("check mutation mutation indices: %s", mutation_indices)

    return (
        get_mutation_stack_trace(func.placeholders, mutation_indices)
        if mutation_indices
        else None
    )


def _get_use_stack_trace(node) -> Optional[str]:
    for use in node.users:
        if stack_trace := use.meta.get("stack_trace", None):
            return stack_trace
    return None


def check_multiple_devices_or_any_cpu_nodes(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
) -> Optional[str]:
    """
    Check if the device-to-node mapping contains any CPU nodes or multiple devices.

    This function performs the following checks in order:
    1. If there is a node mapped to the CPU device, it attempts to retrieve its stack trace.
       Returns a formatted skip message with the CPU node name and stack trace if found.
    2. If the mapping contains exactly one device and that device is of type "musa",
       returns None indicating no issues.
    3. Otherwise, returns a formatted skip message indicating multiple devices are present.

    Args:
        device_node_mapping (Dict[torch.device, torch.fx.Node]): Mapping from devices to FX nodes.

    Returns:
        Optional[str]: Formatted skip message describing the issue if CPU or multiple devices
        are detected; otherwise, None.
    """

    if cpu_node := device_node_mapping.get(torch.device("cpu")):
        msg = f"cpu device ({cpu_node.name})"
        if stack_trace := _get_use_stack_trace(cpu_node):
            return format_default_skip_message(f"{msg}. Found from : \n {stack_trace}")

        return format_default_skip_message(msg)

    if (
        len(device_node_mapping) == 1
        and next(iter(device_node_mapping.keys())).type == "musa"
    ):
        return None

    keys_repr = (repr(key) for key in device_node_mapping.keys())
    return format_default_skip_message(f"multiple devices: {', '.join(keys_repr)}")


def check_lowering_disable_musagraph(
    device_node_mapping: Dict[torch.device, torch.fx.Node]
):
    return check_multiple_devices_or_any_cpu_nodes(device_node_mapping)


def log_musagraph_skip_and_bump_counter(msg):
    perf_hint_log.warning(msg)
    counters["inductor"]["musagraph_skips"] += 1


@dataclasses.dataclass
class BoxedDeviceIndex:
    value: Optional[int]

    def set(self, device_idx: Optional[int]):
        assert device_idx is None or isinstance(device_idx, int)
        self.value = device_idx


def check_for_mutation_ignore_musa_graph_managed_tensor(
    gm: torch.fx.GraphModule, compiled_graph, static_input_idxs: Sequence[int]
) -> Optional[str]:
    """
    Check for mutations in the compiled graph inputs, ignoring static inputs.

    When `torch._inductor.config.triton.cudagraph_trees` is enabled, this function filters out
    mutations on static inputs and returns a detailed mutation stack trace if any
    non-static mutations are found.

    If the configuration is disabled, it only checks if there are any mutated inputs
    in the compiled graph and returns a default skip message if mutations exist.

    Args:
        gm (torch.fx.GraphModule): The FX graph module to inspect.
        compiled_graph: The compiled graph object with mutation metadata.
        static_input_idxs (Sequence[int]): Indices of inputs considered static.

    Returns:
        Optional[str]: A stack trace message describing the mutations if found;
        otherwise, None.
    """

    default_msg = format_default_skip_message("mutated inputs")

    # doesnt work for non-trees because the warmup run would apply mutation twice
    if torch._inductor.config.triton.cudagraph_trees:
        unique_idxs = set(static_input_idxs)
        # checking if mutation is only on parameters/static inputs
        mutation_indices = [
            idx for idx in compiled_graph.mutated_input_idxs if idx not in unique_idxs
        ]
        has_mutation = len(mutation_indices) != 0
        if not has_mutation:
            return None
        placeholders = get_placeholder_info(gm.graph)
        return get_mutation_stack_trace(placeholders, mutation_indices)
    has_mutation = len(compiled_graph.mutated_inputs) != 0
    return None if not has_mutation else default_msg


def get_placeholder_stack_trace(placeholder: PlaceholderInfo) -> Optional[str]:
    """
    Gets the first non-empty stack trace of a placeholder or its users.
    """

    if placeholder.stack_trace:
        return placeholder.stack_trace

    for user in placeholder.users:
        if user.stack_trace:
            return user.stack_trace

    return None


class CheckInvariantStatus(Enum):
    """
    Enum representing various invariant check statuses for musagraph.

    Attributes:
        SUCCESS: Check invariant succeeded.
        MusagraphManagedIdxMismatch: Previously managed data pointers are not stable.
        StaticInputIdxMismatch: Static tensor input addresses are not stable.
        ExpectedDeadIndicesBeforeGraphMismatch: Expected dead indices before graph are live.
    """

    # Check invariant succeeded
    SUCCESS = 1

    # Previously managed data pointers are not stable
    MusagraphManagedIdxMismatch = 2

    # Static tensor input addresses are not stable
    StaticInputIdxMismatch = 3

    # Expected dead indices before graph are live
    ExpectedDeadIndicesBeforeGraphMismatch = 4

    def __str__(self) -> str:
        if self.name == "MusagraphManagedIdxMismatch":
            return "musagraph managed tensor data pointer changed"
        if self.name == "StaticInputIdxMismatch":
            return "static input data pointer changed"
        if self.name == "ExpectedDeadIndicesBeforeGraphMismatch":
            return "expected dead indices before graph are live"
        return f"{self.name}: {self.value}"


def log_data_ptr_mismatch(
    placeholders: Sequence[PlaceholderInfo],
    inputs: List[InputType],
    recorded_data_ptr: Sequence[Optional[int]],
    target_idxs: Sequence[int],
    mismatch: CheckInvariantStatus,
) -> str:
    """
    Logs the mismatch between input data pointers and recorded data pointers.
    This checks only idxs in target_idxs.
    """

    assert len(inputs) == len(recorded_data_ptr) and len(inputs) == len(
        placeholders
    ), "length mismatch between inputs, recorded_data_ptr, and placeholders"

    t_tensors = [inputs[i] for i in target_idxs]
    t_data_ptrs = [recorded_data_ptr[i] for i in target_idxs]
    error_msg = f"{mismatch}.\n"
    for i, (tensor, data_ptr) in enumerate(zip(t_tensors, t_data_ptrs)):
        assert isinstance(tensor, torch.Tensor)
        index = target_idxs[i]
        if tensor.data_ptr() != data_ptr:
            placeholder = placeholders[index]
            error_msg = (
                f"{error_msg}input name: {placeholder.name}. "
                f"data pointer changed from {data_ptr} to {tensor.data_ptr()}. "
                f"input stack trace: {get_placeholder_stack_trace(placeholder)}\n"
            )
    return error_msg


def maybe_warning_due_to_dynamic_shape(
    fn_cache: Dict[Tuple[int, ...], Callable[..., Any]],
    new_int_key: Any,
) -> bool:
    """
    Check if the number of recorded MUSAGraphs exceeds the dynamic shape warning limit,
    and log a warning if necessary.

    This function monitors how many distinct input shapes have been recorded as separate
    MUSAGraphs. Recording too many distinct graphs may introduce performance overhead.
    When the count exceeds a configured threshold, it logs a warning suggesting users to
    either pad inputs to fixed shapes or enable a config flag to skip dynamic graphs.

    Args:
        fn_cache (Dict[Tuple[int, ...], Callable[..., Any]]): Cache mapping input shape keys
                                                              to graphs.
        new_int_key (Any): The new input shape key being checked (not directly used here).

    Returns:
        bool: True if a warning was logged due to exceeding the dynamic shape limit;
              False otherwise.
    """

    num_musagraphs = len(fn_cache.keys()) + 1

    def warn_msg():
        return (
            "MUSAGraph supports dynamic shapes by recording a new graph for each "
            "distinct input size. Recording too many MUSAGraphs may lead to "
            f"extra overhead. We have observed {num_musagraphs} distinct sizes. "
            "Please consider the following options for better performance: "
            "a) padding inputs to a few fixed number of shapes; or b) set "
            "torch._inductor.config.triton.musagraph_skip_dynamic_graphs=True. "
            "Set torch._inductor.config.triton.musagraph_dynamic_shape_warn_limit=None "
            "to silence this warning."
        )

    if (
        torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
        and num_musagraphs
        > torch._inductor.config.triton.cudagraph_dynamic_shape_warn_limit
    ):
        perf_hint_log.warning(warn_msg())
        return True

    return False


@dataclasses.dataclass(frozen=True)
class MusagraphCachedInfo:
    """
    Info needed to realign inputs
    """

    placeholders: Sequence[PlaceholderInfo]
    stack_traces: List[Optional[str]]
    musagraph_fail_reasons: List[str]
