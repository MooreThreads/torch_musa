"""Torch musa autograd profiler."""

import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn

import torch
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
    _disable_profiler,
    _enable_profiler,
    _kineto_step,
    _prepare_profiler,
    _ProfilerResult,
    _supported_activities,
    DeviceType,
    kineto_available,
    ProfilerActivity,
    ProfilerConfig,
    ProfilerState,
)
from torch.futures import Future

from .profiler_util import (
    _filter_name,
    _filter_stack_entry,
    _rewrite_name,
    EventList,
    FunctionEvent,
    MEMORY_EVENT_NAME,
    MemRecordsAcc,
    OUT_OF_MEMORY_EVENT_NAME,
)

__all__ = [
    "profile",
    "record_function",
    "emit_itt",
    "emit_nvtx",
    "load_nvprof",
    "EnforceUnique",
    "parse_nvprof_trace",
    "KinetoStepTracker",
    "EventList",
    "FunctionEvent",
    "MemRecordsAcc",
]

try:
    # Available in Python >= 3.2
    from contextlib import ContextDecorator as _ContextDecorator
except ImportError:
    import functools

    class _ContextDecorator:  # type: ignore[no-redef]

        def __enter__(self):
            raise NotImplementedError

        def __exit__(self, exc_type, exc_val, exc_tb):
            raise NotImplementedError

        def __call__(self, func):
            @functools.wraps(func)
            def wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            return wrapped


# pylint: disable=C0103
class profile:
    """Context manager that manages autograd profiler state and holds a summary of results.
    Under the hood it just records events of functions being executed in C++ and
    exposes those events to Python. You can wrap any code into it and it will
    only report runtime of PyTorch functions.
    Note: profiler is thread local and is automatically propagated into the async tasks

    Args:
        enabled (bool, optional): Setting this to False makes this context manager a no-op.

        use_musa (bool, optional): Enables timing of MUSA events as well using the musaEvent API.
            Adds approximately 4us of overhead to each tensor operation.

        record_shapes (bool, optional): If shapes recording is set, information
            about input dimensions will be collected. This allows one to see which
            dimensions have been used under the hood and further group by them
            using prof.key_averages(group_by_input_shape=True). Please note that
            shape recording might skew your profiling data. It is recommended to
            use separate runs with and without shape recording to validate the timing.
            Most likely the skew will be negligible for bottom most events (in a case
            of nested function calls). But for higher level functions the total
            self cpu time might be artificially increased because of the shape
            collection.

        with_flops (bool, optional): If with_flops is set, the profiler will estimate
            the FLOPs (floating point operations) value using the operator's input shape.
            This allows one to estimate the hardware performance. Currently,
            this option only works for the matrix multiplication and 2D convolution operators.

        profile_memory (bool, optional): track tensor memory allocation/deallocation.

        with_stack (bool, optional): record source information (file and line number) for the ops.

        with_modules (bool): record module hierarchy (including function names)
            corresponding to the callstack of the op. e.g. If module A's forward call's
            module B's forward which contains an aten::add op,
            then aten::add's module hierarchy is A.B
            Note that this support exist, at the moment, only for TorchScript models
            and not eager mode models.

        use_kineto (bool, optional): experimental, enable profiling with Kineto profiler.

        use_cpu (bool, optional): profile CPU events; setting to ``False`` requires
            ``use_kineto=True`` and can be used to lower the overhead for GPU-only profiling.

        experimental_config (_ExperimentalConfig) : A set of experimental options
            used by profiler libraries like Kineto. Note, backward compatibility is not guaranteed.


    .. warning:
        Enabling memory profiling or source attribution incurs additional profiler
        overhead

    .. warning:
        This context managers should not be called recursively, i.e. no nested
        instances are allowed

    .. warning:
        Due to some MUSA multiprocessing limitations (multiprocessing-musa-note_),
        one cannot use the profiler with ``use_musa = True`` to benchmark
        DataLoaders with ``num_workers > 0``. If you wish to benchmark data loading,
        please use ``use_musa = False`` or ``num_workers = 0``.

    Example:
        >>> # xdoctest: +SKIP
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        >>>     for _ in range(100):  # any normal python code, really!
        >>>         y = x ** 2
        >>>         y.backward()
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total   CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        mul                                  32.048ms         32.048ms         200
        pow                                  27.041ms         27.041ms         200
        PowBackward0                         9.727ms          55.483ms         100
        torch::autograd::AccumulateGrad      9.148ms          9.148ms          100
        torch::autograd::GraphRoot           691.816us        691.816us        100
        -----------------------------------  ---------------  ---------------  ---------------

    """

    def __init__(
        self,
        enabled=True,
        *,
        use_musa=False,
        record_shapes=False,
        with_flops=False,
        profile_memory=False,
        with_stack=False,
        with_modules=False,
        use_kineto=False,
        use_cpu=True,
        experimental_config=None,
    ):
        self.enabled: bool = enabled
        if not self.enabled:
            return
        self.use_musa = use_musa
        self.function_events: Optional[EventList] = None
        self.entered = False
        self.record_shapes = record_shapes
        self.with_flops = with_flops
        self.record_shapes |= self.with_flops
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_modules = with_modules
        self.use_cpu = use_cpu
        if experimental_config is None:
            experimental_config = _ExperimentalConfig()
        self.experimental_config = experimental_config
        self.kineto_results: Optional[_ProfilerResult] = None

        if not self.use_cpu:
            assert (
                use_kineto
            ), "Device-only events supported only with Kineto (use_kineto=True)"

        if self.use_musa and not torch.musa.is_available():
            warn("MUSA is not available, disabling MUSA profiling")
            self.use_musa = False

        self.kineto_activities = set()
        if self.use_cpu:
            self.kineto_activities.add(ProfilerActivity.CPU)

        self.profiler_kind = ProfilerState.KINETO
        if self.use_musa:
            if not use_kineto or ProfilerActivity.MUSA not in _supported_activities():
                assert self.use_cpu, "Legacy MUSA profiling requires use_cpu=True"
                self.profiler_kind = ProfilerState.KINETO_GPU_FALLBACK
            else:
                self.kineto_activities.add(ProfilerActivity.MUSA)

        assert (
            len(self.kineto_activities) > 0
        ), "No activities specified for the profiler"

    def config(self):
        return ProfilerConfig(
            self.profiler_kind,
            self.record_shapes,
            self.profile_memory,
            self.with_stack,
            self.with_flops,
            self.with_modules,
            self.experimental_config,
        )

    def __enter__(self):
        if not self.enabled:
            return None
        if self.entered:
            raise RuntimeError("Profiler context manager is not reentrant")
        self._prepare_trace()
        self._start_trace()
        return self

    def _prepare_trace(self):
        self.entered = True
        _prepare_profiler(self.config(), self.kineto_activities)

    def _start_trace(self):
        self.entered = True
        _enable_profiler(self.config(), self.kineto_activities)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return None
        if self.use_musa:
            torch.musa.synchronize()
        self.kineto_results = _disable_profiler()
        parsed_results = self._parse_kineto_results(self.kineto_results)
        self.function_events = EventList(
            parsed_results,
            use_musa=self.use_musa,
            profile_memory=self.profile_memory,
            with_flops=self.with_flops,
        )
        self.function_events._build_tree()
        return False

    def __repr__(self):
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        return repr(self.function_events)

    def __str__(self):
        if self.function_events is None:
            return "<unfinished torch.autograd.profile>"
        return str(self.function_events)

    def _check_finish(self):
        if self.function_events is None:
            raise RuntimeError("Profiler didn't finish running")

    def table(
        self,
        sort_by=None,
        row_limit=100,
        max_src_column_width=75,
        max_name_column_width=55,
        max_shapes_column_width=80,
        header=None,
        top_level_events_only=False,
    ):
        """Show profiling table."""
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.table(
            sort_by=sort_by,
            row_limit=row_limit,
            max_src_column_width=max_src_column_width,
            max_name_column_width=max_name_column_width,
            max_shapes_column_width=max_shapes_column_width,
            header=header,
            top_level_events_only=top_level_events_only,
        )

    table.__doc__ = EventList.table.__doc__

    def export_chrome_trace(self, path):
        self._check_finish()
        if kineto_available():
            self.kineto_results.save(path)  # type: ignore[union-attr]
            return None
        return self.function_events.export_chrome_trace(path)  # type: ignore[union-attr]

    export_chrome_trace.__doc__ = EventList.export_chrome_trace.__doc__

    def export_stacks(self, path: str, metric: str = "self_cpu_time_total"):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        assert self.with_stack, "export_stacks() requires with_stack=True"
        return self.function_events.export_stacks(path, metric)

    def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)

    key_averages.__doc__ = EventList.key_averages.__doc__

    def total_average(self):
        self._check_finish()
        assert self.function_events is not None, "Expected profiling results"
        return self.function_events.total_average()

    total_average.__doc__ = EventList.total_average.__doc__

    @property
    def self_cpu_time_total(self):
        """Returns total time spent on CPU obtained as a sum of
        all self times across all the events.
        """
        self._check_finish()
        assert self.function_events is not None
        return self.function_events.self_cpu_time_total

    def _parse_kineto_results(self, result):
        # result.events() has most of the events - PyTorch op-level and device-level events

        trace_start_us = result.trace_start_us()
        mem_records = [
            [evt, False] for evt in result.events() if evt.name() == MEMORY_EVENT_NAME
        ]
        oom_records = [
            evt for evt in result.events() if evt.name() == OUT_OF_MEMORY_EVENT_NAME
        ]
        mem_records_acc = MemRecordsAcc(mem_records)

        def _cpu_memory_usage(mem_record):
            return (
                mem_record.nbytes()
                if mem_record.device_type()
                in [DeviceType.CPU, DeviceType.MKLDNN, DeviceType.IDEEP]
                else 0
            )

        def _musa_memory_usage(mem_record):
            return mem_record.nbytes()

        # Create and return FunctionEvent list
        function_events = []
        musa_corr_map: Dict[int, List[FunctionEvent]] = {}
        max_evt_id = 0
        for kineto_event in result.events():
            if _filter_name(kineto_event.name()):
                continue
            rel_start_us = kineto_event.start_us() - trace_start_us
            rel_end_us = rel_start_us + kineto_event.duration_us()
            abs_end_us = kineto_event.start_us() + kineto_event.duration_us()

            cpu_memory_usage = 0
            musa_memory_usage = 0
            if kineto_event.device_type() == DeviceType.CPU:
                # find the corresponding memory allocation events
                for mem_record in mem_records_acc.in_interval(
                    kineto_event.start_us(), abs_end_us
                ):
                    cpu_memory_usage += _cpu_memory_usage(mem_record[0])
                    musa_memory_usage += _musa_memory_usage(mem_record[0])
                    mem_record[1] = True

            is_async = kineto_event.is_async() or (
                kineto_event.start_thread_id() != kineto_event.end_thread_id()
            )

            function_event = FunctionEvent(
                id=kineto_event.correlation_id(),
                name=_rewrite_name(name=kineto_event.name(), with_wildcard=True),
                trace_name=_rewrite_name(name=kineto_event.name(), with_wildcard=False),
                thread=kineto_event.start_thread_id(),
                start_us=rel_start_us,
                end_us=rel_end_us,
                fwd_thread=kineto_event.fwd_thread_id(),
                input_shapes=kineto_event.shapes(),
                stack=[
                    entry
                    for entry in kineto_event.stack()
                    if _filter_stack_entry(entry)
                ],
                scope=kineto_event.scope(),
                cpu_memory_usage=cpu_memory_usage,
                musa_memory_usage=musa_memory_usage,
                is_async=is_async,
                sequence_nr=kineto_event.sequence_nr(),
                device_type=kineto_event.device_type(),
                device_index=kineto_event.device_index(),
                flops=kineto_event.flops(),
            )
            max_evt_id = (
                function_event.id if function_event.id > max_evt_id else max_evt_id
            )
            if (
                function_event.device_type == DeviceType.CPU
                and not function_event.is_async
            ):
                # Check if we have MUSA time as a fallback
                # TODO: The following is not adapted to MUSA temporarily
                musa_time = kineto_event.cuda_elapsed_us()
                if musa_time > 0:
                    function_event.append_kernel(
                        function_event.name, function_event.device_index, musa_time
                    )
                    function_event.is_legacy = True
            function_events.append(function_event)
            corr_id = kineto_event.linked_correlation_id()
            if corr_id > 0:
                if corr_id not in musa_corr_map:
                    musa_corr_map[corr_id] = []
                musa_corr_map[corr_id].append(function_event)

        # associate MUSA kernels and MUSA runtime (CPU) with CPU events
        for function_event in function_events:
            if (
                function_event.device_type == DeviceType.CPU
                and not function_event.is_async
                and function_event.id in musa_corr_map
            ):
                for f_evt in musa_corr_map[function_event.id]:
                    if f_evt.device_type != DeviceType.CPU:
                        function_event.append_kernel(
                            f_evt.name,
                            f_evt.device_index,
                            f_evt.time_range.end - f_evt.time_range.start,
                        )
                    elif f_evt.device_type == DeviceType.CPU:
                        # make sure that 'thread' of a CPU Kineto (e.g. MUSA Runtime) event is
                        # associated with the 'thread' of the corresponding linked PyTorch event
                        # to properly track parents and children
                        f_evt.thread = function_event.thread

        # pylint: disable=C0103
        def createFunctionEventForMemoryEvents(evt):
            rel_start_us = evt.start_us() - trace_start_us
            func_event = FunctionEvent(
                id=max_evt_id,
                name=evt.name(),
                trace_name=None,  # not outputting in the trace
                thread=evt.start_thread_id(),
                start_us=rel_start_us,
                end_us=rel_start_us,  # no duration
                fwd_thread=evt.start_thread_id(),
                input_shapes=[],
                stack=[],
                scope=0,  # RecordScope::FUNCTION
                cpu_memory_usage=_cpu_memory_usage(evt),
                musa_memory_usage=_musa_memory_usage(evt),
                is_async=False,
                sequence_nr=-1,
                device_type=DeviceType.CPU,
                device_index=0,
            )
            return func_event

        # output top-level memory events
        for mem_record in mem_records:
            if not mem_record[1]:
                max_evt_id += 1
                function_event = createFunctionEventForMemoryEvents(mem_record[0])
                function_events.append(function_event)

        for oom_record in oom_records:
            max_evt_id += 1
            function_event = createFunctionEventForMemoryEvents(oom_record)
            function_events.append(function_event)

        function_events.sort(
            key=lambda evt: [evt.time_range.start, -evt.time_range.end]
        )
        return function_events


# pylint: disable=C0103
class record_function(_ContextDecorator):
    """Context manager/function decorator that adds a label to a block of
    Python code (or function) when running autograd profiler. It is
    useful when tracing the code profile.

    Args:
        name (str): Label assigned to the block of code.
        node_id (int): ID of node, for distributed profiling. Unset in
        non-distributed cases.

    Example:
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> x = torch.randn((1, 1), requires_grad=True)
        >>> with torch.autograd.profiler.profile() as prof:
        ...     y = x ** 2
        ...     with torch.autograd.profiler.record_function("label-z"): # label the block
        ...         z = y ** 3
        ...     y.backward()
        ...
        >>> # xdoctest: +IGNORE_WANT
        >>> # NOTE: some columns were removed for brevity
        >>> print(prof.key_averages().table(sort_by="self_cpu_time_total"))
        -----------------------------------  ---------------  ---------------  ---------------
        Name                                 Self CPU total %  CPU time avg     Number of Calls
        -----------------------------------  ---------------  ---------------  ---------------
        pow                                  60.77%           47.470us         3
        mul                                  21.73%           25.465us         2
        PowBackward0                         12.03%           121.891us        1
        torch::autograd::AccumulateGrad      2.70%            6.324us          1
        label-z                              2.13%            12.421us         1
        torch::autograd::GraphRoot           0.64%            1.503us          1
        -----------------------------------  ---------------  ---------------  ---------------
        Self CPU time total: 234.344us
        MUSA time total: 0.000us

    """

    def __init__(self, name: str, args: Optional[str] = None):
        self.name: str = name
        self.args: Optional[str] = args
        # Whether or not we should run record function's end callbacks when exiting.
        self.run_callbacks_on_exit: bool = True
        # TODO: TorchScript ignores standard type annotation here
        # self.record: Optional["torch.classes.profiler._RecordFunction"] = None
        self.record = torch.jit.annotate(
            Optional["torch.classes.profiler._RecordFunction"], None
        )

    def __enter__(self):
        self.record = torch.ops.profiler._record_function_enter_new(
            self.name, self.args
        )
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        if not self.run_callbacks_on_exit:
            return

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        assert record is not None

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                torch.ops.profiler._record_function_exit._RecordFunction(record)
        else:
            torch.ops.profiler._record_function_exit(record)

    def _call_end_callbacks_on_future(self, fut: Future[Any]) -> Future[Any]:
        """
        _call_end_callbacks_on_future is meant to be used for profiling async
        calls that return a future. Calling this function will extend recording
        beyond this scope, until the future is satisfied. It is useful for profiling
        the end to end time of asynchronous calls. This function should only be called
        once to attach the callback onto the future, and will throw if called multiple
        times.

        Args:
            fut: (torch._C.Future): future for which to schedule
            callback for.

        Returns:
            A future that completes with the value of the passed in future when
            the profiling callbacks have ran.

        """
        # Throw if we have already attached a callback onto the future.
        if not self.run_callbacks_on_exit:
            raise RuntimeError("_call_end_callbacks_on_future can only be called once.")

        # We are scheduling to run this RecordFunction's end callbacks when the
        # passed in future completes, so don't run end callbacks on exit.
        self.run_callbacks_on_exit = False

        # Local variable is needed by TorchScript to refine Optional[T] to T
        record = self.record
        assert record is not None

        # TODO: Too slow with __torch_function__ handling enabled
        # See https://github.com/pytorch/pytorch/issues/76410
        if not torch.jit.is_scripting():
            with torch._C.DisableTorchFunctionSubclass():
                profiled_future = (
                    torch.ops.profiler._call_end_callbacks_on_jit_fut._RecordFunction(
                        record, fut
                    )
                )
        else:
            profiled_future = torch.ops.profiler._call_end_callbacks_on_jit_fut(
                record, fut
            )
        return profiled_future


# pylint: disable=C0103
class emit_itt:
    """Context manager that makes every autograd operation emit an ITT range.

    It is useful when running the program under Intel(R) VTune Profiler::

        vtune <--vtune-flags> <regular command here>

    The Instrumentation and Tracing Technology (ITT) API enables your application to generate and
    control the collection of trace data during its execution across different Intel tools. This
    context manager is to annotate Intel(R) VTune Profiling trace. With help of this context
    manager, you will be able to see labled ranges in Intel(R) VTune Profiler GUI.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args: enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
    Default: ``True``. record_shapes (bool, optional): If ``record_shapes=True``, the itt range
    wrapping each autograd op will append information about the sizes of Tensor arguments
    received by that op, in the following format: ``[[arg0.size(0), arg0.size(1), ...],
    [arg1.size(0), arg1.size(1), ...], ...]`` Non-tensor arguments will be represented by ``[]``.
    Arguments will be listed in the order they are received by the backend op. Please note that
    this order may not match the order in which those arguments were passed on the Python side.
    Also note that shape recording may increase the overhead of itt range creation. Default:
    ``False``

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.autograd.profiler.emit_itt():
        ...     model(x)

    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return None
        if self.entered:
            raise RuntimeError("ITT annotation context manager is not reentrant")
        self.entered = True
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.ITT,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return None
        _disable_profiler()
        return False


# pylint: disable=C0103
class emit_nvtx:
    """Context manager that makes every autograd operation emit an NVTX range.

    It is useful when running the program under nvprof::

        nvprof --profile-from-start off -o trace_name.prof -- <regular command here>

    Unfortunately, there's no way to force nvprof to flush the data it collected
    to disk, so for MUSA profiling one has to use this context manager to annotate
    nvprof traces and wait for the process to exit before inspecting them.
    Then, either NVIDIA Visual Profiler (nvvp) can be used to visualize the timeline, or
    :func:`torch.autograd.profiler.load_nvprof` can load the results for inspection
    e.g. in Python REPL.

    .. warning:
        This context manager should not be called recursively, i.e. at most one
        instance should be enabled at any given time.

    Args: enabled (bool, optional): Setting ``enabled=False`` makes this context manager a no-op.
    Default: ``True``. record_shapes (bool, optional): If ``record_shapes=True``, the nvtx range
    wrapping each autograd op will append information about the sizes of Tensor arguments
    received by that op, in the following format: ``[[arg0.size(0), arg0.size(1), ...],
    [arg1.size(0), arg1.size(1), ...], ...]`` Non-tensor arguments will be represented by ``[]``.
    Arguments will be listed in the order they are received by the backend op. Please note that
    this order may not match the order in which those arguments were passed on the Python side.
    Also note that shape recording may increase the overhead of nvtx range creation. Default:
    ``False``

    Example:
        >>> # xdoctest: +SKIP("undefined variables")
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_AUTOGRAD_PROFILER)
        >>> with torch.musa.profiler.profile():
        ...     model(x)  # Warmup MUSA memory allocator and profiler
        ...     with torch.autograd.profiler.emit_nvtx():
        ...         model(x)

    **Forward-backward correlation**

    When viewing a profile created using :class:`emit_nvtx` in the Nvidia Visual Profiler,
    correlating each backward-pass op with the corresponding forward-pass op can be difficult.
    To ease this task, :class:`emit_nvtx` appends sequence number information to the ranges it
    generates.

    During the forward pass, each function range is decorated with ``seq=<N>``.  ``seq`` is a
    running counter, incremented each time a new backward Function object is created and stashed
    for backward. Thus, the ``seq=<N>`` annotation associated with each forward function range
    tells you that if a backward Function object is created by this forward function,
    the backward object will receive sequence number N. During the backward pass, the top-level
    range wrapping each C++ backward Function's ``apply()`` call is decorated with ``stashed
    seq=<M>``.  ``M`` is the sequence number that the backward object was created with.  By
    comparing ``stashed seq`` numbers in backward with ``seq`` numbers in forward, you can track
    down which forward op created each backward Function.

    Any functions executed during the backward pass are also decorated with ``seq=<N>``.  During
    default backward (with ``create_graph=False``) this information is irrelevant, and in fact,
    ``N`` may simply be 0 for all such functions.  Only the top-level ranges associated with
    backward Function objects' ``apply()`` methods are useful, as a way to correlate these Function
    objects with the earlier forward pass.

    **Double-backward**

    If, on the other hand, a backward pass with ``create_graph=True`` is underway (in other
    words, if you are setting up for a double-backward), each function's execution during
    backward is given a nonzero, useful ``seq=<N>``.  Those functions may themselves create
    Function objects to be executed later during double-backward, just as the original functions
    in the forward pass did. The relationship between backward and double-backward is
    conceptually the same as the relationship between forward and backward: The functions still
    emit current-sequence-number-tagged ranges, the Function objects they create still stash
    those sequence numbers, and during the eventual double-backward, the Function objects'
    ``apply()`` ranges are still tagged with ``stashed seq`` numbers, which can be compared to
    `seq` numbers from the backward pass.

    .. warning: The sequence number is thread-local, and some forward functions don't create an
    associated backward Function object (instead delegating that to sub-functions further down
    the call chain). For these reasons, the correspondence of stashed sequence numbers in
    backward Function ``apply()`` ranges with `seq` numbers in forward-pass ranges is not
    guaranteed to be 1 to 1.  The sequence numbers alone may not be enough to fully disambiguate
    which forward function created which backward Function object.  You may need to make a
    judgment based on analytic knowledge of what the expected correspondence should be.
    """

    def __init__(self, enabled=True, record_shapes=False):
        self.enabled = enabled
        self.entered = False
        self.record_shapes = record_shapes

    def __enter__(self):
        if not self.enabled:
            return None
        if self.entered:
            raise RuntimeError("NVTX annotation context manager is not reentrant")
        self.entered = True
        torch.musa.synchronize()
        _enable_profiler(
            ProfilerConfig(
                ProfilerState.NVTX,
                self.record_shapes,
                False,
                False,
                False,
                False,
                _ExperimentalConfig(),
            ),
            set(),
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.enabled:
            return None
        torch.musa.synchronize()
        _disable_profiler()
        return False


def load_nvprof(path):
    """Opens an nvprof trace file and parses autograd annotations.

    Args:
        path (str): path to nvprof trace
    """
    return EventList(parse_nvprof_trace(path))


class EnforceUnique:
    """Raises an error if a key is seen more than once."""

    def __init__(self):
        self.seen = set()

    def see(self, *key):
        if key in self.seen:
            raise RuntimeError("duplicate key: " + str(key))
        self.seen.add(key)


def parse_nvprof_trace(path):
    """Parse nvprof trace."""
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Parse strings table
    strings = {}
    for r in conn.execute("SELECT _id_ as id, value FROM StringTable"):
        strings[r["id"]] = torch._C._demangle(r["value"])

    # First, find all functions and create FunctionEvents for them
    marker_query = """
    SELECT
        start.id AS marker_id, start.name, start.timestamp AS start_time, end.timestamp AS end_time
    FROM
        CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS end
        ON start.id = end.id
    WHERE
        start.name != 0 AND end.name = 0
    """
    functions = []
    functions_map = {}
    unique = EnforceUnique()
    for row in conn.execute(marker_query):
        unique.see(row["marker_id"])
        evt = FunctionEvent(
            id=row["marker_id"],
            node_id=0,  # missing a node_id when calling FunctionEvent. This is just to ensure
            # that pytorch doesn't crash when creating a FunctionEvent() object
            name=strings[row["name"]],
            start_us=row["start_time"],
            end_us=row["end_time"],
            thread=0,
        )  # TODO: find in sqlite database
        functions.append(evt)
        functions_map[evt.id] = evt

    # Now, correlate all kernels with FunctionEvents
    kernel_query = """SELECT start.id AS marker_id, start.name, start.timestamp, end.timestamp,
    runtime._id_ AS runtime_id, runtime.cbid, runtime.start AS runtime_start, runtime.end AS 
    runtime_end, kernel.start AS kernel_start, kernel.end AS kernel_end, kernel.name AS 
    kernel_name FROM CUPTI_ACTIVITY_KIND_MARKER AS start INNER JOIN CUPTI_ACTIVITY_KIND_MARKER AS 
    end ON start.id = end.id INNER JOIN CUPTI_ACTIVITY_KIND_RUNTIME as runtime ON (
    start.timestamp < runtime.start AND runtime.end < end.timestamp) INNER JOIN 
    CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL AS kernel ON kernel.correlationId = 
    runtime.correlationId """
    unique = EnforceUnique()
    for row in conn.execute(kernel_query):
        unique.see(row["marker_id"], row["runtime_id"])
        # 211 is musaKernelLaunch for musa >= 9.2
        assert row["cbid"] == 211
        evt = functions_map[row["marker_id"]]
        evt.append_kernel(
            row["kernel_name"], 0, row["kernel_end"] - row["kernel_start"]
        )

    functions.sort(key=lambda evt: evt.time_range.start)
    return functions


class KinetoStepTracker:
    """Provides an abstraction for incrementing the step count globally.
    Previously, we only had one place to mark that a step() has occurred
    in the program via pytorch profiler step(). We will now add step hooks
    in the Optimizer class https://github.com/pytorch/pytorch/issues/88446

    - This could mean programs that already call profiler.step() every
      iteration can end up double incrementing step count.
    - If a model uses multiple optimizers we can also have double or more
      counting of the step.

    We fix this by adding a layer of abstraction before calling step()
    to the kineto library. The idea is to maintain steps per requester in a dict:
    ```
    {
       "ProfilerStep": 100,  # triggered by profiler step() call
       "Optimizer1Step": 100,   # Optimizer 1 or 2 are just examples, could be SGD, Adam etc
       "Optimizer2Step": 100,
    }
    ```
    To figure out the global step count just take the max of dict values (100).

    If one of the count increments the max will go up.
    ```
    {
       "ProfilerStep": 100,
       "Optimizer1Step": 101,   # Optimizer1 got incremented first say
       "Optimizer2Step": 100,
    }
    ```
    Then global step count is 101
    We only call the kineto step() function when global count increments.

    NOTE: Please do not use the KinetoStepTracker in modules beside the Optimizer
    for now. The result could be incorrect increments of the step count.
    """

    _current_step = -1
    _step_dict: Dict[str, int] = defaultdict(int)

    @classmethod
    def init_step_count(cls, requester: str):
        cls._step_dict[requester] = cls._current_step

    @classmethod
    def erase_step_count(cls, requester: str) -> bool:
        return cls._step_dict.pop(requester, None) is not None

    @classmethod
    def increment_step(cls, requester: str) -> int:
        """Increments the step count for the requester.
        Additionally if the max over all step counts has incremented then
        trigger the _kineto_step()
        returns global step count
        """
        if requester not in cls._step_dict:
            cls.init_step_count(requester)
        cls._step_dict[requester] += 1

        new_step = max(cls._step_dict.values())
        if new_step > cls._current_step:
            delta = new_step - cls._current_step
            if delta > 1:
                warn(
                    "Profiler step count has increased more than 1 - "
                    f"current_step = {cls._current_step} step dict =  {cls._step_dict}"
                )
            for _ in range(0, delta):
                _kineto_step()
            cls._current_step = new_step
        return cls._current_step

    @classmethod
    def current_step(cls) -> int:
        return cls._current_step
