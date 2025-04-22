"""Test multiprocee features."""

# pylint: disable=invalid-name, missing-docstring, unused-import, undefined-loop-variable
# pylint: disable=broad-exception-caught, unused-argument, unidiomatic-typecheck
import os
import copy
import time
import pytest
import torch.multiprocessing as mp
import torch
import torch_musa
from torch_musa import testing


MAX_WAITING_TIME_IN_SECONDS = 30


def simple_fill(queue, event):
    data = queue.get()
    data[0][:] = 4
    event.set()


def queue_get_exception(inqueue, outqueue):
    os.close(2)  # hide expected error message
    try:
        torch.zeros(5, 5).musa()
    except Exception as e:
        outqueue.put(e)
    else:
        outqueue.put("no exception")


def send_tensor(queue, event, device, dtype):
    t = torch.ones(5, 5, device=device, dtype=dtype)
    queue.put(t)
    queue.put(t)
    event.wait()


def send_and_delete_tensors(queue, event, device, dtype, count, size=5):
    for i in range(count):
        t = torch.full([size], i, device=device, dtype=dtype)
        queue.put(t)
        del t
    event.wait()


def _test_ipc_deadlock_actor(queue, iterations):
    for _ in range(iterations):
        if not queue.empty():
            queue.get()
        time.sleep(0.01)


def _test_ipc_deadlock_learner(queue, iterations):
    net = torch.nn.Conv2d(16, 16, 3).musa()
    for _ in range(iterations):
        if not queue.full():
            queue.put(copy.deepcopy(net.state_dict()))
        time.sleep(0.01)


def receive_and_send_sum(queue, out_queue, event, device, dtype, count, size=5):
    s = torch.full([size], 0, device=device, dtype=dtype)
    for _ in range(count):
        t = queue.get()
        s += t
    out_queue.put(s)
    event.wait()


def receive_and_send(queue, out_queue, event, count):
    for _ in range(count):
        t = queue.get()
        out_queue.put(t.clone())
    event.wait()


def sum_tensors(inq, outq):
    with torch.musa.device(1):
        tensors = inq.get()
        for tensor in tensors:
            outq.put(
                (
                    tensor.sum().item(),
                    tensor.get_device(),
                    tensor.numel(),
                    tensor.storage().size(),
                )
            )


# Multiply by two in a separate stream
def multiply_two(queue, ready, done):
    ready.set()
    with torch.musa.stream(torch.musa.Stream()):
        musa_event, tensor = queue.get()
        musa_event.wait()
        tensor.mul_(3)
        musa_event.record()
        done.set()
        del musa_event


def autograd_sharing(queue, ready, master_modified, device, is_parameter):
    var = queue.get()
    ready.set()
    master_modified.wait()

    expected_var = torch.arange(1.0, 26, device=device).view(5, 5)
    expected_var[0, 0] = 1000
    is_ok = var.data.equal(expected_var)
    var.data[:] = torch.ones(5, 5, device=device)

    is_ok &= var.grad is None
    is_ok &= not var._backward_hooks
    if is_parameter:
        is_ok &= type(var) == Parameter
    else:
        is_ok &= type(var) == torch.Tensor
    var._grad = torch.ones(5, 5, device=device)

    queue.put(is_ok)


def mixed_type_producer(queue, event):
    for _ in range(10):
        float_tensor = torch.ones(2, 2).float().musa()
        byte_tensor = torch.zeros(2, 2).byte().musa()

        queue.put(float_tensor)
        queue.put(byte_tensor)
        event.wait()
        event.clear()


class leak_checker:
    def __init__(self, test_case):
        self.checked_pids = [os.getpid()]
        self.test_case = test_case

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if torch.musa.is_available():
            torch.musa.ipc_collect()
        return False

    def check_pid(self, pid):
        self.checked_pids.append(pid)


class TestMultiprocessing:
    def _test_sharing(self, ctx=mp, device="cpu", dtype=torch.float, repeat=1):
        def test_fill():
            x = torch.zeros(5, 5).to(device, dtype)
            q = ctx.Queue()
            e = ctx.Event()

            data = [x, x[:, 1]]
            q.put(data)

            p = ctx.Process(target=simple_fill, args=(q, e))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            total_waiting_time = 0
            waiting_time = 0.5
            is_set = False
            # Once the child process is done, it will set the event to notify the
            # parent accordingly
            while total_waiting_time <= MAX_WAITING_TIME_IN_SECONDS and not is_set:
                time.sleep(waiting_time)
                total_waiting_time += waiting_time
                is_set = e.is_set()

            assert is_set
            assert data[0].eq(4).all()
            assert data[1].eq(4).all()

            p.join(100)
            assert not p.is_alive()

        def test_receive():
            q = ctx.Queue()
            e = ctx.Event()

            p = ctx.Process(target=send_tensor, args=(q, e, device, dtype))
            p.daemon = True
            lc.check_pid(p.pid)
            p.start()

            t1 = q.get()
            t2 = q.get()
            assert t1.eq(1).all()
            s1 = t1.untyped_storage()
            s2 = t2.untyped_storage()
            assert (t1 == t2).all()
            assert type(s1) is type(s2)
            assert s1.data_ptr() == s2.data_ptr()

            # We need to delete this tensors to allow producer (child process)
            # collect them properly
            del t1, t2

            # Mark the event as done and join the process
            e.set()
            p.join(100)
            assert not p.is_alive()

        with leak_checker(self) as lc:
            for _ in range(repeat):
                test_fill()
                test_receive()

    @pytest.mark.skip("TODO: failed on M3D dev4.0.0")
    @testing.skip_if_musa_unavailable
    def test_simple(self):
        self._test_sharing(mp.get_context("spawn"), "musa", torch.float)

    @testing.skip_if_musa_unavailable
    def test_bad_call(self):
        # Initialize MUSA
        t = torch.zeros(5, 5).musa().cpu()
        inq = mp.Queue()
        outq = mp.Queue()
        p = mp.Process(target=queue_get_exception, args=(inq, outq))
        p.start()
        inq.put(t)
        p.join(10)
        # This test might report BrokenPipeError in pytest, but the test is always valid.
        with pytest.raises(
            RuntimeError, match="Cannot re-initialize MUSA in forked subprocess."
        ):
            raise outq.get()

    @testing.skip_if_musa_unavailable
    def test_memory_allocation(self):
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        e = ctx.Event()
        p = ctx.Process(
            target=send_and_delete_tensors, args=(q, e, "musa", torch.int, 5)
        )
        p.start()
        t = []
        for _ in range(5):
            t.append(q.get())
        assert (t[0] == torch.full([5], 0, device="musa", dtype=torch.int32)).all()
        del t
        e.set()
        p.join(1)

    @testing.skip_if_musa_unavailable
    def test_ipc_deadlock(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue(1)
        processes = [
            ctx.Process(target=_test_ipc_deadlock_actor, args=(queue, 100)),
            ctx.Process(target=_test_ipc_deadlock_learner, args=(queue, 100)),
        ]

        for p in processes:
            p.start()

        for p in processes:
            p.join(10)

        for p in processes:
            assert not p.is_alive()

    @pytest.mark.skip(reason="May stuck")
    def test_send_many(self, name=None, size=5, count=1000):
        """
        MUSA have the unofficial limit on the number of ipc handle,
        so we only test 1000 tensor.
        """
        ctx = mp.get_context("spawn")
        q1 = ctx.Queue()
        q2 = ctx.Queue()
        q3 = ctx.Queue()
        e1 = ctx.Event()
        e2 = ctx.Event()
        e3 = ctx.Event()
        p1 = ctx.Process(
            target=send_and_delete_tensors,
            args=(q1, e1, "musa", torch.long, count, size),
        )
        p2 = ctx.Process(target=receive_and_send, args=(q1, q2, e2, count))
        p3 = ctx.Process(
            target=receive_and_send_sum,
            args=(q2, q3, e3, "musa", torch.long, count, size),
        )
        p1.start()
        p2.start()
        p3.start()
        result = q3.get()
        assert result[0].item() == int(count * (count - 1) / 2)
        del result
        e1.set()
        e2.set()
        e3.set()
        p1.join(1)
        p2.join(1)
        p3.join(1)

    @testing.skip_if_not_multiple_musa_device
    def test_small_tensors(self):
        # Check multiple small tensors which will likely use the same
        # underlying cached allocation
        ctx = mp.get_context("spawn")
        tensors = []
        for i in range(5):
            device = i % 2
            tensors += [torch.arange(i * 5.0, (i + 1) * 5).musa(device)]

        inq = ctx.Queue()
        outq = ctx.Queue()
        inq.put(tensors)
        p = ctx.Process(target=sum_tensors, args=(inq, outq))
        p.start()

        results = []
        for _ in range(5):
            results.append(outq.get())
        p.join()

        for i, _tensor in enumerate(tensors):
            v, device, tensor_size, storage_size = results[i]
            assert v == torch.arange(i * 5.0, (i + 1) * 5).sum().item()
            assert device == i % 2
            assert tensor_size == 5
            assert storage_size == 5

        del _tensor
        del tensors

    @pytest.mark.skipif(
        True, reason="musa is inconsistent with cuda behavior, JIRA SW-36868 for this."
    )
    def test_event(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        ready = ctx.Event()
        done = ctx.Event()
        p = ctx.Process(target=multiply_two, args=(queue, ready, done))
        p.start()

        ready.wait()
        with torch.musa.stream(torch.musa.Stream()):
            tensor = torch.musa.FloatTensor([1, 1, 1, 1])
            # Use a sleep kernel to test events. Without the event, the
            # multiply happens before the add.
            event = torch.musa.Event(interprocess=True)
            tensor.add_(1)
            event.record()
            event.synchronize()
            queue.put((event, tensor))
            done.wait()  # must wait until subprocess records event
            event.synchronize()
            assert tensor.tolist() == [6, 6, 6, 6]
        p.join()

    @staticmethod
    def _test_event_multiprocess_child(event, p2c, c2p):
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        event.synchronize()
        c2p.put(1)  # notify parent synchronization is done

    @pytest.mark.skipif(
        True, reason="musa is inconsistent with cuda behavior, JIRA SW-36868 for this."
    )
    def test_event_multiprocess(self):
        event = torch.musa.Event(enable_timing=False, interprocess=True)
        assert event.query()

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_multiprocess_child,
            args=(event, p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for until child process is ready
        torch.musa._sleep(50000000)  # spin for about 50 ms
        event.record()
        p2c.put(0)  # notify child event is recorded

        # TODO(Xiaokang Shang): time sleep not accurate, clock64 feature
        # will be available, so we skip this assert temporarily
        # assert not event.query()
        c2p.get()  # wait for synchronization in child
        assert event.query()
        p.join()

    @pytest.mark.skipif(
        True, reason="musa is inconsistent with cuda behavior, JIRA SW-36868 for this."
    )
    @testing.skip_if_not_multiple_musa_device
    def test_event_handle_multi_gpu(self):
        d0 = torch.device("musa:0")
        d1 = torch.device("musa:1")
        with torch.musa.device(d0):
            e0 = torch.musa.Event(enable_timing=False, interprocess=True)

        with torch.musa.device(d1):
            # create handle on different device from un-recorded event
            e0.ipc_handle()

        with torch.musa.device(d0):
            e1 = torch.musa.Event(enable_timing=False, interprocess=True)
            stream = torch.musa.Stream()
            torch.musa._sleep(50000000)  # spin for about 50 ms
            e1.record(stream)

        with torch.musa.device(d1):
            # create handle on different device from recorded event
            e1.ipc_handle()

    @staticmethod
    def _test_event_handle_importer_consumer(handle, p2c, c2p):
        e1 = torch.musa.Event.from_ipc_handle(0, handle)
        c2p.put(0)  # notify parent child is ready
        p2c.get()  # wait for record in parent
        e1.synchronize()
        c2p.put(1)  # notify synchronization is done in child
        p2c.get()  # wait for parent to finish before destructing child event

    @pytest.mark.skipif(
        True, reason="musa is inconsistent with cuda behavior, JIRA SW-36868 for this."
    )
    def test_event_handle_importer(self):
        e0 = torch.musa.Event(enable_timing=False, interprocess=True)
        assert e0.query()

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_importer_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()

        c2p.get()  # wait for child to become ready
        torch.musa._sleep(50000000)  # spin for about 50 ms
        e0.record()
        p2c.put(0)  # notify child event is recorded

        # TODO(Xiaokang Shang): time sleep not accurate, clock64 feature
        # will be available, so we skip this assert temporarily
        # assert not e0.query()
        c2p.get()  # wait for synchronization in child
        assert e0.query()
        p2c.put(1)  # notify child that parent is done
        p.join()

    @staticmethod
    def _test_event_handle_exporter_consumer(handle, p2c, c2p):
        stream = torch.musa.Stream()
        with torch.musa.stream(stream):
            e1 = torch.musa.Event.from_ipc_handle(torch.musa.current_device(), handle)
            torch.musa._sleep(50000000)  # spin for about 50 ms
            e1.record()
            c2p.put(0)
            # wait for parent process finished synchronization before
            # destructing e1
            p2c.get()

    @pytest.mark.skipif(
        True, reason="musa is inconsistent with cuda behavior, JIRA SW-36868 for this."
    )
    def test_event_handle_exporter(self):
        e0 = torch.musa.Event(enable_timing=False, interprocess=True)

        ctx = mp.get_context("spawn")
        p2c = ctx.SimpleQueue()
        c2p = ctx.SimpleQueue()
        p = ctx.Process(
            target=TestMultiprocessing._test_event_handle_exporter_consumer,
            args=(e0.ipc_handle(), p2c, c2p),
        )
        p.start()
        # wait for event in child process is recorded
        c2p.get()

        # TODO(Xiaokang Shang): time sleep not accurate, clock64 feature
        # will be available, so we skip this assert temporarily
        # assert not e0.query()
        e0.synchronize()
        assert e0.query()
        p2c.put(0)
        p.join()

    def _test_empty_tensor_sharing(self, dtype, device):
        q = mp.Queue()
        empty = torch.tensor([], dtype=dtype, device=device)
        q.put(empty)
        out = q.get(timeout=1)
        assert len(out) == 0 and len(out) == len(empty)

    def test_empty_tensor_sharing(self):
        self._test_empty_tensor_sharing(torch.float32, torch.device("musa"))
        self._test_empty_tensor_sharing(torch.int64, torch.device("musa"))

    def _test_autograd_sharing(self, var, ctx=mp, is_parameter=False):
        device = "musa"

        ready = ctx.Event()
        master_modified = ctx.Event()
        queue = ctx.Queue()
        p = ctx.Process(
            target=autograd_sharing,
            args=(queue, ready, master_modified, device, is_parameter),
        )
        p.daemon = True
        p.start()

        # This would cause an error if we tried to serialize the hooks,
        # because it's a closure and pickle doesn't support closures.
        @torch.utils.hooks.unserializable_hook
        def hook(*unused):
            pass

        if var.requires_grad:
            var.register_hook(hook)
        var._grad = torch.zeros(5, 5, device=device)
        queue.put(var)

        ready.wait()
        var.data[0, 0] = 1000
        var.grad.data[:] = torch.ones(5, 5, device=device) * 4
        master_modified.set()

        worker_ok = queue.get()
        assert worker_ok

        assert (var.data == torch.ones(5, 5, device=device)).all()
        assert (var.grad.data == (torch.ones(5, 5, device=device) * 4)).all()
        p.join(100)
        assert not p.is_alive()

    def test_variable_sharing(self):
        for requires_grad in [False]:
            var = (
                torch.arange(1.0, 26, device="musa")
                .view(5, 5)
                .requires_grad_(requires_grad)
            )
            self._test_autograd_sharing(var, mp.get_context("spawn"))

    def _test_mixed_types_sharing(self, ctx=mp):
        all_ones = torch.ones(2, 2).float()
        all_zeros = torch.zeros(2, 2).byte()
        queue = ctx.Queue()
        event = ctx.Event()

        p = ctx.Process(target=mixed_type_producer, args=(queue, event))

        p.start()

        for _ in range(10):
            float_tensor = queue.get()
            byte_tensor = queue.get()
            assert (float_tensor == all_ones.musa()).all()
            assert (byte_tensor == all_zeros.musa()).all()
            del float_tensor, byte_tensor
            event.set()

        time.sleep(5)
        p.join()

    def test_mixed_types_sharing(self):
        self._test_mixed_types_sharing(mp.get_context("spawn"))

    def test_is_shared_musa(self):
        t = torch.randn(5, 5).musa()
        assert t.is_shared()
