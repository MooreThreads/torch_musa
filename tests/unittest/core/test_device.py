"""Test device features."""
# pylint: disable=invalid-name, comparison-with-itself, unused-variable, unused-import, C0415, C0121, C2801, W0611
import queue
import threading
import torch
import pytest
import torch_musa
from torch_musa import testing
from torch_musa.testing import get_cycles_per_ms

FIFTY_MIL_CYCLES = 50000000


@testing.skip_if_not_multiple_musa_device
def test_musa_set_device():
    """Test cases include set_device and device context"""
    x = torch.randn(5, 5)
    with torch.musa.device(1):
        assert x.to("musa").get_device() == 1
        assert torch.musa.current_device() == 1
        torch.musa.set_device(0)
        assert x.to("musa").get_device() == 0
        assert torch.musa.current_device() == 0
        with torch.musa.device(1):
            assert x.to("musa").get_device() == 1
            assert torch.musa.current_device() == 1
        assert x.to("musa").get_device() == 0
        assert torch.musa.current_device() == 0
        torch.musa.set_device(1)
    assert x.to("musa").get_device() == 0
    assert torch.musa.current_device() == 0


@testing.skip_if_not_multiple_musa_device
def test_get_musa_devcie_index():
    """Test exception case about torch.musa.device(xxx)"""
    with torch.musa.device("musa:1"):
        assert torch.musa.current_device() == 1
        with torch.musa.device("musa:2"):
            assert torch.musa.current_device() == 2
        assert torch.musa.current_device() == 1
    assert torch.musa.current_device() == 0

    with pytest.raises(ValueError, match="Expected a musa device, but got: cuda"):
        torch.musa.device("cuda")

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch.musa.device("cpu")


@testing.skip_if_not_multiple_musa_device
def test_can_device_access_peer():
    """Test canAccessPeer and DeviceProperties"""
    assert torch.musa.can_device_access_peer(0, 0) is False

    with pytest.raises(AssertionError, match="Invalid peer device id"):
        torch.musa.can_device_access_peer(0, 100)

    with pytest.raises(AssertionError, match="Invalid device id"):
        torch.musa.can_device_access_peer(-1, 1)


@testing.skip_if_not_multiple_musa_device
def test_device_of():
    """Test device of context"""
    x = torch.randn(5, 5).to("musa:1")
    assert torch.musa.current_device() == 0
    with torch.musa.device_of(x):
        assert torch.musa.current_device() == 1
        y = torch.randn(5, 5).to("musa:0")
        with torch.musa.device_of(y):
            assert torch.musa.current_device() == 0
        assert torch.musa.current_device() == 1
    assert torch.musa.current_device() == 0


def test_get_musa_device_index():
    "Test get device index"
    from torch_musa.core._utils import _get_musa_device_index

    with pytest.raises(RuntimeError, match="Invalid device string: 'musa0'"):
        _get_musa_device_index("musa0", optional=True)

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        _get_musa_device_index(torch.device("cpu"), optional=True)


@testing.skip_if_musa_unavailable
def test_synchronize():
    """Test torch.musa synchronize feature"""
    torch.musa.synchronize()
    torch.musa.synchronize("musa")
    torch.musa.synchronize("musa:0")
    torch.musa.synchronize(0)
    torch.musa.synchronize(torch.device("musa:0"))

    if torch.musa.device_count() > 1:
        torch.musa.synchronize("musa:1")
        torch.musa.synchronize(1)
        torch.musa.synchronize(torch.device("musa:1"))

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch.musa.synchronize(torch.device("cpu"))

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch.musa.synchronize("cpu")


@testing.skip_if_musa_unavailable
def test_musa_get_device_name():
    """Testing the behaviour with None as an argument"""
    current_device = torch.musa.current_device()
    current_device_name = torch.musa.get_device_name(current_device)
    device_name_None = torch.musa.get_device_name(None)
    assert current_device_name == device_name_None

    # Testing the behaviour for No argument
    device_name_no_argument = torch.musa.get_device_name()
    assert current_device_name == device_name_no_argument


@testing.skip_if_musa_unavailable
def test_musa_get_device_capability():
    """Testing the behaviour with None as an argument"""
    current_device = torch.musa.current_device()
    current_device_capability = torch.musa.get_device_capability(current_device)
    device_capability_None = torch.musa.get_device_capability(None)
    assert current_device_capability == device_capability_None

    # Testing the behaviour for No argument
    device_capability_no_argument = torch.musa.get_device_capability()
    assert current_device_capability, device_capability_no_argument


@testing.skip_if_not_multiple_musa_device
def test_copy_device():
    "Testing copy on multi cards"
    x = torch.randn(5, 5).to("musa")
    with torch.musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y
        z = y.to("musa:0")
        assert z.get_device() == 0
        assert z.to("musa:0") is z

    x = torch.randn(5, 5)
    with torch.musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y
        z = y.to("musa:0")
        assert z.get_device() == 0
        assert z.to("musa:0") is z


def _test_copy_sync_current_stream(x, y):
    x_plus_one = x + 1
    s0 = torch.musa.Stream(device=x.device)
    s1 = torch.musa.Stream(device=x.device)
    s2 = torch.musa.Stream(device=x.device)
    s3 = torch.musa.Stream(device=x.device)

    # same dst stream different src streams
    with torch.musa.stream(s0):
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        with torch.musa.stream(s1):
            y.copy_(x_plus_one)

    with torch.musa.stream(s2), torch.musa.stream(s1):
        y.copy_(x)

    s1.synchronize()

    with torch.musa.stream(s1):
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        with torch.musa.stream(s0):
            y.copy_(x_plus_one)

    with torch.musa.stream(s3), torch.musa.stream(s0):
        y.copy_(x)

    s0.synchronize()

    assert torch.all(x == y)


@testing.skip_if_not_multiple_musa_device
def test_copy_streams():
    """Testing copy with diff streams"""
    d0 = torch.device("musa:0")
    x0 = torch.zeros(5, 5, device=d0)

    d1 = torch.device("musa:1")
    x1 = torch.zeros(5, 5, device=d1)
    _test_copy_sync_current_stream(x0, x1)

    x2 = torch.zeros(5, 5, device=d0)
    _test_copy_sync_current_stream(x0, x2)


@testing.skip_if_musa_unavailable
def test_to_cpu_blocking_by_default():
    """Testing block copy to cpu"""
    src = torch.randn(1000000, device="musa")
    torch.musa.synchronize()
    torch.musa._sleep(int(100 * get_cycles_per_ms()))
    dst = src.to(device="cpu")
    assert torch.musa.current_stream().query() is True
    assert torch.all(src.cpu() == dst)
    assert dst.is_pinned() is False


@testing.skip_if_not_multiple_musa_device
def test_current_stream():
    "Testing NULL stream"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    s0 = torch.musa.current_stream()
    s1 = torch.musa.current_stream(device=1)
    s2 = torch.musa.current_stream(device=0)

    assert d0 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert s0 == s2

    with torch.musa.device(d1):
        s0 = torch.musa.current_stream()
        s1 = torch.musa.current_stream(1)
        s2 = torch.musa.current_stream(d0)

    assert d1 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert s0 == s1

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch.musa.current_stream(torch.device("cpu"))


@testing.skip_if_not_multiple_musa_device
def test_default_stream():
    "Testing No-NULL stream from stream pool"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch.musa.device(d0):
        s0 = torch.musa.default_stream()

    with torch.musa.device(d1):
        s1 = torch.musa.default_stream()

    s2 = torch.musa.default_stream(device=0)
    s3 = torch.musa.default_stream(d1)

    assert d0 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert d1 == s3.device
    assert s0 == s2
    assert s1 == s3
    with torch.musa.device(d0):
        assert torch.musa.current_stream() == s0

    with torch.musa.device(d1):
        assert torch.musa.current_stream() == s1

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch.musa.default_stream(torch.device("cpu"))


@testing.skip_if_not_multiple_musa_device
def test_streams():
    "Testing stream base methods"
    default_stream = torch.musa.current_stream()
    user_stream = torch.musa.Stream()
    assert torch.musa.current_stream() == default_stream
    assert default_stream != user_stream
    assert default_stream.musa_stream == 0
    assert user_stream.musa_stream != 0
    with torch.musa.stream(user_stream):
        assert torch.musa.current_stream() == user_stream
    assert user_stream.query() is True
    # tensor1 = torch.Tensor(5) # TODO(mt-ai): .pin_memory is unsupported
    # tensor2 = tensor1.to(device="musa", non_blocking=True) + 1
    # default_stream.synchronize()
    # assert default_stream.query() is True


@testing.skip_if_not_multiple_musa_device
def test_stream_context():
    "Testing stream context"
    s0 = torch.musa.current_stream()
    s1 = torch.musa.Stream(device=1)
    s2 = torch.musa.Stream(device=0)

    with torch.musa.device(s1.device):
        prev_stream_on_musa1 = torch.musa.current_stream()

    assert torch.musa.current_stream() == s0
    assert 0 == torch.musa.current_device()
    with torch.musa.stream(s1):
        assert torch.musa.current_stream() == s1
        assert 1 == torch.musa.current_device()
        with torch.musa.stream(s2):
            assert torch.musa.current_stream() == s2
            assert 0 == torch.musa.current_device()
            with torch.musa.stream(s0):
                assert torch.musa.current_stream() == s0
                assert 0 == torch.musa.current_device()
            assert torch.musa.current_stream() == s2
            assert 0 == torch.musa.current_device()
        assert torch.musa.current_stream() == s1
        assert 1 == torch.musa.current_device()

    with torch.musa.device(s1.device):
        assert prev_stream_on_musa1 == torch.musa.current_stream()

    assert torch.musa.current_stream() == s0
    assert 0 == torch.musa.current_device()


@testing.skip_if_not_multiple_musa_device
def test_streams_multi_gpu():
    "Testing stream across multi cards"
    default_stream = torch.musa.current_stream()
    assert default_stream.device, torch.device("musa:0")
    stream = torch.musa.Stream(device=1)
    assert stream.device == torch.device("musa:1")
    with torch.musa.device(1):
        assert torch.musa.current_stream().device == torch.device("musa:1")
        assert torch.musa.current_stream() != default_stream


@testing.skip_if_not_multiple_musa_device
def test_streams_multi_gpu_query():
    "Testing streams query"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")
    torch.musa.synchronize(d0)
    torch.musa.synchronize(d1)

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()

    with torch.musa.device(d1):
        s1 = torch.musa.current_stream()
        torch.musa._sleep(FIFTY_MIL_CYCLES)

    assert s0.query() is True
    assert s1.query() is False

    with torch.musa.device(d0):
        assert s0.query() is True
        assert s1.query() is False

    with torch.musa.device(d1):
        assert s0.query() is True
        assert s1.query() is False

    # deliberately using a different device
    with torch.musa.device(d0):
        s1.synchronize()

    assert s0.query() is True
    assert s1.query() is True

    with torch.musa.device(d0):
        assert s0.query() is True
        assert s1.query() is True

    with torch.musa.device(d1):
        assert s0.query() is True
        assert s1.query() is True


@testing.skip_if_not_multiple_musa_device
def test_streams_multi_gpu_eq():
    "Testing streams equal"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        s1 = torch.musa.current_stream()

    with torch.musa.device(d1):
        s2 = torch.musa.current_stream()
        s3 = torch.musa.current_stream()

    assert s0 == s0
    assert s0 == s1
    assert s2 == s2
    assert s2 == s3
    assert s0 != s2
    assert s1 != s3

    assert s0.device == s1.device
    assert s0.musa_stream == s1.musa_stream
    assert s2.device == s3.device
    assert s2.musa_stream == s3.musa_stream
    assert s0.device != s3.device

    assert hash(s0) == hash(s1)
    assert hash(s2) == hash(s3)
    assert hash(s0) != hash(s3)


@testing.skip_if_not_multiple_musa_device
def test_streams_priority():
    "Testing streams priority"
    low, high = torch.musa.Stream.priority_range()
    s0 = torch.musa.Stream(device=0, priority=low)

    assert low == s0.priority
    assert torch.device("musa:0") == s0.device

    s1 = torch.musa.Stream(device=1, priority=high)

    assert high == s1.priority
    assert torch.device("musa:1") == s1.device


# TODO(mt-ai): Need import musart
@pytest.mark.skip(reason="Waiting musart import")
def test_external_streams():
    "Testing external streams"


@testing.skip_if_not_multiple_musa_device
def test_malloc_multi_device():
    """Test multiple device allocator"""
    curr_device = torch.musa.current_device()
    device_num = torch.musa.device_count()
    for device_i in range(device_num):
        torch.musa.set_device(device_i)
        tensor = torch.rand(5, 5).to("musa")
        assert tensor.device == torch.device("musa:" + str(device_i))
    torch.musa.set_device(curr_device)  # reset device


@testing.skip_if_not_multiple_musa_device
def test_stream_event_device():
    """Test stream event device"""
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")
    e0 = torch.musa.Event()

    assert e0.device is None

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        s0.record_event(e0)

    with torch.musa.device(d1):
        s1 = torch.musa.Stream()
        e1 = s1.record_event()

    assert s0.device == torch.device("musa:0")
    assert e0.device == torch.device("musa:0")
    assert s1.device == torch.device("musa:1")
    assert e1.device == torch.device("musa:1")


def test_stream_event_repr():
    """Test stream event repr"""
    s = torch.musa.current_stream()
    assert ("torch_musa.Stream" in s.__repr__()) is True
    e = torch.musa.Event()
    assert ("torch_musa.Event" in e.__repr__()) is True
    s.record_event(e)
    assert ("torch_musa.Event" in e.__repr__()) is True


@testing.skip_if_not_multiple_musa_device
def test_tensor_device():
    assert torch.FloatTensor(1).to("musa").get_device() == 0
    assert torch.FloatTensor(1).to("musa:1").get_device() == 1
    with torch.musa.device(1):
        assert torch.FloatTensor(1).to("musa").get_device() == 1
        assert torch.FloatTensor(1).to("musa:0").get_device() == 0
        assert torch.FloatTensor(1).to("musa").get_device() == 1


def test_events():
    """Testing event base api"""
    stream = torch.musa.current_stream()
    event = torch.musa.Event(enable_timing=True)
    assert event.query() is True
    start_event = torch.musa.Event(enable_timing=True)
    stream.record_event(start_event)
    torch.musa._sleep(FIFTY_MIL_CYCLES)
    stream.record_event(event)
    assert event.query() is False
    event.synchronize()
    assert event.query() is True
    assert start_event.elapsed_time(event) > 0


def _stream_synchronize(spin_time_cycles):
    s = torch.musa.current_stream()
    e_tik = torch.musa.Event(enable_timing=True)
    e_tok = torch.musa.Event(enable_timing=True)

    e_tik.record(s)
    torch.musa._sleep(spin_time_cycles)
    e_tok.record(s)
    s.synchronize()

    assert s.query() is True

    return e_tik.elapsed_time(e_tok)


def _event_synchronize(spin_time_cycles):
    s = torch.musa.current_stream()
    e_tik = torch.musa.Event(enable_timing=True)
    e_tok = torch.musa.Event(enable_timing=True)

    e_tik.record(s)
    torch.musa._sleep(spin_time_cycles)
    s.record_event(e_tok)
    e_tok.synchronize()

    assert s.query() is True

    # not necessary to check e_tik and e_tok, as elapsed_time would throw
    # exception if otherwise.
    return e_tik.elapsed_time(e_tok)


def _event_wait(spin_time_cycles):
    s0 = torch.musa.current_stream()
    s1 = torch.musa.Stream()
    e_tik = torch.musa.Event(blocking=True, enable_timing=True)
    e_tok = torch.musa.Event(blocking=True, enable_timing=True)

    e_tik.record(s0)
    torch.musa._sleep(spin_time_cycles - 10)
    e_sync = torch.musa.Event(blocking=True)
    e_sync.record()
    e_sync.wait(s1)
    with torch.musa.stream(s1):
        torch.musa._sleep(FIFTY_MIL_CYCLES)
    s1.synchronize()
    e_tok.record()
    e_tok.synchronize()

    assert s0.query() is True
    assert s1.query() is True
    assert e_sync.query() is True

    # not necessary to check e_tik and e_tok, as elapsed_time would throw
    # exception if otherwise.
    return e_tik.elapsed_time(e_tok)


def _test_stream_event_nogil(sync_func, p2c, c2p):
    with torch.musa.device("musa:1"):
        c2p.put(0)
        p2c.get()
        c2p.put(sync_func(FIFTY_MIL_CYCLES))


@testing.skip_if_not_multiple_musa_device
def test_stream_event_nogil():
    """Testing stream and event with nogil"""
    for sync_func in [_stream_synchronize, _event_synchronize, _event_wait]:
        p2c = queue.Queue()
        c2p = queue.Queue()
        e_tik = torch.musa.Event(enable_timing=True)
        e_tok = torch.musa.Event(enable_timing=True)

        t = threading.Thread(
            target=_test_stream_event_nogil, args=(sync_func, p2c, c2p)
        )
        t.daemon = True
        t.start()

        c2p.get()
        with torch.musa.device("musa:0"):
            e_tik.record()
            p2c.put(0)
            parent_time = sync_func(FIFTY_MIL_CYCLES)
            child_time = c2p.get()
            e_tok.record()
            e_tok.synchronize()
            total_time = e_tik.elapsed_time(e_tok)

        # Without GIL, synchronizations in parent and child threads can
        # overlap. The total execution time should be a little bit longer
        # than spinning fifty million cycles and much shorter than twice of
        # that. However, testing absolute execution time is not reliable as
        # it may vary on different hardware in different environments.
        # Therefore, this test uses relative comparisons, checking if the
        # sum of parent and child threads execution time is greater than the
        # real execution time by least 40%.
        # assert parent_time + child_time > total_time * 1.4


@testing.skip_if_not_multiple_musa_device
def test_events_wait():
    """Testing evetns wait"""
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")
    torch.musa.synchronize(d0)
    torch.musa.synchronize(d1)

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        e0 = torch.musa.Event()
        s0.record_event(e0)

    with torch.musa.device(d1):
        s1 = torch.musa.current_stream()

    assert s0.query() is False
    assert s1.query() is True

    s1.wait_event(e0)
    s1.synchronize()

    assert e0.query() is True
    assert s0.query() is True
    assert s1.query() is True


@testing.skip_if_not_multiple_musa_device
def test_events_multi_gpu_query():
    """Testing event query on multi-gpu env"""
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        e0 = s0.record_event()
        s0.synchronize()

    with torch.musa.device(d1):
        s1 = torch.musa.current_stream()
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        e1 = s1.record_event()

    assert e0.query() is True
    assert e1.query() is False

    with torch.musa.device(d0):
        assert e0.query() is True
        assert e1.query() is False

    with torch.musa.device(d1):
        assert e0.query() is True
        assert e1.query() is False

    # deliberately using a different device
    with torch.musa.device(d0):
        e1.synchronize()

    assert e0.query() is True
    assert e1.query() is True

    with torch.musa.device(d0):
        assert e0.query() is True
        assert e1.query() is True

    with torch.musa.device(d1):
        assert e0.query() is True
        assert e1.query() is True


@testing.skip_if_not_multiple_musa_device
def test_events_multi_gpu_elapsed_time():
    """Testing events elapsed time on multi-gpu env"""
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        e0 = torch.musa.Event(enable_timing=True)
        torch.musa._sleep(10)
        s0.record_event(e0)

    with torch.musa.device(d1):
        s1 = torch.musa.current_stream()
        e1 = torch.musa.Event(enable_timing=True)
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        s1.record_event(e1)

    e0.synchronize()
    e1.synchronize()
    with torch.musa.device(d0):
        with pytest.raises(RuntimeError):
            assert Greater(e0.elapsed_time(e1), 0)

    with torch.musa.device(d1):
        with pytest.raises(RuntimeError):
            assert e0.elapsed_time(e1) > 0

    with torch.musa.device(d0):
        s0 = torch.musa.current_stream()
        e2 = torch.musa.Event(enable_timing=True)
        torch.musa._sleep(FIFTY_MIL_CYCLES)
        s0.record_event(e2)
        s0.synchronize()

    assert e0.elapsed_time(e2) > 0

    # deliberately calling from a different device
    with torch.musa.device(d1):
        assert e0.elapsed_time(e2) > 0


# TODO(MT-AI): Support pin_memory and tensor record_stream op.
@pytest.mark.skipif(True, reason="pin_memory is unsupport")
def test_record_stream():
    """Testing tensor record stream"""
    cycles_per_ms = get_cycles_per_ms()

    t = torch.FloatTensor([1, 2, 3, 4]).pin_memory()
    result = torch.musa.FloatTensor(t.size())
    stream = torch.musa.Stream()
    ptr = [None]

    # Performs the CPU->GPU copy in a background stream
    def perform_copy():
        with torch.musa.stream(stream):
            tmp = t.to(device="musa", non_blocking=True)
            ptr[0] = tmp.data_ptr()
        torch.musa.current_stream().wait_stream(stream)
        tmp.record_stream(torch.musa.current_stream())
        torch.musa._sleep(int(50 * cycles_per_ms))  # delay the copy
        result.copy_(tmp)

    perform_copy()
    with torch.musa.stream(stream):
        tmp2 = torch.musa.FloatTensor(t.size())
        tmp2.zero_()
        assert tmp2.data_ptr() != ptr[0], "allocation re-used to soon"

    assert result.tolist() == [1, 2, 3, 4]

    if not TEST_MUSAMALLOCASYNC:
        # In the native allocator, we expect "tmp"'s side-stream-tagged block will be reused
        # in that side stream after result.copy_(tmp) in the main stream finishes.
        torch.musa.current_stream().synchronize()
        with torch.musa.stream(stream):
            tmp3 = torch.musa.FloatTensor(t.size())
            assert tmp3.data_ptr() == ptr[0], "allocation not re-used"
