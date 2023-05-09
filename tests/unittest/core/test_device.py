"""Test device features."""
# pylint: disable=invalid-name, comparison-with-itself, unused-variable, C0415, C0121
import torch
import pytest
import torch_musa

TEST_MUSA = torch_musa.is_available()
TEST_MULTIGPU = TEST_MUSA and torch_musa.device_count() >= 2
FIFTY_MIL_CYCLES = 50000000


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_musa_set_device():
    """Test cases include set_device and device context"""
    x = torch.randn(5, 5)
    with torch_musa.device(1):
        assert x.to("musa").get_device() == 1
        assert torch_musa.current_device() == 1
        torch_musa.set_device(0)
        assert x.to("musa").get_device() == 0
        assert torch_musa.current_device() == 0
        with torch_musa.device(1):
            assert x.to("musa").get_device() == 1
            assert torch_musa.current_device() == 1
        assert x.to("musa").get_device() == 0
        assert torch_musa.current_device() == 0
        torch_musa.set_device(1)
    assert x.to("musa").get_device() == 0
    assert torch_musa.current_device() == 0


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected no mtGPU")
def test_get_musa_devcie_index():
    """Test exception case about torch_musa.device(xxx)"""
    with torch_musa.device("musa:1"):
        assert torch_musa.current_device() == 1
        with torch_musa.device("musa:2"):
            assert torch_musa.current_device() == 2
        assert torch_musa.current_device() == 1
    assert torch_musa.current_device() == 0

    with pytest.raises(ValueError, match="Expected a musa device, but got: cuda"):
        torch_musa.device("cuda")

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch_musa.device("cpu")


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected no mtGPU")
def test_can_device_access_peer():
    """Test canAccessPeer and DeviceProperties"""
    assert torch_musa.can_device_access_peer(0, 0) is False

    with pytest.raises(AssertionError, match="Invalid peer device id"):
        torch_musa.can_device_access_peer(0, 100)

    with pytest.raises(AssertionError, match="Invalid device id"):
        torch_musa.can_device_access_peer(-1, 1)


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_device_of():
    """Test device of context"""
    x = torch.randn(5, 5).to("musa:1")
    assert torch_musa.current_device() == 0
    with torch_musa.device_of(x):
        assert torch_musa.current_device() == 1
        y = torch.randn(5, 5).to("musa:0")
        with torch_musa.device_of(y):
            assert torch_musa.current_device() == 0
        assert torch_musa.current_device() == 1
    assert torch_musa.current_device() == 0


def test_get_musa_device_index():
    "Test get device index"
    from torch_musa.core._utils import _get_musa_device_index

    with pytest.raises(RuntimeError, match="Invalid device string: 'musa0'"):
        _get_musa_device_index("musa0", optional=True)

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        _get_musa_device_index(torch.device("cpu"), optional=True)


@pytest.mark.skipif(not TEST_MUSA, reason="detected no mtGPU")
def test_synchronize():
    """Test torch_musa synchronize feature"""
    torch_musa.synchronize()
    torch_musa.synchronize("musa")
    torch_musa.synchronize("musa:0")
    torch_musa.synchronize(0)
    torch_musa.synchronize(torch.device("musa:0"))

    if TEST_MULTIGPU:
        torch_musa.synchronize("musa:1")
        torch_musa.synchronize(1)
        torch_musa.synchronize(torch.device("musa:1"))

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch_musa.synchronize(torch.device("cpu"))

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch_musa.synchronize("cpu")


@pytest.mark.skipif(not TEST_MUSA, reason="detected no mtGPU")
def test_musa_get_device_name():
    """Testing the behaviour with None as an argument"""
    current_device = torch_musa.current_device()
    current_device_name = torch_musa.get_device_name(current_device)
    device_name_None = torch_musa.get_device_name(None)
    assert current_device_name == device_name_None

    # Testing the behaviour for No argument
    device_name_no_argument = torch_musa.get_device_name()
    assert current_device_name == device_name_no_argument


@pytest.mark.skipif(not TEST_MUSA, reason="detected no mtGPU")
def test_musa_get_device_capability():
    """Testing the behaviour with None as an argument"""
    current_device = torch_musa.current_device()
    current_device_capability = torch_musa.get_device_capability(current_device)
    device_capability_None = torch_musa.get_device_capability(None)
    assert current_device_capability == device_capability_None

    # Testing the behaviour for No argument
    device_capability_no_argument = torch_musa.get_device_capability()
    assert current_device_capability, device_capability_no_argument


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_copy_device():
    "Testing copy on multi cards"
    x = torch.randn(5, 5).to("musa")
    with torch_musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y
        z = y.to("musa:0")
        assert z.get_device() == 0
        assert z.to("musa:0") is z

    x = torch.randn(5, 5)
    with torch_musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y
        z = y.to("musa:0")
        assert z.get_device() == 0
        assert z.to("musa:0") is z


def _test_copy_sync_current_stream(x, y):
    x_plus_one = x + 1
    s0 = torch_musa.Stream(device=x.device)
    s1 = torch_musa.Stream(device=x.device)
    s2 = torch_musa.Stream(device=x.device)
    s3 = torch_musa.Stream(device=x.device)

    # same dst stream different src streams
    with torch_musa.stream(s0):
        torch_musa._sleep(FIFTY_MIL_CYCLES)
        with torch_musa.stream(s1):
            y.copy_(x_plus_one)

    with torch_musa.stream(s2), torch_musa.stream(s1):
        y.copy_(x)

    s1.synchronize()

    with torch_musa.stream(s1):
        torch_musa._sleep(FIFTY_MIL_CYCLES)
        with torch_musa.stream(s0):
            y.copy_(x_plus_one)

    with torch_musa.stream(s3), torch_musa.stream(s0):
        y.copy_(x)

    s0.synchronize()

    assert torch.all(x == y) == True


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_copy_streams():
    """Testing copy with diff streams"""
    d0 = torch.device("musa:0")
    x0 = torch.zeros(5, 5, device=d0)

    d1 = torch.device("musa:1")
    x1 = torch.zeros(5, 5, device=d1)
    _test_copy_sync_current_stream(x0, x1)

    x2 = torch.zeros(5, 5, device=d0)
    _test_copy_sync_current_stream(x0, x2)


@pytest.mark.skipif(not TEST_MUSA, reason="detected no mtGPU")
def test_to_cpu_blocking_by_default():
    """Testing block copy to cpu"""
    src = torch.randn(1000000, device="musa")
    torch_musa.synchronize()
    torch_musa._sleep(FIFTY_MIL_CYCLES)
    dst = src.to(device="cpu")
    assert torch_musa.current_stream().query() is True
    assert torch.all(src.cpu() == dst) == True
    assert dst.is_pinned() is False


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_current_stream():
    "Testing NULL stream"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    s0 = torch_musa.current_stream()
    s1 = torch_musa.current_stream(device=1)
    s2 = torch_musa.current_stream(device=0)

    assert d0 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert s0 == s2

    with torch_musa.device(d1):
        s0 = torch_musa.current_stream()
        s1 = torch_musa.current_stream(1)
        s2 = torch_musa.current_stream(d0)

    assert d1 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert s0 == s1

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch_musa.current_stream(torch.device("cpu"))


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_default_stream():
    "Testing No-NULL stream from stream pool"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch_musa.device(d0):
        s0 = torch_musa.default_stream()

    with torch_musa.device(d1):
        s1 = torch_musa.default_stream()

    s2 = torch_musa.default_stream(device=0)
    s3 = torch_musa.default_stream(d1)

    assert d0 == s0.device
    assert d1 == s1.device
    assert d0 == s2.device
    assert d1 == s3.device
    assert s0 == s2
    assert s1 == s3
    with torch_musa.device(d0):
        assert torch_musa.current_stream() == s0

    with torch_musa.device(d1):
        assert torch_musa.current_stream() == s1

    with pytest.raises(ValueError, match="Expected a musa device, but got: cpu"):
        torch_musa.default_stream(torch.device("cpu"))


@pytest.mark.skipif(not TEST_MUSA, reason="detected only one mtGPU")
def test_streams():
    "Testing stream base methods"
    default_stream = torch_musa.current_stream()
    user_stream = torch_musa.Stream()
    assert torch_musa.current_stream() == default_stream
    assert default_stream != user_stream
    assert default_stream.musa_stream == 0
    assert user_stream.musa_stream != 0
    with torch_musa.stream(user_stream):
        assert torch_musa.current_stream() == user_stream
    assert user_stream.query() is True
    # tensor1 = torch.Tensor(5) # TODO(mt-ai): .pin_memory is unsupported
    # tensor2 = tensor1.to(device="musa", non_blocking=True) + 1
    # default_stream.synchronize()
    # assert default_stream.query() is True


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_stream_context():
    "Testing stream context"
    s0 = torch_musa.current_stream()
    s1 = torch_musa.Stream(device=1)
    s2 = torch_musa.Stream(device=0)

    with torch_musa.device(s1.device):
        prev_stream_on_musa1 = torch_musa.current_stream()

    assert torch_musa.current_stream() == s0
    assert 0 == torch_musa.current_device()
    with torch_musa.stream(s1):
        assert torch_musa.current_stream() == s1
        assert 1 == torch_musa.current_device()
        with torch_musa.stream(s2):
            assert torch_musa.current_stream() == s2
            assert 0 == torch_musa.current_device()
            with torch_musa.stream(s0):
                assert torch_musa.current_stream() == s0
                assert 0 == torch_musa.current_device()
            assert torch_musa.current_stream() == s2
            assert 0 == torch_musa.current_device()
        assert torch_musa.current_stream() == s1
        assert 1 == torch_musa.current_device()

    with torch_musa.device(s1.device):
        assert prev_stream_on_musa1 == torch_musa.current_stream()

    assert torch_musa.current_stream() == s0
    assert 0 == torch_musa.current_device()


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_streams_multi_gpu():
    "Testing stream across multi cards"
    default_stream = torch_musa.current_stream()
    assert default_stream.device, torch.device("musa:0")
    stream = torch_musa.Stream(device=1)
    assert stream.device == torch.device("musa:1")
    with torch_musa.device(1):
        assert torch_musa.current_stream().device == torch.device("musa:1")
        assert torch_musa.current_stream() != default_stream


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_streams_multi_gpu_query():
    "Testing streams query"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")
    torch_musa.synchronize(d0)
    torch_musa.synchronize(d1)

    with torch_musa.device(d0):
        s0 = torch_musa.current_stream()

    with torch_musa.device(d1):
        s1 = torch_musa.current_stream()
        torch_musa._sleep(FIFTY_MIL_CYCLES)

    assert s0.query() is True
    assert s1.query() is False

    with torch_musa.device(d0):
        assert s0.query() is True
        assert s1.query() is False

    with torch_musa.device(d1):
        assert s0.query() is True
        assert s1.query() is False

    # deliberately using a different device
    with torch_musa.device(d0):
        s1.synchronize()

    assert s0.query() is True
    assert s1.query() is True

    with torch_musa.device(d0):
        assert s0.query() is True
        assert s1.query() is True

    with torch_musa.device(d1):
        assert s0.query() is True
        assert s1.query() is True


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected only one mtGPU")
def test_streams_multi_gpu_eq():
    "Testing streams equal"
    d0 = torch.device("musa:0")
    d1 = torch.device("musa:1")

    with torch_musa.device(d0):
        s0 = torch_musa.current_stream()
        s1 = torch_musa.current_stream()

    with torch_musa.device(d1):
        s2 = torch_musa.current_stream()
        s3 = torch_musa.current_stream()

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


@pytest.mark.skipif(not TEST_MULTIGPU, reason="multi-GPU not supported")
def test_streams_priority():
    "Testing streams priority"
    low, high = torch_musa.Stream.priority_range()
    s0 = torch_musa.Stream(device=0, priority=low)

    assert low == s0.priority
    assert torch.device("musa:0") == s0.device

    s1 = torch_musa.Stream(device=1, priority=high)

    assert high == s1.priority
    assert torch.device("musa:1") == s1.device


# TODO(mt-ai): Need import musart
@pytest.mark.skipif(True, reason="Waiting musart import")
def test_external_streams():
    "Testing external streams"


@pytest.mark.skipif(not TEST_MULTIGPU, reason="detected no mtGPU")
def test_malloc_multi_device():
    """Test multiple device allocator"""
    curr_device = torch_musa.current_device()
    device_num = torch_musa.device_count()
    for device_i in range(device_num):
        torch_musa.set_device(device_i)
        tensor = torch.rand(5, 5).to("musa")
        assert tensor.device == torch.device("musa:" + str(device_i))
    torch_musa.set_device(curr_device)  # reset device
