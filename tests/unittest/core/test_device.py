"""Test device features."""
# pylint: disable=invalid-name, C0415
import torch
import pytest
import torch_musa

TEST_MUSA = torch_musa.is_available()
TEST_MULTIGPU = TEST_MUSA and torch_musa.device_count() >= 2


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


# TODO(Xiaokang Shang): Open after device allocator ready.
@pytest.mark.skipif(True, reason="Wait device allocator")
def test_copy_device():
    "Testing copy on multi cards"
    x = torch.randn(5, 5).to("musa")
    with torch_musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y

    x = torch.randn(5, 5)
    with torch_musa.device(1):
        y = x.to("musa")
        assert y.get_device() == 1
        assert y.to("musa") is y
        z = y.to("musa:0")
        assert z.get_device() == 0
        assert z.to("musa:0") is z
