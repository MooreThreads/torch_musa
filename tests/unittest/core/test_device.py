"""Test device features."""
#!/usr/bin/env python
# coding=utf-8
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
    try:
        with torch_musa.device("musa"):
            pass
    except ValueError as err:
        assert (
            str(err)
            == "Expected a torch.device with a specified index or an integer, but got:None"
        )

    try:
        with torch_musa.device("cuda"):
            pass
    except ValueError as err:
        assert str(err) == "Expected a musa device, but got: cuda"

    try:
        with torch_musa.device("cpu"):
            pass
    except ValueError as err:
        assert str(err) == "Expected a musa device, but got: cpu"


def test_synchronize():
    """Test torch_musa synchronize feature"""
    # TODO(Xiaokang Shang): Add synchronize test cases
