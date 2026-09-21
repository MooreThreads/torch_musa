"""Tests for the torchrun MUSA NUMA monkey patch."""

import shutil
from types import SimpleNamespace
from unittest import mock

import pytest
from torch.numa import binding
import torch_musa

from torch_musa.distributed.numa_binding import apply_torchrun_numa_patch
from torch_musa.distributed.numa_binding import _assemble_binding_command
from torch_musa.distributed.numa_binding import _get_numa_node_for_accelerator


def test_patch_is_idempotent():
    """Applying the patch repeatedly must be safe."""
    apply_torchrun_numa_patch()
    apply_torchrun_numa_patch()

    assert binding._get_gpu_count() == torch_musa.device_count()


def test_numa_node_uses_device_properties_pci_address():
    """The accelerator PCI address should select the NUMA sysfs file."""
    properties = SimpleNamespace(
        pci_domain_id=0,
        pci_bus_id=0xDC,
        pci_device_id=0,
    )
    get_device_properties = mock.Mock(return_value=properties)
    numa_node_file = mock.mock_open(read_data="-1\n")

    with mock.patch.object(
        torch_musa, "get_device_properties", get_device_properties
    ), mock.patch("builtins.open", numa_node_file):
        assert _get_numa_node_for_accelerator(gpu_index=3) == 0

    get_device_properties.assert_called_once_with(3)
    numa_node_file.assert_called_once_with(
        "/sys/bus/pci/devices/0000:dc:00.0/numa_node", encoding="utf-8"
    )


@pytest.mark.skipif(shutil.which("taskset") is None, reason="taskset is unavailable")
def test_taskset_fallback():
    """taskset should be used when numactl is unavailable."""
    with mock.patch("shutil.which") as which:
        which.side_effect = lambda command: (
            "/usr/bin/taskset" if command == "taskset" else None
        )
        command = _assemble_binding_command(
            original_command_args=("python", "train.py"),
            logical_cpu_indices={0, 1, 4},
        )

    assert command == (
        "/usr/bin/taskset",
        "--cpu-list",
        "0-1,4",
        "python",
        "train.py",
    )
