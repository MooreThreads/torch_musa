"""Monkey patches that make ``torchrun --numa-binding`` work with MUSA.

PyTorch 2.11's NUMA implementation assumes CUDA device properties and requires
the external ``numactl`` command for subprocess entrypoints. This module keeps
the original torchrun/elastic launch flow and patches only those backend-specific
helpers. The patch is intentionally version-checked and idempotent.
"""

# pylint: disable=protected-access

import shutil

import torch
from torch.numa import binding


_PATCH_APPLIED = False
_REQUIRED_BINDING_HELPERS = (
    "_assemble_numactl_command_args",
    "_get_gpu_count",
    "_get_numa_node_index_for_gpu_index",
    "_get_ranges_str_from_ints",
    "_raise_if_binding_invalid",
)


def _get_accelerator_count() -> int:
    return torch.musa.device_count()


def _get_numa_node_for_accelerator(*, gpu_index: int) -> int:
    device_properties = torch.musa.get_device_properties(gpu_index)

    domain = device_properties.pci_domain_id
    bus = device_properties.pci_bus_id
    device = device_properties.pci_device_id

    # Format to sysfs PCI address: "0000:dc:00.0"
    pci_addr = f"{domain:04x}:{bus:02x}:{device:02x}.0"

    pci_numa_node_absolute_path = f"/sys/bus/pci/devices/{pci_addr}/numa_node"
    with open(pci_numa_node_absolute_path, encoding="utf-8") as numa_node_file:
        # In systems with only one NUMA node, this will often be saved as -1.
        # In those cases, there is at least one NUMA node, 0, so use that.
        return max(int(numa_node_file.read().strip()), 0)


def _assemble_binding_command(
    *, original_command_args: tuple[str, ...], logical_cpu_indices: set[int]
) -> tuple[str, ...]:
    cpu_list = binding._get_ranges_str_from_ints(logical_cpu_indices)
    if numactl := shutil.which("numactl"):
        return (numactl, f"--physcpubind={cpu_list}", *original_command_args)
    if taskset := shutil.which("taskset"):
        return (taskset, "--cpu-list", cpu_list, *original_command_args)
    raise RuntimeError("numactl or taskset CLI is required for NUMA binding")


def _validate_binding(*, logical_cpu_indices: set[int]) -> None:
    if shutil.which("numactl") is None and shutil.which("taskset") is None:
        raise RuntimeError("numactl or taskset CLI is required for NUMA binding")
    if not logical_cpu_indices:
        raise RuntimeError("Must bind to a non-empty set of CPU indices")


def apply_torchrun_numa_patch() -> None:
    """Patch PyTorch NUMA backend helpers for the current process.

    Existing references to ``_maybe_wrap_command_args_with_numa_binding`` remain
    valid because that function resolves these helpers from its module globals.
    """
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    missing = [name for name in _REQUIRED_BINDING_HELPERS if not hasattr(binding, name)]
    if missing:
        raise RuntimeError(
            "Unsupported PyTorch NUMA binding implementation; missing helpers: "
            + ", ".join(missing)
        )

    binding._get_gpu_count = _get_accelerator_count
    binding._get_numa_node_index_for_gpu_index = _get_numa_node_for_accelerator
    binding._assemble_numactl_command_args = _assemble_binding_command
    binding._raise_if_binding_invalid = _validate_binding
    _PATCH_APPLIED = True


__all__ = ["apply_torchrun_numa_patch"]
