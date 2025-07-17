"""MUSADeviceOpOverrides implementation"""

# pylint:disable=unused-import
from torch._inductor.codegen.common import (
    DeviceOpOverrides,
    register_device_op_overrides,
)


class MUSADeviceOpOverrides(DeviceOpOverrides):
    """class of MUSADeviceOpOverrides"""

    def import_get_raw_stream_as(self, name):
        return f"from torch_musa._MUSAC import _musa_getCurrentRawStream as {name}"

    def set_device(self, device_idx):
        return f"torch.musa.set_device({device_idx})"

    def synchronize(self):
        return "torch.musa.synchronize()"

    def device_guard(self, device_idx):
        return f"torch.musa._DeviceGuard({device_idx})"
