"""define MusaInterface"""

# pylint: disable=abstract-method, unused-import
import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
    _device_t,
    register_interface_for_device,
)

from torch_musa._MUSAC import _musa_getCurrentRawStream as get_musa_stream

_backend_device = torch._C._get_privateuse1_backend_name()


class MusaInterface(DeviceInterface):
    """MusaInterface"""

    device = torch.musa.device
    Event = torch.musa.Event
    Stream = torch.musa.Stream

    class Worker:
        """Worker"""

        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices[_backend_device] = device

        @staticmethod
        def current_device() -> int:
            if _backend_device in caching_worker_current_devices:
                return caching_worker_current_devices[_backend_device]
            return torch.musa.current_device()

        @staticmethod
        def get_device_properties(device=None):
            """get musa device properties"""
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == _backend_device
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MusaInterface.Worker.current_device()

            if _backend_device not in caching_worker_device_properties:
                device_prop = [
                    torch.musa.get_device_properties(i)
                    for i in range(torch.musa.device_count())
                ]
                caching_worker_device_properties[_backend_device] = device_prop

            return caching_worker_device_properties[_backend_device][device]

    current_device = staticmethod(torch.musa.current_device)
    set_device = staticmethod(torch.musa.set_device)
    device_count = staticmethod(torch.musa.device_count)
    stream = staticmethod(torch.musa.stream)
    current_stream = staticmethod(torch.musa.current_stream)
    set_stream = staticmethod(torch.musa.set_stream)
    # _set_stream_by_id = staticmethod(torch.musa._set_stream_by_id)
    synchronize = staticmethod(torch.musa.synchronize)
    get_device_properties = staticmethod(torch.musa.get_device_properties)
    get_raw_stream = staticmethod(get_musa_stream)

    @staticmethod
    def is_available() -> bool:
        return torch.musa.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        _major, _min = torch.musa.get_device_capability(device)
        return _major * 10 + _min
