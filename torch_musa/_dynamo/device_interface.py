"""MusaInterface implementation"""

# pylint: disable=unused-import
import torch
from torch._dynamo.device_interface import (
    DeviceInterface,
    caching_worker_current_devices,
    caching_worker_device_properties,
    _device_t,
    register_interface_for_device,
    get_registered_device_interfaces,
)

from torch_musa._MUSAC import _musa_getCurrentRawStream as get_musa_stream


class MusaInterface(DeviceInterface):
    """class of MusaInterface"""

    device = torch.musa.device

    # register Event and Stream class into the backend interface
    # make sure Event and Stream are implemented and inherited from the _EventBase and _StreamBase
    Event = torch.musa.Event
    Stream = torch.musa.Stream

    class Worker:
        """class of Worker"""

        @staticmethod
        def set_device(device: int):
            caching_worker_current_devices["musa"] = device

        @staticmethod
        def current_device() -> int:
            if "musa" in caching_worker_current_devices:
                return caching_worker_current_devices["musa"]
            return torch.musa.current_device()

        @staticmethod
        def get_device_properties(device: _device_t = None):
            """return musa device properties and cache them"""
            if device is not None:
                if isinstance(device, str):
                    device = torch.device(device)
                    assert device.type == "musa"
                if isinstance(device, torch.device):
                    device = device.index
            if device is None:
                device = MusaInterface.Worker.current_device()

            if "musa" not in caching_worker_device_properties:
                device_prop = [
                    torch.musa.get_device_properties(i)
                    for i in range(torch.musa.device_count())
                ]
                caching_worker_device_properties["musa"] = device_prop

            return caching_worker_device_properties["musa"][device]

    current_device = staticmethod(torch.musa.current_device)
    set_device = staticmethod(torch.musa.set_device)
    device_count = staticmethod(torch.musa.device_count)
    stream = staticmethod(torch.musa.stream)
    current_stream = staticmethod(torch.musa.current_stream)
    set_stream = staticmethod(torch.musa.set_stream)
    _set_stream_by_id = staticmethod(torch.musa._set_stream_by_id)
    synchronize = staticmethod(torch.musa.synchronize)
    get_device_properties = staticmethod(torch.musa.get_device_properties)
    get_raw_stream = staticmethod(get_musa_stream)
    exchange_device = staticmethod(torch.musa._exchange_device)
    maybe_exchange_device = staticmethod(torch.musa._maybe_exchange_device)

    # Can be mock patched by @patch decorator.
    @staticmethod
    def is_available() -> bool:
        return torch.musa.is_available()

    @staticmethod
    def get_compute_capability(device: _device_t = None):
        major_, min_ = torch.musa.get_device_capability(device)
        return major_ * 10 + min_

    @staticmethod
    def is_bf16_supported(including_emulation: bool = False) -> bool:
        return torch.musa.is_bf16_supported()
