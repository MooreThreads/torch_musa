#ifndef TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>
#include "musa_runtime_api.h"
namespace py = pybind11;
namespace torch_musa {

using c10::DeviceIndex;

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

DeviceIndex exchangeDevice(DeviceIndex);

void set_device(DeviceIndex);

musaDeviceProp* getDeviceProperties(int device);

bool canDeviceAccessPeer(int device, int peer_device);

void Synchronize();

void registerMusaDeviceProperties(PyObject* module);

} // namespace torch_musa
#endif // TORCH_MUSA_CSRC_CORE_MUSADEVICE_H_
