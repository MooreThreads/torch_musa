#ifndef TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>
#include "musa_runtime_api.h"

#define MUSA_COMPILE_TIME_MAX_GPUS 16

namespace c10 {
namespace musa {

using c10::DeviceIndex;

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

DeviceIndex exchangeDevice(DeviceIndex);

void set_device(DeviceIndex);

void Synchronize();

void init_mem_get_func(PyObject* module);
} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSADEVICE_H_
