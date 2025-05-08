#ifndef TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_

#include <c10/core/Device.h>
#include <pybind11/pybind11.h>

#define MUSA_COMPILE_TIME_MAX_GPUS 16

namespace c10::musa {

DeviceIndex current_device();

void set_device(DeviceIndex device);

void Synchronize();

void init_mem_get_func(PyObject* module);

} // namespace c10::musa

#endif // TORCH_MUSA_CSRC_CORE_MUSADEVICE_H_
