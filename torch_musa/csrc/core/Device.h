#ifndef TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_DEVICE_H_

#include <c10/core/Device.h>
#include "musa_runtime_api.h"

namespace torch_musa {

using c10::DeviceIndex;

DeviceIndex device_count() noexcept;

DeviceIndex current_device();

void set_device(DeviceIndex);

} // namespace torch_musa
#endif // TORCH_MUSA_CSRC_CORE_MUSADEVICE_H_
