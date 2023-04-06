#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace torch_musa {

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      int result;
      TORCH_MUSARUNTIME_CHECK(musaGetDeviceCount(&result));
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many MUSA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // Terminated if fail and log the following message.
      TORCH_INTERNAL_ASSERT(false, "MUSA initialization: ", ex.msg());
    }
  }();

  return static_cast<DeviceIndex>(count);
}

DeviceIndex current_device() {
  int cur_device;
  TORCH_MUSARUNTIME_CHECK(musaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  TORCH_MUSARUNTIME_CHECK(musaSetDevice(static_cast<int>(device)));
}

} // namespace torch_musa
