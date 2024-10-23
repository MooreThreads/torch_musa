#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/Device.h"

#include <c10/util/Exception.h>
#include <c10/util/irange.h>

namespace c10 {
namespace musa {
// this function has to be called from callers performing musa synchronizing
// operations, to raise proper error or warning
void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing MUSA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing MUSA operation");
  }
}

musaError_t GetDevice(int* device) {
  return musaGetDevice(device);
}

bool hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < c10::musa::device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      device_index);
  unsigned int ctx_flags;
  int ctx_is_active = 0;
  AT_MUSA_DRIVER_CHECK(
      muDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

c10::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  auto current_device_index = c10::musa::current_device();
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(c10::musa::device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return c10::nullopt;
}

} // namespace musa
} // namespace c10
