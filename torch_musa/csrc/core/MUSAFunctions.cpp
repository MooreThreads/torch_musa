#include "torch_musa/csrc/core/MUSAFunctions.h"

#include <c10/core/DeviceGuard.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"

namespace c10::musa {

int32_t GetDriverVersion() {
  int driver_version = -1;
  C10_MUSA_IGNORE_ERROR(musaDriverGetVersion(&driver_version));
  return driver_version;
}

namespace {

int device_count_impl(bool fail_if_no_driver) {
  int count = 0;
  auto err = C10_MUSA_ERROR_HANDLED(GetDeviceCount(&count));
  if (err == musaSuccess) {
    return count;
  }
  musaError_t last_err C10_UNUSED = musaGetLastError();
  switch (err) {
    case musaErrorNoDevice:
      // Zero devices is ok here
      count = 0;
      break;
    case musaErrorInsufficientDriver: {
      auto version = GetDriverVersion();
      if (version <= 0) {
        if (!fail_if_no_driver) {
          // No MUSA driver means no devices
          count = 0;
          break;
        }
        TORCH_CHECK(
            false,
            "Found no driver on your system. Please check that you "
            "have an Mthreads GPU and installed a driver");
      } else {
        TORCH_CHECK(
            false,
            "The driver on your system is too old (found version ",
            version,
            ").");
      }
    } break;
    case musaErrorInitializationError:
      TORCH_CHECK(
          false,
          "MUSA driver initialization failed, you might not "
          "have a Mthreads GPU.");
      break;
    case musaErrorUnknown:
      TORCH_CHECK(
          false,
          "MUSA unknown error - this may be due to an "
          "incorrectly set up environment, e.g. changing env "
          "variable MUSA_VISIBLE_DEVICES after program start. "
          "Setting the available devices to be zero.");
      break;
    default:
      TORCH_CHECK(
          false,
          "Unexpected error from musaGetDeviceCount(). Did you run "
          "some musa functions before calling NumMusaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          musaGetErrorString(err));
  }
  return count;
}

} // anonymous namespace

bool is_musa_available() {
  return device_count() > 0;
}

DeviceIndex device_count() noexcept {
  static int count = []() {
    try {
      auto result = device_count_impl(/*fail_if_no_driver=*/false);
      TORCH_INTERNAL_ASSERT(
          result <= std::numeric_limits<DeviceIndex>::max(),
          "Too many MUSA devices, DeviceIndex overflowed");
      return result;
    } catch (const c10::Error& ex) {
      // We don't want to fail, but still log the warning
      // msg() returns the message without the stack trace
      TORCH_WARN("MUSA initialization: ", ex.msg());
      return 0;
    }
  }();

  return static_cast<DeviceIndex>(count);
}

DeviceIndex device_count_ensure_non_zero() {
  int count = device_count_impl(/*fail_if_no_driver=*/true);
  TORCH_CHECK(count, "No MUSA GPUs are available");
  TORCH_INTERNAL_ASSERT(
      count <= std::numeric_limits<DeviceIndex>::max(),
      "Too many MUSA devices, DeviceIndex overflowed");
  return static_cast<DeviceIndex>(count);
}

musaError_t GetDeviceCount(int* dev_count) {
  return musaGetDeviceCount(dev_count);
}

thread_local DeviceIndex targetDeviceIndex = -1;

musaError_t GetDevice(DeviceIndex* device) {
  if (targetDeviceIndex >= 0) {
    *device = targetDeviceIndex;
    return musaSuccess;
  }
  int tmp_device = -1;
  auto err = musaGetDevice(&tmp_device);
  if (err == musaSuccess) {
    TORCH_INTERNAL_ASSERT(
        tmp_device >= 0 &&
            tmp_device <= std::numeric_limits<DeviceIndex>::max(),
        "musaGetDevice returns invalid device ",
        tmp_device);
    *device = static_cast<DeviceIndex>(tmp_device);
  }
  return err;
}

musaError_t SetDevice(DeviceIndex device) {
  TORCH_CHECK(device >= 0, "device id must be positive!", device);
  targetDeviceIndex = -1;
  int cur_device = -1;
  C10_MUSA_CHECK(musaGetDevice(&cur_device));
  if (device == cur_device) {
    return musaSuccess;
  }
  return musaSetDevice(device);
}

musaError_t MaybeSetDevice(DeviceIndex device) {
  if (hasPrimaryContext(device)) {
    return SetDevice(device);
  }
  targetDeviceIndex = device;
  return musaSuccess;
}

DeviceIndex ExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_MUSA_CHECK(musaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  C10_MUSA_CHECK(musaSetDevice(to_device));
  return cur_device;
}

DeviceIndex MaybeExchangeDevice(DeviceIndex to_device) {
  auto cur_device = targetDeviceIndex;
  targetDeviceIndex = -1;
  if (cur_device < 0) {
    int tmp_device = -1;
    C10_MUSA_CHECK(musaGetDevice(&tmp_device));
    cur_device = static_cast<DeviceIndex>(tmp_device);
    if (to_device == cur_device) {
      return cur_device;
    }
  }
  if (hasPrimaryContext(to_device)) {
    C10_MUSA_CHECK(musaSetDevice(to_device));
  } else {
    targetDeviceIndex = to_device;
  }
  return cur_device;
}

void SetTargetDevice() {
  if (targetDeviceIndex >= 0) {
    C10_MUSA_CHECK(SetDevice(targetDeviceIndex));
  }
}

bool hasPrimaryContext(DeviceIndex device_index) {
  TORCH_CHECK(
      device_index >= 0 && device_index < device_count(),
      "hasPrimaryContext expects a valid device index, but got device_index=",
      device_index);
  unsigned int ctx_flags;
  int ctx_is_active = 0;
  AT_MUSA_DRIVER_CHECK(
      muDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active));
  return ctx_is_active == 1;
}

void device_synchronize() {
  const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
  if (C10_UNLIKELY(interp)) {
    (*interp)->trace_gpu_device_synchronization(c10::kPrivateUse1);
  }
  C10_MUSA_CHECK(musaDeviceSynchronize());
}

void warn_or_error_on_sync() {
  if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_ERROR) {
    TORCH_CHECK(false, "called a synchronizing MUSA operation");
  } else if (warning_state().get_sync_debug_mode() == SyncDebugMode::L_WARN) {
    TORCH_WARN("called a synchronizing MUSA operation");
  }
}

Device getDeviceFromPtr(void* ptr) {
  musaPointerAttributes attr{};
  TORCH_MUSA_CHECK(musaPointerGetAttributes(&attr, ptr));
  TORCH_CHECK(
      attr.type != musaMemoryTypeUnregistered,
      "The specified pointer resides on host memory and is not registered with any MUSA device.");
  return {c10::kPrivateUse1, static_cast<DeviceIndex>(attr.device)};
}

std::optional<DeviceIndex> getDeviceIndexWithPrimaryContext() {
  // check current device first
  DeviceIndex current_device_index = -1;
  C10_MUSA_CHECK(GetDevice(&current_device_index));
  if (current_device_index >= 0) {
    if (hasPrimaryContext(current_device_index)) {
      return current_device_index;
    }
  }
  for (const auto device_index : c10::irange(device_count())) {
    if (device_index == current_device_index)
      continue;
    if (hasPrimaryContext(device_index)) {
      return device_index;
    }
  }
  return std::nullopt;
}

bool isPinnedPtr(const void* data) {
  if (!is_musa_available()) {
    return false;
  }
  // musaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = getDeviceIndexWithPrimaryContext();
  if (primary_ctx_device_index.has_value()) {
    device_guard.reset_device(
        at::Device(at::DeviceType::PrivateUse1, *primary_ctx_device_index));
  }
  musaPointerAttributes attr;
  musaError_t err = musaPointerGetAttributes(&attr, data);
  if (err == musaErrorInvalidValue) {
    (void)musaGetLastError();
    return false;
  }
  TORCH_MUSA_CHECK(err);
  return attr.type == musaMemoryTypeHost;
}

} // namespace c10::musa
