#include "torch_musa/csrc/core/Device.h"
#include <c10/util/CallOnce.h>
#include <deque>
#include "torch_musa/csrc/core/MUSAException.h"

namespace c10 {
namespace musa {

namespace {

int32_t driver_version() {
  int driver_version = -1;
  musaDriverGetVersion(&driver_version);
  return driver_version;
}

int device_count_impl(bool fail_if_no_driver) {
  int count;
  auto err = musaGetDeviceCount(&count);
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
      auto version = driver_version();
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
          "some musa functions before calling NumCudaDevices() "
          "that might have already set an error? Error ",
          err,
          ": ",
          musaGetErrorString(err));
  }
  return count;
}

} // anonymous namespace

DeviceIndex device_count() noexcept {
  // initialize number of devices only once
  static int count = []() {
    try {
      int result = device_count_impl(/*fail_if_no_driver=*/false);
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

DeviceIndex current_device() {
  int cur_device;
  TORCH_MUSA_CHECK(musaGetDevice(&cur_device));
  return static_cast<DeviceIndex>(cur_device);
}

void set_device(DeviceIndex device) {
  TORCH_MUSA_CHECK(musaSetDevice(static_cast<int>(device)));
}

DeviceIndex exchangeDevice(DeviceIndex device) {
  if (device < 0) {
    return static_cast<DeviceIndex>(-1);
  }
  auto cur_device = current_device();
  if (cur_device != device) {
    set_device(device);
  }
  return cur_device;
}

void Synchronize() {
  musaDeviceSynchronize();
}

} // namespace musa
} // namespace c10
