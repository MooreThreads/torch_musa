#ifndef TORCH_MUSA_CSRC_CORE_MUSA_GUARD_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_GUARD_H_
#include <c10/core/DeviceType.h>
#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "torch_musa/csrc/core/GuardImpl.h"

namespace torch_musa {
using at::native::musa::kMUSA;
using c10::Device;
using c10::DeviceType;
using c10::optional;
using c10::Stream;

struct MUSAGuard {
  /// No default constructor.
  explicit MUSAGuard() = delete;

  /// Set the current MUSA device to the passed device index.
  explicit MUSAGuard(DeviceIndex device_index) : guard_(device_index) {}

  /// Sets the current MUSA device to the passed device.  Errors if the passed
  /// device is not a MUSA device.
  explicit MUSAGuard(Device device) : guard_(device) {}

  // Copy constructor is not allowed.
  MUSAGuard(const MUSAGuard&) = delete;
  MUSAGuard& operator=(const MUSAGuard&) = delete;

  // Move constructor is not allowed.
  MUSAGuard(MUSAGuard&& other) = delete;
  MUSAGuard& operator=(MUSAGuard&& other) = delete;

  /// Sets the MUSA device to the given device.  Errors if the given device
  /// is not a MUSA device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  /// Sets the MUSA device to the given device.  Errors if the given device
  /// is not a MUSA device.
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  /// Sets the MUSA device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  /// Returns the device that was set upon construction of the guard.
  Device original_device() const {
    return guard_.original_device();
  }

  /// Returns the last device that was set via `set_device`, if any, otherwise
  /// the device passed during construction.
  Device current_device() const {
    return guard_.current_device();
  }

 private:
  /// The guard for the current device.
  c10::impl::InlineDeviceGuard<impl::MUSAGuardImpl> guard_;
};

struct OptionalMUSAGuard {
  /// Create an uninitialized OptionalMUSAGuard.
  explicit OptionalMUSAGuard() : guard_() {}

  /// Set the current MUSA device to the passed Device, if it is not nullopt.
  explicit OptionalMUSAGuard(optional<Device> device_opt)
      : guard_(device_opt) {}

  /// Set the current MUSA device to the passed device index, if it is not
  /// nullopt
  explicit OptionalMUSAGuard(optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  // Copy constructor is not allowed.
  OptionalMUSAGuard(const OptionalMUSAGuard&) = delete;
  OptionalMUSAGuard& operator=(const OptionalMUSAGuard&) = delete;

  // Move constructor is not allowed.
  OptionalMUSAGuard(OptionalMUSAGuard&& other) = delete;
  OptionalMUSAGuard& operator=(OptionalMUSAGuard&& other) = delete;

  // Sets the MUSA device to the given device.
  void set_device(Device device) {
    guard_.set_device(device);
  }

  // Set the MUSA device to the given device.
  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  // Set the MUSA device to the given device index.
  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  optional<Device> original_device() const {
    return guard_.original_device();
  }

  // Return s the most recent device that was set using this device guard,
  // either from construction, or via set_device.
  optional<Device> current_device() const {
    return guard_.current_device();
  }

  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::MUSAGuardImpl> guard_;
};

} // namespace torch_musa

#endif // TORCH_MUSA_CSRC_CORE_MUSA_GUARD_H_
