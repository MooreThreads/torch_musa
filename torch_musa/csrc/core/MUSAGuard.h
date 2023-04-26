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

/// A variant of StreamGuard that is specialized for MUSA.  See MUSAGuard
/// for when you can use this.
struct MUSAStreamGuard {
  /// No default constructor, see Note [Omitted default constructor from RAII]
  explicit MUSAStreamGuard() = delete;

  /// Set the current MUSA device to the device associated with the passed
  /// stream, and set the current MUSA stream on that device to the passed
  /// stream. Errors if the Stream is not a MUSA stream.
  explicit MUSAStreamGuard(Stream stream) : guard_(stream) {}

  /// Copy is disallowed
  MUSAStreamGuard(const MUSAStreamGuard&) = delete;
  MUSAStreamGuard& operator=(const MUSAStreamGuard&) = delete;

  /// Move is disallowed, as MUSAStreamGuard does not have an uninitialized
  /// state, which is required for moves on types with nontrivial destructors.
  MUSAStreamGuard(MUSAStreamGuard&& other) = delete;
  MUSAStreamGuard& operator=(MUSAStreamGuard&& other) = delete;

  /// Resets the currently set stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Errors if the stream passed is not a MUSA stream.
  ///
  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on MUSA, use MUSAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the MUSA stream that was set at the time the guard was
  /// constructed.
  MUSAStream original_stream() const {
    return MUSAStream(MUSAStream::UNCHECKED, guard_.original_stream());
  }

  /// Returns the most recent MUSA stream that was set using this device guard,
  /// either from construction, or via set_stream.
  MUSAStream current_stream() const {
    return MUSAStream(MUSAStream::UNCHECKED, guard_.current_stream());
  }

  /// Returns the most recent MUSA device that was set using this device guard,
  /// either from construction, or via set_device/reset_device/set_index.
  Device current_device() const {
    return guard_.current_device();
  }

  /// Returns the MUSA device that was set at the most recent reset_stream(),
  /// or otherwise the device at construction time.
  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::MUSAGuardImpl> guard_;
};

/// A variant of OptionalStreamGuard that is specialized for MUSA.  See
/// MUSAGuard for when you can use this.
struct OptionalMUSAStreamGuard {
  /// Create an uninitialized guard.
  explicit OptionalMUSAStreamGuard() : guard_() {}

  /// Set the current MUSA device to the device associated with the passed
  /// stream, and set the current MUSA stream on that device to the passed
  /// stream. Errors if the Stream is not a MUSA stream.
  explicit OptionalMUSAStreamGuard(Stream stream) : guard_(stream) {}

  /// Set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream,
  /// if the passed stream is not nullopt.
  explicit OptionalMUSAStreamGuard(optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  /// Copy is disallowed
  OptionalMUSAStreamGuard(const OptionalMUSAStreamGuard&) = delete;
  OptionalMUSAStreamGuard& operator=(const OptionalMUSAStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  OptionalMUSAStreamGuard(OptionalMUSAStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  OptionalMUSAStreamGuard& operator=(OptionalMUSAStreamGuard&& other) = delete;

  /// Resets the currently set MUSA stream to the original stream and
  /// the currently set device to the original device.  Then,
  /// set the current device to the device associated with the passed stream,
  /// and set the current stream on that device to the passed stream.
  /// Initializes the guard if it was not previously initialized.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  /// Returns the MUSA stream that was set at the time the guard was most
  /// recently initialized, or nullopt if the guard is uninitialized.
  optional<MUSAStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return c10::make_optional(MUSAStream(MUSAStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Returns the most recent MUSA stream that was set using this stream guard,
  /// either from construction, or via reset_stream, if the guard is
  /// initialized, or nullopt if the guard is uninitialized.
  optional<MUSAStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return c10::make_optional(MUSAStream(MUSAStream::UNCHECKED, r.value()));
    } else {
      return c10::nullopt;
    }
  }

  /// Restore the original MUSA device and stream, resetting this guard to
  /// uninitialized state.
  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::MUSAGuardImpl> guard_;
};

/// A variant of MultiStreamGuard that is specialized for MUSA.
struct MUSAMultiStreamGuard {
  explicit MUSAMultiStreamGuard(c10::ArrayRef<MUSAStream> streams)
      : guard_(unwrapStreams(streams)) {}

  /// Copy is disallowed
  MUSAMultiStreamGuard(const MUSAMultiStreamGuard&) = delete;
  MUSAMultiStreamGuard& operator=(const MUSAMultiStreamGuard&) = delete;

  // See Note [Move construction for RAII guards is tricky]
  MUSAMultiStreamGuard(MUSAMultiStreamGuard&& other) = delete;

  // See Note [Move assignment for RAII guards is tricky]
  MUSAMultiStreamGuard& operator=(MUSAMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<impl::MUSAGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(
      c10::ArrayRef<MUSAStream> MUSAStreams) {
    std::vector<Stream> streams;
    streams.reserve(MUSAStreams.size());
    for (const MUSAStream& MUSAStream : MUSAStreams) {
      streams.push_back(MUSAStream);
    }
    return streams;
  }
};

} // namespace torch_musa

#endif // TORCH_MUSA_CSRC_CORE_MUSA_GUARD_H_
