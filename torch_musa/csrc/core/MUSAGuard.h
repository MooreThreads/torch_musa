#ifndef TORCH_MUSA_CSRC_CORE_MUSAGUARD_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGUARD_H_

#include <c10/core/impl/InlineDeviceGuard.h>
#include <c10/core/impl/InlineStreamGuard.h>

#include "torch_musa/csrc/core/GuardImpl.h"

namespace c10::musa {

using at::musa::kMUSA;

struct MUSAGuard {
  explicit MUSAGuard() = delete;

  explicit MUSAGuard(DeviceIndex device_index) : guard_(device_index) {}

  explicit MUSAGuard(Device device) : guard_(device) {}

  MUSAGuard(const MUSAGuard&) = delete;
  MUSAGuard& operator=(const MUSAGuard&) = delete;

  MUSAGuard(MUSAGuard&& other) = delete;
  MUSAGuard& operator=(MUSAGuard&& other) = delete;

  void set_device(Device device) {
    guard_.set_device(device);
  }

  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  Device original_device() const {
    return guard_.original_device();
  }

  Device current_device() const {
    return guard_.current_device();
  }

 private:
  c10::impl::InlineDeviceGuard<impl::MUSAGuardImpl> guard_;
};

struct OptionalMUSAGuard {
  explicit OptionalMUSAGuard() : guard_() {}

  explicit OptionalMUSAGuard(std::optional<Device> device_opt)
      : guard_(device_opt) {}

  explicit OptionalMUSAGuard(std::optional<DeviceIndex> device_index_opt)
      : guard_(device_index_opt) {}

  OptionalMUSAGuard(const OptionalMUSAGuard&) = delete;
  OptionalMUSAGuard& operator=(const OptionalMUSAGuard&) = delete;

  OptionalMUSAGuard(OptionalMUSAGuard&& other) = delete;
  OptionalMUSAGuard& operator=(OptionalMUSAGuard&& other) = delete;

  void set_device(Device device) {
    guard_.set_device(device);
  }

  void reset_device(Device device) {
    guard_.reset_device(device);
  }

  void set_index(DeviceIndex device_index) {
    guard_.set_index(device_index);
  }

  std::optional<Device> original_device() const {
    return guard_.original_device();
  }

  std::optional<Device> current_device() const {
    return guard_.current_device();
  }

  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalDeviceGuard<impl::MUSAGuardImpl> guard_;
};

struct MUSAStreamGuard {
  explicit MUSAStreamGuard() = delete;

  explicit MUSAStreamGuard(Stream stream) : guard_(stream) {}

  MUSAStreamGuard(const MUSAStreamGuard&) = delete;
  MUSAStreamGuard& operator=(const MUSAStreamGuard&) = delete;

  MUSAStreamGuard(MUSAStreamGuard&& other) = delete;
  MUSAStreamGuard& operator=(MUSAStreamGuard&& other) = delete;

  /// NOTE: this implementation may skip some stream/device setting if
  /// it can prove that it is unnecessary.
  ///
  /// WARNING: reset_stream does NOT preserve previously set streams on
  /// different devices.  If you need to set streams on multiple devices
  /// on MUSA, use MUSAMultiStreamGuard instead.
  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  MUSAStream original_stream() const {
    return MUSAStream(MUSAStream::UNCHECKED, guard_.original_stream());
  }

  MUSAStream current_stream() const {
    return MUSAStream(MUSAStream::UNCHECKED, guard_.current_stream());
  }

  Device current_device() const {
    return guard_.current_device();
  }

  Device original_device() const {
    return guard_.original_device();
  }

 private:
  c10::impl::InlineStreamGuard<impl::MUSAGuardImpl> guard_;
};

struct OptionalMUSAStreamGuard {
  explicit OptionalMUSAStreamGuard() : guard_() {}

  explicit OptionalMUSAStreamGuard(Stream stream) : guard_(stream) {}

  explicit OptionalMUSAStreamGuard(std::optional<Stream> stream_opt)
      : guard_(stream_opt) {}

  OptionalMUSAStreamGuard(const OptionalMUSAStreamGuard&) = delete;
  OptionalMUSAStreamGuard& operator=(const OptionalMUSAStreamGuard&) = delete;

  OptionalMUSAStreamGuard(OptionalMUSAStreamGuard&& other) = delete;
  OptionalMUSAStreamGuard& operator=(OptionalMUSAStreamGuard&& other) = delete;

  void reset_stream(Stream stream) {
    guard_.reset_stream(stream);
  }

  std::optional<MUSAStream> original_stream() const {
    auto r = guard_.original_stream();
    if (r.has_value()) {
      return std::make_optional(MUSAStream(MUSAStream::UNCHECKED, r.value()));
    } else {
      return std::nullopt;
    }
  }

  std::optional<MUSAStream> current_stream() const {
    auto r = guard_.current_stream();
    if (r.has_value()) {
      return std::make_optional(MUSAStream(MUSAStream::UNCHECKED, r.value()));
    } else {
      return std::nullopt;
    }
  }

  void reset() {
    guard_.reset();
  }

 private:
  c10::impl::InlineOptionalStreamGuard<impl::MUSAGuardImpl> guard_;
};

struct MUSAMultiStreamGuard {
  explicit MUSAMultiStreamGuard(ArrayRef<MUSAStream> streams)
      : guard_(unwrapStreams(streams)) {}

  MUSAMultiStreamGuard(const MUSAMultiStreamGuard&) = delete;
  MUSAMultiStreamGuard& operator=(const MUSAMultiStreamGuard&) = delete;

  MUSAMultiStreamGuard(MUSAMultiStreamGuard&& other) = delete;
  MUSAMultiStreamGuard& operator=(MUSAMultiStreamGuard&& other) = delete;

 private:
  c10::impl::InlineMultiStreamGuard<impl::MUSAGuardImpl> guard_;

  static std::vector<Stream> unwrapStreams(ArrayRef<MUSAStream> musaStreams) {
    std::vector<Stream> streams;
    streams.reserve(musaStreams.size());
    for (const MUSAStream& musaStream : musaStreams) {
      streams.push_back(musaStream);
    }
    return streams;
  }
};

} // namespace c10::musa

#endif // TORCH_MUSA_CSRC_CORE_MUSAGUARD_H_
