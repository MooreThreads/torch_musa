#ifndef TORCH_MUSA_CSRC_CORE_MUSAEVENT_H_
#define TORCH_MUSA_CSRC_CORE_MUSAEVENT_H_

#include <cstdint>
#include <utility>

#include <musa_runtime_api.h>

#include <c10/core/impl/GPUTrace.h>

#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::musa {

namespace detail {

template <typename E, bool Lazy>
class EventPoolBase;

} // namespace detail

/*
 * MUSAEvents are movable not copyable wrappers around MUSA's events.
 *
 * MUSAEvents are constructed lazily when first recorded unless it is
 * reconstructed from a musaIpcEventHandle_t. The event has a device, and this
 * device is acquired from the first recording stream. However, if reconstructed
 * from a handle, the device should be explicitly specified; or if ipc_handle()
 * is called before the event is ever recorded, it will use the current device.
 * Later streams that record the event must match this device.
 */
struct MUSAEvent {
  friend class detail::EventPoolBase<MUSAEvent, true>;
  friend class detail::EventPoolBase<MUSAEvent, false>;

  MUSAEvent() noexcept = default;

  MUSAEvent(unsigned int flags) noexcept : flags_{flags} {}

  MUSAEvent(DeviceIndex device_index, const musaIpcEventHandle_t* handle)
      : device_index_(device_index) {
    const MUSAGuard guard(device_index_);

    C10_MUSA_CHECK(musaIpcOpenEventHandle(&event_, *handle));
    is_created_ = true;
  }

  // Note: event destruction done on creating device to avoid creating a
  // MUSA context on other devices.
  ~MUSAEvent() {
    try {
      if (is_created_) {
        const MUSAGuard guard(device_index_);
        const c10::impl::PyInterpreter* interp =
            c10::impl::GPUTrace::get_trace();
        if (C10_UNLIKELY(interp)) {
          (*interp)->trace_gpu_event_deletion(
              kMUSA, reinterpret_cast<uintptr_t>(event_));
        }
        C10_MUSA_CHECK(musaEventDestroy(event_));
      }
    } catch (...) {
    }
  }

  MUSAEvent(const MUSAEvent&) = delete;
  MUSAEvent& operator=(const MUSAEvent&) = delete;

  MUSAEvent(MUSAEvent&& other) noexcept {
    moveHelper(std::move(other));
  }
  MUSAEvent& operator=(MUSAEvent&& other) noexcept {
    if (this != &other) {
      moveHelper(std::move(other));
    }
    return *this;
  }

  operator musaEvent_t() const {
    return event();
  }

  friend bool operator<(const MUSAEvent& left, const MUSAEvent& right) {
    return left.event_ < right.event_;
  }

  std::optional<Device> device() const {
    if (is_created_) {
      return Device(kMUSA, device_index_);
    } else {
      return std::nullopt;
    }
  }

  bool isCreated() const {
    return is_created_;
  }

  DeviceIndex device_index() const {
    return device_index_;
  }

  musaEvent_t event() const {
    return event_;
  }

  // Note: musaEventQuery can be safely called from any device
  bool query() const {
    if (!is_created_) {
      return true;
    }

    const auto err = TORCH_MUSA_ERROR_HANDLE(musaEventQuery(event_));
    if (err == musaSuccess) {
      return true;
    } else if (err != musaErrorNotReady) {
      C10_MUSA_CHECK(err);
    } else {
      musaGetLastError();
    }

    return false;
  }

  void record() {
    record(getCurrentMUSAStream());
  }

  void recordOnce(const MUSAStream& stream) {
    if (!was_recorded_) {
      record(stream);
    }
  }

  // Note: musaEventRecord must be called on the same device as the event.
  void record(const MUSAStream& stream) {
    if (!is_created_) {
      createEvent(stream.device_index());
    }

    TORCH_CHECK(
        device_index_ == stream.device_index(),
        "Event device ",
        device_index_,
        " does not match recording stream's device ",
        stream.device_index(),
        ".");
    const MUSAGuard guard(device_index_);
    C10_MUSA_CHECK(musaEventRecord(event_, stream));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          kMUSA,
          reinterpret_cast<uintptr_t>(event_),
          reinterpret_cast<uintptr_t>(stream.stream()));
    }
    was_recorded_ = true;
  }

  // Note: musaStreamWaitEvent must be called on the same device as the stream.
  // The event has no actual GPU resources associated with it.
  void block(const MUSAStream& stream) {
    if (is_created_) {
      const MUSAGuard guard(stream.device_index());
      C10_MUSA_CHECK(musaStreamWaitEvent(stream, event_, 0));
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_wait(
            kMUSA,
            reinterpret_cast<uintptr_t>(event_),
            reinterpret_cast<uintptr_t>(stream.stream()));
      }
    }
  }

  // Note: musaEventElapsedTime can be safely called from any device
  float elapsed_time(const MUSAEvent& other) const {
    TORCH_CHECK(
        is_created_ && other.isCreated(),
        "Both events must be recorded before calculating elapsed time.");
    float time_ms = 0;
    const MUSAGuard guard(device_index_);
    // raise musaErrorNotReady if either event is recorded but not yet completed
    C10_MUSA_CHECK(musaEventElapsedTime(&time_ms, event_, other.event_));
    return time_ms;
  }

  // Note: musaEventSynchronize can be safely called from any device
  void synchronize() const {
    if (is_created_) {
      const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
      if (C10_UNLIKELY(interp)) {
        (*interp)->trace_gpu_event_synchronization(
            kMUSA, reinterpret_cast<uintptr_t>(event_));
      }
      C10_MUSA_CHECK(musaEventSynchronize(event_));
    }
  }

  // Note: musaIpcGetEventHandle must be called on the same device as the event
  void ipc_handle(musaIpcEventHandle_t* handle) {
    if (!is_created_) {
      // this MUSAEvent object was initially constructed from flags but event_
      // is not created yet.
      createEvent(getCurrentMUSAStream().device_index());
    }
    const MUSAGuard guard(device_index_);
    C10_MUSA_CHECK(musaIpcGetEventHandle(handle, event_));
  }

 private:
  unsigned int flags_ = musaEventDisableTiming;
  bool is_created_ = false;
  bool was_recorded_ = false;
  DeviceIndex device_index_ = -1;
  musaEvent_t event_{};

  void createEvent(DeviceIndex device_index) {
    device_index_ = device_index;
    const MUSAGuard guard(device_index_);
    C10_MUSA_CHECK(musaEventCreateWithFlags(&event_, flags_));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          kMUSA, reinterpret_cast<uintptr_t>(event_));
    }
    is_created_ = true;
  }

  void moveHelper(MUSAEvent&& other) {
    std::swap(flags_, other.flags_);
    std::swap(is_created_, other.is_created_);
    std::swap(was_recorded_, other.was_recorded_);
    std::swap(device_index_, other.device_index_);
    std::swap(event_, other.event_);
  }
};

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_CORE_MUSAEVENT_H_
