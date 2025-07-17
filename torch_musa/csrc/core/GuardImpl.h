#ifndef TORCH_MUSA_CSRC_CORE_GUARDIMPL_H_
#define TORCH_MUSA_CSRC_CORE_GUARDIMPL_H_

#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>

#include <musa_runtime_api.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10::musa::impl {

using at::musa::kMUSA;

struct MUSAGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  static constexpr DeviceType static_type = kMUSA;

  MUSAGuardImpl() = default;

  explicit MUSAGuardImpl(DeviceType t) {
    TORCH_INTERNAL_ASSERT(t == kMUSA);
  }

  DeviceType type() const override {
    return kMUSA;
  }

  Device exchangeDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == kMUSA);
    const auto old_device_index = ExchangeDevice(d.index());
    return Device(kMUSA, old_device_index);
  }

  Device getDevice() const override {
    DeviceIndex device_id = 0;
    C10_MUSA_CHECK(GetDevice(&device_id));
    return Device(kMUSA, device_id);
  }

  std::optional<Device> uncheckedGetDevice() const noexcept {
    DeviceIndex device_id = -1;
    const auto err = C10_MUSA_ERROR_HANDLED(GetDevice(&device_id));
    TORCH_MUSA_CHECK_WARN(err);
    if (err != musaSuccess) {
      return std::nullopt;
    }
    return Device(kMUSA, device_id);
  }

  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == kMUSA);
    C10_MUSA_CHECK(SetDevice(d.index()));
  }

  void uncheckedSetDevice(Device d) const noexcept override {
    TORCH_MUSA_CHECK_WARN(MaybeSetDevice(d.index()));
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentMUSAStream(d.index()).unwrap();
  }

  Stream getDefaultStream(Device d) const override {
    return getDefaultMUSAStream(d.index());
  }

  Stream getNewStream(Device d, int priority = 0) const override {
    return getStreamFromPool(priority, d.index());
  }

  Stream getStreamFromGlobalPool(Device d, bool isHighPriority = false)
      const override {
    return getStreamFromPool(isHighPriority, d.index());
  }

  Stream exchangeStream(Stream s) const noexcept override {
    MUSAStream ms(s);
    auto old_stream = getCurrentMUSAStream(s.device().index());
    setCurrentMUSAStream(ms);
    return old_stream.unwrap();
  }

  DeviceIndex deviceCount() const noexcept override {
    return device_count();
  }

  bool queryStream(const Stream& stream) const override {
    MUSAStream musa_stream(stream);
    return musa_stream.query();
  }

  void synchronizeStream(const Stream& stream) const override {
    MUSAStream musa_stream(stream);
    musa_stream.synchronize();
  }

  void createEvent(musaEvent_t* musa_event, const EventFlag flag) const {
    auto musa_flag = musaEventDefault;
    switch (flag) {
      case EventFlag::PYTORCH_DEFAULT:
        musa_flag = musaEventDisableTiming;
        break;
      case EventFlag::BACKEND_DEFAULT:
        musa_flag = musaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "MUSA event received unknown flag");
    }

    C10_MUSA_CHECK(musaEventCreateWithFlags(musa_event, musa_flag));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          kMUSA, reinterpret_cast<uintptr_t>(musa_event));
    }
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event) {
      return;
    }
    auto musa_event = static_cast<musaEvent_t>(event);
    DeviceIndex orig_device{-1};
    TORCH_MUSA_CHECK_WARN(GetDevice(&orig_device));
    TORCH_MUSA_CHECK_WARN(SetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          kMUSA, reinterpret_cast<uintptr_t>(musa_event));
    }
    TORCH_MUSA_CHECK_WARN(musaEventDestroy(musa_event));
    TORCH_MUSA_CHECK_WARN(SetDevice(orig_device));
  }

  void block(void* event, const Stream& stream) const override {
    if (!event) {
      return;
    }
    musaEvent_t musa_event = static_cast<musaEvent_t>(event);
    MUSAStream musa_stream{stream};
    const auto orig_device = getDevice();
    setDevice(stream.device());
    C10_MUSA_CHECK(musaStreamWaitEvent(musa_stream, musa_event, 0));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          kMUSA,
          reinterpret_cast<uintptr_t>(musa_event),
          reinterpret_cast<uintptr_t>(musa_stream.stream()));
    }
    setDevice(orig_device);
  }

  bool queryEvent(void* event) const override {
    if (!event) {
      return true;
    }
    musaEvent_t musa_event = static_cast<musaEvent_t>(event);
    const musaError_t err = TORCH_MUSA_ERROR_HANDLE(musaEventQuery(musa_event));
    if (err != musaErrorNotReady) {
      C10_MUSA_CHECK(err);
    } else {
      // ignore and clear the error if not ready
      (void)musaGetLastError();
    }
    return (err == musaSuccess);
  }

  void record(
      void** event,
      const Stream& stream,
      const DeviceIndex device_index,
      const EventFlag flag) const override {
    TORCH_CHECK(
        device_index == -1 || device_index == stream.device_index(),
        "Event device index ",
        device_index,
        " does not match recording stream's device index ",
        stream.device_index(),
        ".");

    musaEvent_t musa_event = static_cast<musaEvent_t>(*event);
    MUSAStream musa_stream{stream};

    // Moves to stream's device to record
    const auto orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!musa_event) {
      createEvent(&musa_event, flag);
    }
    C10_MUSA_CHECK(musaEventRecord(musa_event, musa_stream));
    // Makes the void* point to the (possibly just allocated) MUSA event
    *event = musa_event;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
          kMUSA,
          reinterpret_cast<uintptr_t>(musa_event),
          reinterpret_cast<uintptr_t>(musa_stream.stream()));
    }

    // Resets device
    setDevice(orig_device);
  }

  void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream& stream)
      const override {
    MUSAStream musa_stream{stream};
    MUSACachingAllocator::recordStream(data_ptr, musa_stream);
  }

  void synchronizeEvent(void* event) const override {
    if (!event) {
      return;
    }
    musaEvent_t musa_event = static_cast<musaEvent_t>(event);
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_synchronization(
          kMUSA, reinterpret_cast<uintptr_t>(musa_event));
    }

    C10_MUSA_CHECK(musaEventSynchronize(musa_event));
  }

  double elapsedTime(void* event1, void* event2, const DeviceIndex device_index)
      const override {
    TORCH_CHECK(
        event1 && event2,
        "Both events must be recorded before calculating elapsed time.");

    DeviceIndex orig_device{-1};
    C10_MUSA_CHECK(GetDevice(&orig_device));
    C10_MUSA_CHECK(SetDevice(device_index));
    musaEvent_t musa_event1 = static_cast<musaEvent_t>(event1);
    musaEvent_t musa_event2 = static_cast<musaEvent_t>(event2);
    float time_ms = 0;
    // raise musaErrorNotReady if either event is recorded but not yet completed
    C10_MUSA_CHECK(musaEventElapsedTime(&time_ms, musa_event1, musa_event2));
    C10_MUSA_CHECK(SetDevice(orig_device));
    return static_cast<double>(time_ms);
  }
};

} // namespace c10::musa::impl

#endif // TORCH_MUSA_CSRC_CORE_GUARDIMPL_H_
