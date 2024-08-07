#ifndef TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <c10/core/impl/GPUTrace.h>

#include "musa_runtime_api.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10 {
namespace musa {
using at::musa::kMUSA;
using c10::Device;
using c10::DeviceType;
using c10::Stream;

namespace impl {

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
    Device old_device = getDevice();
    if (old_device.index() != d.index()) {
      TORCH_MUSA_CHECK(musaSetDevice(d.index()));
    }
    return old_device;
  }
  Device getDevice() const override {
    int device;
    TORCH_MUSA_CHECK(musaGetDevice(&device));
    return Device(kMUSA, device);
  }
  c10::optional<Device> uncheckedGetDevice() const noexcept {
    int device;
    const auto err = TORCH_MUSA_ERROR_HANDLE(musaGetDevice(&device));
    TORCH_MUSA_CHECK_WARN(err);
    if (err != musaSuccess) {
      return c10::nullopt;
    }
    return Device(kMUSA, device);
  }

  bool hasPrimaryContext(int64_t device_index) const {
    TORCH_CHECK(
        device_index >= 0 && device_index < deviceCount(),
        "hasPrimaryContext expects a valid device index, but got device_index=",
        device_index);
    unsigned int ctx_flags;
    int ctx_is_active = 0;
    muDevicePrimaryCtxGetState(device_index, &ctx_flags, &ctx_is_active);
    return ctx_is_active == 1;
  }

  void setDevice(Device d) const override {
    if (!hasPrimaryContext(d.index())) {
      // To keep consistent logic with `Engine::thread_init`
      // in pytorch/torch/csrc/autograd/engine.cpp
      return;
    }

    TORCH_INTERNAL_ASSERT(d.type() == kMUSA);
    Device current_device = getDevice();
    if (current_device != d) {
      TORCH_MUSA_CHECK(musaSetDevice(d.index()));
    }
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    auto current_device = uncheckedGetDevice();
    if (!current_device.has_value() || current_device.value() != d) {
      TORCH_MUSA_CHECK_WARN(musaSetDevice(d.index()));
    }
  }

  Stream getStream(Device d) const noexcept override {
    return getCurrentMUSAStream(d.index()).unwrap();
  }

  Stream getDefaultStream(Device d) const override {
    return getDefaultMUSAStream(d.index());
  }

  Stream getStreamFromGlobalPool(Device d, bool high_priority = false)
      const override {
    return getStreamFromPool(high_priority, d.index());
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

  void createEvent(musaEvent_t* musa_event, const c10::EventFlag flag) const {
    auto musa_flag = musaEventDefault;
    switch (flag) {
      case c10::EventFlag::PYTORCH_DEFAULT:
        // c10::EventFlag defined CUDA_EVENT_DISABLE_TIME,
        // we don't need this enum, so just use PYTORCH_DEFAULT.
        musa_flag = musaEventDisableTiming;
        break;
      case c10::EventFlag::BACKEND_DEFAULT:
        // c10::EventFlag defined CUDA_EVENT_DEFAULT,
        // we don't need this enum, so just use BACKEND_DEFAULT.
        musa_flag = musaEventDefault;
        break;
      default:
        TORCH_CHECK(false, "MUSA event received unknown flag");
    }
    TORCH_MUSA_CHECK(musaEventCreateWithFlags(musa_event, musa_flag));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_creation(
          reinterpret_cast<uintptr_t>(musa_event));
    }
  }

  void destroyEvent(void* event, const DeviceIndex device_index)
      const noexcept override {
    if (!event)
      return;
    auto musa_event = static_cast<musaEvent_t>(event);
    int orig_device;
    TORCH_MUSA_CHECK_WARN(musaGetDevice(&orig_device));
    TORCH_MUSA_CHECK_WARN(musaSetDevice(device_index));
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_deletion(
          reinterpret_cast<uintptr_t>(musa_event));
    }
    TORCH_MUSA_CHECK_WARN(musaEventDestroy(musa_event));
    TORCH_MUSA_CHECK_WARN(musaSetDevice(orig_device));
  }

  void block(void* event, const Stream& stream) const override {
    if (!event)
      return;
    musaEvent_t musa_event = static_cast<musaEvent_t>(event);
    MUSAStream musa_stream{stream};
    const Device orig_device = getDevice();
    setDevice(stream.device());
    TORCH_MUSA_CHECK(
        musaStreamWaitEvent(musa_stream, musa_event, 0)); // TODO:check musa API
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_wait(
          reinterpret_cast<uintptr_t>(musa_event),
          reinterpret_cast<uintptr_t>(musa_stream.stream()));
    }
    setDevice(orig_device);
  }

  bool queryEvent(void* event) const override {
    if (!event)
      return true;
    musaEvent_t musa_event = static_cast<musaEvent_t>(event);
    const musaError_t err = TORCH_MUSA_ERROR_HANDLE(musaEventQuery(musa_event));
    if (err != musaErrorNotReady) {
      TORCH_MUSA_CHECK(err);
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
      const c10::EventFlag flag) const override {
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
    const Device orig_device = getDevice();
    setDevice(stream.device());

    // Creates the event (lazily)
    if (!musa_event)
      createEvent(&musa_event, flag);
    TORCH_MUSA_CHECK(musaEventRecord(musa_event, musa_stream));
    // Makes the void* point to the (possibly just allocated) MUSA event
    *event = musa_event;
    const c10::impl::PyInterpreter* interp = c10::impl::GPUTrace::get_trace();
    if (C10_UNLIKELY(interp)) {
      (*interp)->trace_gpu_event_record(
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
};
} // namespace impl
} // namespace musa
} // namespace c10

#endif // TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_
