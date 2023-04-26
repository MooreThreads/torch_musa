#ifndef TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_
#define TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_

#include <c10/core/DeviceGuard.h>
#include <c10/core/Stream.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>

#include "musa_runtime_api.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace torch_musa {
using at::native::musa::kMUSA;
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
    TORCH_MUSA_WARN(err);
    if (err != musaSuccess) {
      return c10::nullopt;
    }
    return Device(kMUSA, device);
  }
  void setDevice(Device d) const override {
    TORCH_INTERNAL_ASSERT(d.type() == kMUSA);
    Device current_device = getDevice();
    if (current_device != d) {
      TORCH_MUSA_CHECK(musaSetDevice(d.index()));
    }
  }
  void uncheckedSetDevice(Device d) const noexcept override {
    auto current_device = uncheckedGetDevice();
    if (!current_device.has_value() || current_device.value() != d) {
      TORCH_MUSA_WARN(musaSetDevice(d.index()));
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

  // TODO(Xiaokang Shang): Event

  // TODO(Xiaokang Shang): CachineAllocator related API.
  // void recordDataPtrOnStream(const c10::DataPtr& data_ptr, const Stream&
  // stream);
};
} // namespace impl
} // namespace torch_musa

#endif // TORCH_MUSA_CSRC_CORE_MUSA_GUARDIMPL_H_
