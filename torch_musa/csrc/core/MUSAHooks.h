#ifndef TORCH_MUSA_CSRC_CORE_MUSAHOOKS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAHOOKS_H_

#include "torch_musa/csrc/core/MUSAHooksInterface.h"

namespace at::musa::detail {

struct MUSAHooks : MUSAHooksInterface {
  MUSAHooks(MUSAHooksArgs) {}

  ~MUSAHooks() override = default;

  void initMUSA() const override;

  bool hasPrimaryContext(DeviceIndex device_index) const override;

  int getNumGPUs() const override;

  void setCurrentDevice(DeviceIndex device) const override;

  DeviceIndex current_device() const override;

  DeviceIndex exchangeDevice(DeviceIndex device) const override;

  DeviceIndex maybeExchangeDevice(DeviceIndex device) const override;

  bool isPinnedPtr(const void* data) const override;

  Allocator* getPinnedMemoryAllocator() const override;

  const Generator& getDefaultMUSAGenerator(
      DeviceIndex device_index) const override;

  Device getDeviceFromPtr(void* data) const override;

  void resizeMUSABytes(const Storage& storage, size_t newsize) const override;

  bool hasMUSA() const override;

  void deviceSynchronize(DeviceIndex device_index) const override;
};

} // namespace at::musa::detail

#endif // TORCH_MUSA_CSRC_CORE_MUSAHOOKS_H_
