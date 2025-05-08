#ifndef TORCH_MUSA_CSRC_CORE_MUSAHOOKSINTERFACE_H_
#define TORCH_MUSA_CSRC_CORE_MUSAHOOKSINTERFACE_H_

#include <ATen/detail/PrivateUse1HooksInterface.h>
#include <c10/util/Registry.h>

namespace at {

struct MUSAHooksInterface : PrivateUse1HooksInterface {
  ~MUSAHooksInterface() override = default;

  virtual void initMUSA() const {
    TORCH_CHECK(false, "Cannot initialize MUSA without torch_musa library.");
  }
  void initPrivateUse1() const override {
    initMUSA();
  }

  bool hasPrimaryContext(DeviceIndex device_index) const override {
    TORCH_CHECK(
        false,
        "Cannot call hasPrimaryContext(",
        device_index,
        ") without torch_musa library.");
  }

  virtual const Generator& getDefaultMUSAGenerator(
      DeviceIndex device_index) const {
    TORCH_CHECK(
        false, "Cannot get default MUSA generator without torch_musa library.");
  }
  const Generator& getDefaultGenerator(
      DeviceIndex device_index) const override {
    return getDefaultMUSAGenerator(device_index);
  }

  Device getDeviceFromPtr(void* data) const override {
    TORCH_CHECK(
        false,
        "Cannot get device of pointer on MUSA without torch_musa library.");
  }

  Allocator* getPinnedMemoryAllocator() const override {
    TORCH_CHECK(false, "Pinned memory requires MUSA.");
  }

  virtual void resizeMUSABytes(const Storage& storage, size_t newsize) const {
    TORCH_CHECK(false, "Resize bytes requires MUSA.");
  }
  void resizePrivateUse1Bytes(const Storage& storage, size_t newsize)
      const override {
    resizeMUSABytes(storage, newsize);
  }

  virtual bool hasMUSA() const {
    return false;
  }

  virtual DeviceIndex current_device() const {
    return -1;
  }
  DeviceIndex getCurrentDevice() const override {
    return current_device();
  }

  virtual int getNumGPUs() const {
    return 0;
  }
  DeviceIndex deviceCount() const override {
    return static_cast<DeviceIndex>(getNumGPUs());
  }

  virtual void deviceSynchronize(DeviceIndex device_index) const {
    TORCH_CHECK(
        false, "Cannot synchronize MUSA device without torch_musa library.");
  }
};

struct MUSAHooksArgs : PrivateUse1HooksArgs {};

TORCH_DECLARE_REGISTRY(MUSAHooksRegistry, MUSAHooksInterface, MUSAHooksArgs);
#define REGISTER_MUSA_HOOKS(clsname) \
  C10_REGISTER_CLASS(MUSAHooksRegistry, clsname, clsname)

namespace detail {
const MUSAHooksInterface& getMUSAHooks();
} // namespace detail
} // namespace at

#endif // TORCH_MUSA_CSRC_CORE_MUSAHOOKSINTERFACE_H_
