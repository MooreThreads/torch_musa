#include "torch_musa/csrc/core/MUSAHooks.h"

#include <ATen/DeviceGuard.h>

#include "musa_runtime_api.h"
#include "musart_version.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"
#include "torch_musa/csrc/core/PinnedMemoryAllocator.h"

namespace at::musa::detail {

void MUSAHooks::initMUSA() const {
  C10_LOG_API_USAGE_ONCE("torch_musa.init");
  const auto num_devices = c10::musa::device_count_ensure_non_zero();
  c10::musa::MUSACachingAllocator::init(num_devices);
  at::musa::detail::init_p2p_access_cache(num_devices);
}

bool MUSAHooks::hasPrimaryContext(DeviceIndex device_index) const {
  return c10::musa::hasPrimaryContext(device_index);
}

int MUSAHooks::getNumGPUs() const {
  return static_cast<int>(c10::musa::device_count());
}

void MUSAHooks::setCurrentDevice(DeviceIndex device) const {
  c10::musa::set_device(device);
}

DeviceIndex MUSAHooks::current_device() const {
  return c10::musa::current_device();
}

DeviceIndex MUSAHooks::exchangeDevice(DeviceIndex device) const {
  return c10::musa::ExchangeDevice(device);
}

DeviceIndex MUSAHooks::maybeExchangeDevice(DeviceIndex device) const {
  return c10::musa::MaybeExchangeDevice(device);
}

bool MUSAHooks::isPinnedPtr(const void* data) const {
  return c10::musa::isPinnedPtr(data);
}

Allocator* MUSAHooks::getPinnedMemoryAllocator() const {
  return at::musa::getPinnedMemoryAllocator();
}

const Generator& MUSAHooks::getDefaultMUSAGenerator(
    DeviceIndex device_index) const {
  return at::musa::detail::getDefaultMUSAGenerator(device_index);
}

Generator MUSAHooks::getNewGenerator(DeviceIndex device_index) const {
  return make_generator<at::MUSAGeneratorImpl>(device_index);
}

Device MUSAHooks::getDeviceFromPtr(void* data) const {
  return c10::musa::getDeviceFromPtr(data);
}

void MUSAHooks::resizeMUSABytes(const Storage& storage, size_t newsize) const {
  ptrdiff_t size_bytes_i = static_cast<ptrdiff_t>(newsize);
  TORCH_CHECK(
      !c10::overflows<size_t>(size_bytes_i),
      "Requested storage size (",
      size_bytes_i,
      ") cannot be represented as a size_t");
  at::musa::resize_bytes_musa(storage.unsafeGetStorageImpl(), newsize);
}

bool MUSAHooks::hasMUSA() const {
  return c10::musa::is_musa_available();
}

void MUSAHooks::deviceSynchronize(DeviceIndex device_index) const {
  at::DeviceGuard device_guard(at::Device(at::musa::kMUSA, device_index));
  c10::musa::Synchronize();
}

bool MUSAHooks::hasMUSART() const {
#ifdef MUSART_VERSION
  return true;
#else
  return false;
#endif
}

using at::MUSAHooksRegistry;
using at::RegistererMUSAHooksRegistry;

REGISTER_MUSA_HOOKS(MUSAHooks);

} // namespace at::musa::detail
