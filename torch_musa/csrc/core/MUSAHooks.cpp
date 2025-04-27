#include <ATen/DeviceGuard.h>

#include "musa_runtime_api.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAHooks.h"
#include "torch_musa/csrc/core/PeerToPeerAccess.h"

namespace at {
namespace musa {
namespace detail {

void MUSAHooks::initMUSA() const {
  C10_LOG_API_USAGE_ONCE("torch_musa.init");
  const int64_t num_devices = c10::musa::device_count();
  c10::musa::MUSACachingAllocator::init(num_devices);
  at::musa::detail::init_p2p_access_cache(num_devices);
}

bool MUSAHooks::hasMUSA() const {
  return c10::musa::device_count() > 0;
}

const Generator& MUSAHooks::getDefaultMUSAGenerator(
    DeviceIndex device_index) const {
  return at::musa::detail::getDefaultMUSAGenerator(device_index);
}

Device MUSAHooks::getDeviceFromPtr(void* ptr) const {
  musaPointerAttributes attr{};
  TORCH_MUSA_CHECK(musaPointerGetAttributes(&attr, ptr));
  return {at::musa::kMUSA, static_cast<DeviceIndex>(attr.device)};
}

int64_t MUSAHooks::current_device() const {
  return c10::musa::current_device();
}

int MUSAHooks::getNumGPUs() const {
  return c10::musa::device_count();
}

void MUSAHooks::deviceSynchronize(int64_t device_index) const {
  at::DeviceGuard device_guard(at::Device(at::musa::kMUSA, device_index));
  c10::musa::Synchronize();
}

using at::MUSAHooksRegistry;
using at::RegistererMUSAHooksRegistry;

REGISTER_MUSA_HOOKS(MUSAHooks);

} // namespace detail
} // namespace musa
} // namespace at
