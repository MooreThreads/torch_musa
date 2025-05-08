#include "torch_musa/csrc/core/PeerToPeerAccess.h"

#include <vector>

#include <ATen/Context.h>

#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at::musa {

static std::vector<int8_t> p2pAccessEnabled_;
static int64_t num_devices_ = -1;

namespace detail {

void init_p2p_access_cache(int64_t num_devices) {
  // p2pAccessEnabled records if p2p copies are allowed between pairs of
  // devices. Values include "1" (copy allowed), "0" (copy not allowed), and
  // "-1" (unknown).
  p2pAccessEnabled_.clear();
  p2pAccessEnabled_.resize(num_devices * num_devices, -1);
  num_devices_ = num_devices;

  for (const auto i : c10::irange(num_devices)) {
    p2pAccessEnabled_[i * num_devices + i] = 1;
  }
}

} // namespace detail

bool get_p2p_access(int dev, int dev_to_access) {
  at::globalContext().lazyInitPrivateUse1();

  TORCH_INTERNAL_ASSERT(num_devices_ > 0, "p2p access cache not initialized");
  TORCH_CHECK(dev >= 0 && dev < num_devices_, dev, " is not a device");
  TORCH_CHECK(
      dev_to_access >= 0 && dev_to_access < num_devices_,
      dev_to_access,
      " is not a device");

  auto& cache = p2pAccessEnabled_[dev * num_devices_ + dev_to_access];

  if (cache != -1) {
    return cache;
  }

  int access = 0;
  C10_MUSA_CHECK(musaDeviceCanAccessPeer(&access, dev, dev_to_access));
  if (access) {
    cache = 1;
    c10::musa::MUSACachingAllocator::enablePeerAccess(dev, dev_to_access);
  } else {
    cache = 0;
  }
  return static_cast<bool>(cache);
}

} // namespace at::musa
