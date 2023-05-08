#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <deque>
#include <mutex>
#include <vector>

#include <ATen/musa/MUSAConfig.h>
#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace at {
namespace musa {
namespace {

DeviceIndex num_gpus = -1;
c10::once_flag init_flag;
std::deque<c10::once_flag> device_flags;
std::vector<musaDeviceProp> device_properties;

void initMUSAContextVectors() {
  num_gpus = c10::musa::device_count();
  device_flags.resize(num_gpus);
  device_properties.resize(num_gpus);
}

void initDeviceProperty(DeviceIndex device_index) {
  musaDeviceProp device_prop;
  TORCH_MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_index));
  device_properties[device_index] = device_prop;
}

} // anonymous namespace

/* Device info */
int warp_size() {
  return getCurrentDeviceProperties()->warpSize;
}

musaDeviceProp* getCurrentDeviceProperties() {
  auto device = ::c10::musa::current_device();
  return ::c10::musa::getDeviceProperties(device);
}

Allocator* getMUSADeviceAllocator() {
  return ::c10::musa::MUSACachingAllocator::get();
}

} // namespace musa
} // namespace at
