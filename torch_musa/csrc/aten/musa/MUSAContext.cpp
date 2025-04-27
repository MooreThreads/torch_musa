#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <deque>
#include <mutex>
#include <vector>

#include <ATen/musa/MUSAConfig.h>
#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
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
  auto device = c10::musa::current_device();
  return getDeviceProperties(device);
}

musaDeviceProp* getDeviceProperties(int device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  c10::call_once(device_flags[device], initDeviceProperty, device);
  return &device_properties[device];
}

bool canDeviceAccessPeer(int device, int peer_device) {
  c10::call_once(init_flag, initMUSAContextVectors);
  if (device == -1)
    device = current_device();
  AT_ASSERT(device >= 0 && device < num_gpus);
  AT_ASSERT(peer_device >= 0 && peer_device < num_gpus);
  int can_access = 0;
  TORCH_MUSA_CHECK(musaDeviceCanAccessPeer(&can_access, device, peer_device));
  return can_access != 0;
}

Allocator* getMUSADeviceAllocator() {
  return c10::musa::MUSACachingAllocator::get();
}

uint32_t getMUSAArch() {
  // same value as __MUSA_ARCH__ but used on host code
  const musaDeviceProp* device_prop = getCurrentDeviceProperties();
  return device_prop->major * 100 + device_prop->minor * 10;
}

uint32_t getMUSAArch(int device) {
  const musaDeviceProp* device_prop = getDeviceProperties(device);
  return device_prop->major * 100 + device_prop->minor * 10;
}

// start to support real bfloat16 op since QY2
bool maybeDNNOpSupportBFloat16() {
  return getMUSAArch() >= 220;
}

bool maybeDNNOpSupportBFloat16(int device) {
  return getMUSAArch(device) >= 220;
}

} // namespace musa
} // namespace at
