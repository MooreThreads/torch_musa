#include "torch_musa/csrc/core/Device.h"

#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace c10::musa {

DeviceIndex current_device() {
  DeviceIndex cur_device = -1;
  C10_MUSA_CHECK(GetDevice(&cur_device));
  return cur_device;
}

void set_device(DeviceIndex device) {
  C10_MUSA_CHECK(SetDevice(device));
}

void Synchronize() {
  device_synchronize();
}

} // namespace c10::musa
