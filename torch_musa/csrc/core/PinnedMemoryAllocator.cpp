#include <ATen/ATen.h>
#include <ATen/CPUFunctions.h>
#include <ATen/Config.h>
#include <ATen/Context.h>
#include <ATen/TensorUtils.h>
#include <c10/core/Storage.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/PinnedMemoryAllocator.h"

namespace at {
namespace musa {

bool IsPinnedMusa(const Tensor& self, c10::optional<Device> device) {
  // First check if driver is broken/missing, in which case PyTorch CPU
  // functionalities should still work, we should report `false` here.
  if (!at::musa::device_count()) {
    return false;
  }
  // musaPointerGetAttributes grabs context on the current device, so we set
  // device to one that already has context, if exists.
  at::OptionalDeviceGuard device_guard;
  auto primary_ctx_device_index = c10::musa::getDeviceIndexWithPrimaryContext();
  // device_guard can only be initialized by follow ctx, it must has_value,
  // or device_guard will call Destructor with invalid property.
  assert(primary_ctx_device_index.has_value());
  device_guard.reset_device(
      at::Device(at::musa::kMUSA, *primary_ctx_device_index));
  musaPointerAttributes attr;
  // We do not believe that MUSA needs mutable access to the data here.
  const void* data = self.storage().data();
  musaError_t err = musaPointerGetAttributes(&attr, const_cast<void*>(data));
  if (err == musaErrorInvalidValue) {
    (void)musaGetLastError(); // clear MUSA error
    return false;
  }
  TORCH_MUSA_CHECK(err);
  return attr.type == musaMemoryTypeHost;
}

Tensor PinMemoryMusa(const Tensor& self, c10::optional<Device> device) {
  c10::musa::OptionalMUSAGuard device_guard;
  if (device.has_value()) {
    device_guard.set_device(device.value());
  }
  auto* allocator = at::musa::getPinnedMemoryAllocator();
  auto storage = Storage(
      Storage::use_byte_size_t(),
      detail::computeStorageNbytes(
          self.sizes(), self.strides(), self.dtype().itemsize()),
      allocator,
      /*resizable=*/false);
  auto tensor = at::cpu::empty({0}, self.options())
                    .set_(storage, 0, self.sizes(), self.strides());
  tensor.copy_(self);
  return tensor;
}

} // namespace musa
} // namespace at
