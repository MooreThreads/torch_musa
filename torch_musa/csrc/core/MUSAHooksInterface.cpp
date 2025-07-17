#include "torch_musa/csrc/core/MUSAHooksInterface.h"

#include <c10/util/CallOnce.h>

namespace at::musa {
// defined in torch_musa/csrc/aten/ops/TensorFactory.cpp
void resize_bytes_musa(StorageImpl* storage, size_t size_bytes);
} // namespace at::musa

namespace at {
namespace detail {

static MUSAHooksInterface* musa_hooks = nullptr;

const MUSAHooksInterface& getMUSAHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    musa_hooks =
        MUSAHooksRegistry()->Create("MUSAHooks", MUSAHooksArgs{}).release();
    if (!musa_hooks) {
      musa_hooks = new MUSAHooksInterface();
    }
    RegisterPrivateUse1HooksInterface(musa_hooks);
  });
  return *musa_hooks;
}

} // namespace detail

C10_DEFINE_REGISTRY(MUSAHooksRegistry, MUSAHooksInterface, MUSAHooksArgs)

void MUSAHooksInterface::resizePrivateUse1Bytes(
    const Storage& storage,
    size_t newsize) const {
  ptrdiff_t size_bytes_i = newsize;
  TORCH_CHECK(
      !c10::overflows<size_t>(size_bytes_i),
      "Requested storage size (",
      size_bytes_i,
      ") cannot be represented as a size_t");
  const auto size_bytes = static_cast<size_t>(size_bytes_i);
  at::musa::resize_bytes_musa(storage.unsafeGetStorageImpl(), size_bytes);
}

} // namespace at
