#include "torch_musa/csrc/core/MUSAHooksInterface.h"

#include <c10/util/CallOnce.h>

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

} // namespace at
