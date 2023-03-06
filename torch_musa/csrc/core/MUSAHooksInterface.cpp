#include "torch_musa/csrc/core/MUSAHooksInterface.h"
#include <c10/util/CallOnce.h>
#include <c10/util/Exception.h>

namespace at {
namespace detail {

static MUSAHooksInterface* musa_hooks = nullptr;

static at::MUSAHooksInterface* get_private_hooks() {
  return musa_hooks;
}

void RegisterHook() {
  at::RegisterPrivateUse1HooksInterface(get_private_hooks());
}

const MUSAHooksInterface& getMUSAHooks() {
  static c10::once_flag once;
  c10::call_once(once, [] {
    musa_hooks =
        MUSAHooksRegistry()->Create("MUSAHooks", MUSAHooksArgs{}).release();
    if (!musa_hooks) {
      musa_hooks = new MUSAHooksInterface();
    }
    RegisterHook();
  });
  return *musa_hooks;
}
} // namespace detail

C10_DEFINE_REGISTRY(MUSAHooksRegistry, MUSAHooksInterface, MUSAHooksArgs)

} // namespace at
