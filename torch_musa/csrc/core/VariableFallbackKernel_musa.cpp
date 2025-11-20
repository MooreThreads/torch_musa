#include <ATen/core/LegacyTypeDispatch.h>
#include <ATen/core/VariableHooksInterface.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

// Consistent with torch (except for ignoring C10_MOBILE)

namespace {

void autograd_fallback(
    const c10::OperatorHandle& op,
    c10::DispatchKeySet dispatch_keys,
    torch::jit::Stack* stack) {
  // PyTorch has separate builds, some of which don't include autograd.
  // So we define some behavior for when autograd isn't included and
  // go through a layer of indirection (VariableHooksInterface) when it is.
  // See aten/src/ATen/core/VariableHooksInterface.h for more details.
  if (!at::impl::HasVariableHooks()) {
    op.redispatchBoxed(dispatch_keys & c10::after_autograd_keyset, stack);
    return;
  }
  at::impl::GetVariableHooks()->basic_autograd_not_implemented_fallback(
      op, dispatch_keys, stack);
}

#define AUTOGRAD_FALLBACK \
  torch::CppFunction::makeFromBoxedFunction<&autograd_fallback>()

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
  m.fallback(AUTOGRAD_FALLBACK);
}

#undef AUTOGRAD_FALLBACK

} // namespace
