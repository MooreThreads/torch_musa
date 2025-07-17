#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/NamedTensorUtils.h>
#include <c10/util/Exception.h>

#include <ATen/native/cpu/ChannelShuffleKernel.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/channel_shuffle_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/native_channel_shuffle.h>
#include <ATen/ops/native_channel_shuffle_native.h>
#endif

#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

Tensor MathChannelShuffle(const Tensor& self, int64_t groups) {
  c10::musa::MUSAGuard device_guard(self.device());
  int64_t b = self.size(0);
  int64_t c = self.size(1);
  int64_t oc = c / groups;

  auto input_reshaped = self.view({b, groups, oc, -1});
  // TODO: contiguous can be made to preserve the memory format
  // of the input. However since the above reshape clobbers h and w
  // it may not be safe to do that, since channels_last contiguous
  // may think oc and and the last dim correspond to h,w?
  // It is not clear, however from initial looking around it feels that
  // this may not be correct.
  // In this case channels last will likely require custom implementation
  // if we want to preserve the memory order.
  // XNNPACK has channel shuffle op for NHWC. For mobile usecase this is good.
  // For server we will have to do a custom implementation.
  // For ChannelsFirst, a.k.a Contiguous, memory format we will also need
  // a fast custom implementation perhaps.
  Tensor output_tensor =
      input_reshaped.permute({0 /* b */, 2 /* oc */, 1 /* groups */, 3})
          .contiguous()
          .reshape(self.sizes());
  return namedinference::propagate_names_if_nonempty(
      output_tensor, self.has_names() ? self.names() : at::ArrayRef<Dimname>{});
}

} // namespace musa
} // namespace at