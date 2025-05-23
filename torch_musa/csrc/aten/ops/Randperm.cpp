#include <ATen/ATen.h>
#include <torch/torch.h>

#include "torch_musa/csrc/aten/ops/Randperm.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

at::Tensor& RandpermGeneratorOut(
    int64_t n,
    c10::optional<at::Generator> generator,
    at::Tensor& out) {
  const OptionalDeviceGuard device_guard(device_of(out));
  constexpr int64_t randperm_threshold = 256;
  const bool is_support_dtype = out.scalar_type() == at::ScalarType::Int ||
      out.scalar_type() == at::ScalarType::Long;
  if (n > randperm_threshold && is_support_dtype) {
    return RandpermOutMusa(n, generator, out);
  } else {
    return at::native::randperm_out_cuda(n, generator, out);
  }
}

} // namespace musa
} // namespace at
