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
  if (n <= randperm_threshold) {
    return at::native::randperm_out_cuda(n, generator, out);
  } else {
    return RandpermOutMusa(n, generator, out);
  }
}

} // namespace musa
} // namespace at
