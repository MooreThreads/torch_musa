#include <ATen/ATen.h>
#include <torch/library.h>

namespace at {
namespace musa {

at::Tensor gated_silu(const at::Tensor& input);

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("gated_silu", &gated_silu);
}

} // namespace musa
} // namespace at
