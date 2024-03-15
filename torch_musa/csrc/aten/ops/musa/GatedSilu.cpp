#include <ATen/ATen.h>
#include <torch/library.h>
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

at::Tensor gated_silu(const at::Tensor& input);

ADVANCED_REGISTER(aten, PrivateUse1, "gated_silu", gated_silu)

} // namespace musa
} // namespace at
