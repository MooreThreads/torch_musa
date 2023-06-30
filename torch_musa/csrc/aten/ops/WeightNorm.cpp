#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
::std::tuple<at::Tensor, at::Tensor> _WeightNormInterface(
    const at::Tensor& v,
    const at::Tensor& g,
    int64_t dim) {
  c10::musa::MUSAGuard device_guard(v.device());
  return at::native::weight_norm_cuda(v, g, dim);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_weight_norm_interface", &_WeightNormInterface);
}

} // namespace musa
} // namespace at
