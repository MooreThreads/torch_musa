#include <ATen/ATen.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
namespace {

template <bool kReluFused>
at::Tensor QAddTensor(
    const at::Tensor& qa,
    const at::Tensor& qb,
    const at::Scalar& alpha) {
  // This function is a little-bit tricky, it's used from quantized model graph
  // that it takes one quantized-tensor and one float-tensor in, then outputs a
  // float- tensor
  TORCH_CHECK(alpha.equal(1), "Quantized add don't support set alpha yet.");
  at::Tensor fa = qa.is_quantized() ? qa.dequantize() : qa;
  at::Tensor fb = qb.is_quantized() ? qb.dequantize() : qb;
  at::Tensor out = at::add(fa, fb, alpha);
  if (kReluFused) {
    out.relu_();
  }
  return out;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("add.Tensor", TORCH_FN(QAddTensor</*ReLUFused=*/false>));
}

} // namespace
} // namespace musa
} // namespace at
