#include <ATen/ATen.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
namespace {

inline void CheckInputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
  TORCH_CHECK(
      qa.scalar_type() == c10::kQUInt8 || qa.scalar_type() == c10::kQInt8,
      "Add operands support QUint8 and QInt8, now got ",
      toString(qa.scalar_type()));
}

// TODO(@fan.mo): this could be implemented as:
// relu((a_int8 + b_int8 * (b_scale/a_scale))) * (a_scale / out_scale)
// but mudnn doesn't support int8 add and relu, so we have to dequantize
// inputs first, would open a jira request for compute team.
template <bool kReluFused = false>
Tensor QAdd(
    Tensor qa,
    Tensor qb,
    double output_scale,
    int64_t output_zero_point) {
  c10::musa::MUSAGuard device_guard(qa.device());
  if (qa.numel() == 0) {
    return Tensor{};
  }

  // TODO(@fan.mo): add shape checking when broadcasted add is supported. For
  // now we assume the input tensors are the same shape
  TORCH_CHECK(
      qa.sizes() == qb.sizes(),
      "Quantized mudnn add currently expects both input tensors to be the same shape");

  CheckInputs(qa, qb);

  // TODO(fan.mo): these dequantize would increase overhead
  at::Tensor fa = qa.dequantize();
  fa.add_(qb.dequantize());

  if (kReluFused) {
    fa.relu_();
  }

  Tensor quantized_output = at::musa::QuantizePerTensor(
      fa, output_scale, output_zero_point, qa.scalar_type());
  return quantized_output;
}

TORCH_LIBRARY_IMPL(quantized, AutogradPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"), TORCH_FN(QAdd<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"), TORCH_FN(QAdd<true>));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"), TORCH_FN(QAdd<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"), TORCH_FN(QAdd<true>));
}

} // namespace
} // namespace musa
} // namespace at
