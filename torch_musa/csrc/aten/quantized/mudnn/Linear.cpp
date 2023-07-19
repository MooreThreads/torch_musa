#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/ops/add.h>
#include <ATen/ops/addmm.h>
#include <ATen/ops/matmul.h>
#include <c10/core/ScalarType.h>
#include <c10/util/MaybeOwned.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

// mudnn currently doesn't support quantization linear (matmul), so we have to
// dequantize weight and activation, run matmul with FP32 dtype
template <bool kReluFused>
void PackedLinearWeightMudnn::apply_impl_helper(
    at::Tensor& quantized_output,
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  if (quantized_output.numel() == 0) {
    return;
  }

  // we only support per-tensor quantization for linear operation
  TORCH_CHECK(
      input.qscheme() == at::kPerTensorAffine ||
          input.qscheme() == at::kPerTensorSymmetric,
      "Linear only supports per-tensor quantized activation");
  TORCH_CHECK(
      orig_weight.qscheme() == at::kPerTensorAffine ||
          orig_weight.qscheme() == at::kPerTensorSymmetric,
      "Linear only supports per-tensor quantized weight");

  // dequantize activation and weight to float
  at::Tensor input_ = at::musa::DequantizeQuantized(input);
  at::Tensor weight_ = at::musa::DequantizeQuantized(orig_weight);
  at::Tensor result_;

  auto bias = bias_.has_value()
      ? c10::MaybeOwned<at::Tensor>::borrowed(*bias_)
      : c10::MaybeOwned<at::Tensor>::owned(c10::in_place);

  if (input_.dim() == 2 && bias->defined()) {
    result_ = at::addmm(*bias, input_, weight_);
  } else if (input_.dim() == 3 && bias->defined() && input_.is_contiguous()) {
    const auto input_sizes = input_.sym_sizes();
    result_ = at::addmm(
        *bias,
        input_.view_symint({input_sizes[0] * input_sizes[1], input_sizes[2]}),
        weight_.t());
    result_ = result_.view_symint(
        {input_sizes[0], input_sizes[1], result_.sym_size(1)});
  } else {
    result_ = at::matmul(input_, weight_.t());
    if (bias->defined()) {
      if (isTensorSubclassLike(*bias) ||
          bias->_fw_grad(/*level*/ 0).defined()) {
        result_ = at::add(result_, *bias);
      } else {
        result_.add_(*bias);
      }
    }
  }
  // ReLU can also be fused into W8A8 kernel in the future
  if (kReluFused) {
    result_.relu_();
  }

  // quantize result to QUInt8 format
  result_ = at::musa::QuantizePerTensor(
      result_, output_scale, output_zero_point, at::ScalarType::QUInt8);
  quantized_output.copy_(result_);
  return;
}

// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
// Numerics are the same as conv (see aten/src/ATen/native/quantized/Conv.cpp):
template <bool kReluFused>
at::Tensor PackedLinearWeightMudnn::apply_impl(
    const at::Tensor& act,
    double output_scale,
    int64_t output_zero_point) {
  // uint8 * int8 -> uint8
  TORCH_CHECK(
      act.scalar_type() == c10::kQUInt8,
      "Expected input data dtype ",
      toString(c10::kQUInt8),
      " but got ",
      toString(act.scalar_type()));
  TORCH_CHECK(
      act.dim() >= 2,
      "Dimension of input tensor should be greater or equal to 2");

  std::vector<int64_t> original_output_shape{act.sizes().vec()}; // 2D
  original_output_shape.back() = orig_weight.size(0); // output channels
  // expects tensors to be at least 3D.
  std::vector<int64_t> output_shape(3, 1);
  output_shape[1] = original_output_shape[0];
  output_shape[2] = original_output_shape[1];
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(at::ScalarType::QUInt8),
      output_scale,
      output_zero_point);
  // expect tensors to be at least 3D. act is currently 2D. we will create a 3D
  // view
  std::vector<int64_t> new_sizes(3, 1);
  // expect leading dimensions to be the dummy dimensions
  new_sizes.back() = act.sizes().back();
  new_sizes[1] = act.size(0);
  apply_impl_helper<kReluFused>(
      quantized_output, act.view(new_sizes), output_scale, output_zero_point);
  return quantized_output.view(original_output_shape);
}

at::Tensor PackedLinearWeightMudnn::apply(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<false>(input, output_scale, output_zero_point);
}

at::Tensor PackedLinearWeightMudnn::apply_relu(
    at::Tensor input,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<true>(input, output_scale, output_zero_point);
}

namespace at {
namespace musa {
namespace {

template <bool kReluFused>
class QLinearInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      const c10::intrusive_ptr<LinearPackedParamsBase>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    c10::musa::MUSAGuard device_guard(act.device());
    // TODO(@fan.mo): check all zero_points are zero/all tensors are
    // symmetrically quantized
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, AutogradPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::linear"), QLinearInt8<false>::run);
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu"), QLinearInt8<true>::run);
}

} // namespace
} // namespace musa
} // namespace at
