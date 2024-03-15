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

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

template <bool kReluFused>
void PackedLinearWeightMudnn::apply_impl_helper(
    at::Tensor& quantized_output,
    const at::Tensor& input,
    double output_scale,
    int64_t output_zero_point) {
  if (quantized_output.numel() == 0) {
    return;
  }
  double input_scale = input.q_scale();
  double weight_scale = orig_weight.q_scale();
  int64_t input_zp = input.q_zero_point();
  int64_t weight_zp = orig_weight.q_zero_point();

  auto bias = bias_.has_value()
      ? c10::MaybeOwned<at::Tensor>::borrowed(*bias_)
      : c10::MaybeOwned<at::Tensor>::owned(c10::in_place);

  at::Tensor contig_input;
  at::Tensor contig_weight;
  at::musa::muTensor lmt =
      at::musa::CreateMUTensor(at::musa::ContiguousRef(input, contig_input));
  at::musa::muTensor rmt = at::musa::CreateMUTensor(
      at::musa::ContiguousRef(orig_weight, contig_weight));
  at::musa::muTensor rst = at::musa::CreateMUTensor(quantized_output);
  at::musa::muTensor bmt = bias->defined() ? at::musa::CreateMUTensor(*bias)
                                           : at::musa::CreateEmptyMUTensor();
  at::musa::SetMudnnQuantizationInfo(lmt, input_scale, input_zp);
  at::musa::SetMudnnQuantizationInfo(rmt, weight_scale, weight_zp);
  at::musa::SetMudnnQuantizationInfo(rst, output_scale, output_zero_point);

  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::MatMul mm;
  // quantized linear always accept transpose weight
  CHECK_MUDNN_STATUS(mm.SetTranspose(false, true), "SetTranspose");
  if (bias->defined()) {
    CHECK_MUDNN_STATUS(
        mm.RunWithBiasAdd(h, rst, lmt, rmt, bmt, at::musa::InternalMemAlloc),
        "RunWithBiasAdd");
  } else {
    CHECK_MUDNN_STATUS(
        mm.Run(h, rst, lmt, rmt, at::musa::InternalMemAlloc), "Run");
  }

  // ReLU can also be fused into W8A8 kernel in the future
  if (kReluFused) {
    quantized_output.relu_();
  }
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
  // int8 * int8 -> int8
  TORCH_CHECK(
      act.qscheme() == c10::kPerTensorSymmetric ||
          act.qscheme() == c10::kPerTensorAffine,
      "Expect input data is symmetric quantized");
  TORCH_CHECK(
      act.scalar_type() == c10::kQInt8,
      "Expected input data dtype ",
      toString(c10::kQInt8),
      " but got ",
      toString(act.scalar_type()));
  TORCH_CHECK(
      act.dim() == 2 || act.dim() == 3,
      "Dimension of input tensor should be 2 or 3");

  std::vector<int64_t> original_output_shape{act.sizes().vec()};
  original_output_shape.back() = orig_weight.size(0); // output channels
  std::vector<int64_t> output_shape = original_output_shape;
  // expects tensors to be 2D.
  // if the tensors are 3D, reduce the 1st and 2nd dim
  if (output_shape.size() == 3) {
    output_shape[1] *= output_shape[0];
    output_shape.erase(output_shape.begin());
  }
  at::Tensor quantized_output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(at::ScalarType::QInt8),
      output_scale,
      output_zero_point);
  // expect tensors to be 2D, if act is a 3D tensor, we create a 2D view
  std::vector<int64_t> new_sizes(act.sizes().vec());
  // expect leading dimensions to be the dummy dimensions
  if (new_sizes.size() == 3) {
    new_sizes[1] *= new_sizes[0];
    new_sizes.erase(new_sizes.begin());
  }
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
    if (kReluFused) {
      return packed_weight->apply_relu(act, output_scale, output_zero_point);
    } else {
      return packed_weight->apply(act, output_scale, output_zero_point);
    }
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear"),
      TORCH_FN(QLinearInt8<false>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_relu"),
      TORCH_FN(QLinearInt8<true>::run));
}

} // namespace
} // namespace musa
} // namespace at
