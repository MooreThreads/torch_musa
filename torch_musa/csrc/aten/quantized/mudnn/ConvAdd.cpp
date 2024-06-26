#include <c10/util/ArrayRef.h>

#include <ATen/ATen.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/quantized/PackedParams.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/mudnn/Conv.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_add(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::IDENTITY>(
      input, accum, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_add_relu(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::RELU>(
      input, accum, output_scale, output_zero_point);
}

template <int kSpatialDim>
at::Tensor PackedConvWeightMudnn<kSpatialDim>::apply_add_silu(
    const at::Tensor& input,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point) {
  return apply_impl<ActMode::SILU>(
      input, accum, output_scale, output_zero_point);
}

template at::Tensor PackedConvWeightMudnn<2>::apply_add(
    const at::Tensor& act,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightMudnn<2>::apply_add_relu(
    const at::Tensor& act,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point);

template at::Tensor PackedConvWeightMudnn<2>::apply_add_silu(
    const at::Tensor& act,
    const at::Tensor& accum,
    double output_scale,
    int64_t output_zero_point);

namespace at {
namespace musa {
namespace {

template <ActMode act_mode>
class QConvAddInt8 final {
 public:
  static at::Tensor run(
      at::Tensor act,
      at::Tensor accum,
      const c10::intrusive_ptr<ConvPackedParamsBase<2>>& packed_weight,
      double output_scale,
      int64_t output_zero_point) {
    c10::musa::MUSAGuard device_guard(act.device());
    if (ActMode::RELU == act_mode) {
      return dynamic_cast<PackedConvWeightMudnn<2>*>(packed_weight.get())
          ->apply_add_relu(act, accum, output_scale, output_zero_point);
    }
    if (ActMode::IDENTITY == act_mode) {
      return dynamic_cast<PackedConvWeightMudnn<2>*>(packed_weight.get())
          ->apply_add(act, accum, output_scale, output_zero_point);
    }
    if (ActMode::SILU == act_mode) {
      return dynamic_cast<PackedConvWeightMudnn<2>*>(packed_weight.get())
          ->apply_add_silu(act, accum, output_scale, output_zero_point);
    }
    TORCH_CHECK(false, "Unsupported add-fusion activation");
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_add"),
      TORCH_FN(QConvAddInt8<ActMode::IDENTITY>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_add_relu"),
      TORCH_FN(QConvAddInt8<ActMode::RELU>::run));
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::conv2d_add_silu"),
      TORCH_FN(QConvAddInt8<ActMode::SILU>::run));
}

} // namespace
} // namespace musa
} // namespace at
