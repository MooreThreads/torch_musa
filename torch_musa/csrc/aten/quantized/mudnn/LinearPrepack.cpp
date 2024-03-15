#include <ATen/ATen.h>
#include <ATen/native/quantized/PackedParams.h>
#include <ATen/quantized/Quantizer.h>
#include <c10/core/QScheme.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

c10::intrusive_ptr<LinearPackedParamsBase> PackedLinearWeightMudnn::prepack(
    at::Tensor weight,
    c10::optional<at::Tensor> bias) {
  // TODO(@fan.mo): mudnn now only supports sym quant int8 linear
  TORCH_CHECK(
      weight.qscheme() == c10::kPerTensorSymmetric ||
          (weight.qscheme() == c10::kPerTensorAffine &&
           weight.q_zero_point() == 0),
      "Unsupported qscheme: ",
      toString(weight.qscheme()));
  TORCH_CHECK(
      weight.scalar_type() == c10::kQInt8,
      "Quantized Linear only supports QInt8 dtype");
  TORCH_CHECK(weight.dim() == 2, "Quantized Linear weight must be 2D tensor");
  const int output_channels = weight.size(0);
  const auto qtype = weight.qscheme();
  if (bias.has_value()) {
    TORCH_CHECK(bias.value().dim() == 1, "bias should be a vector (1D Tensor)");
    TORCH_CHECK(
        bias.value().size(0) == output_channels,
        "bias should have K elements: " + std::to_string(output_channels));
  }

  auto ret_ptr =
      c10::make_intrusive<PackedLinearWeightMudnn>(weight, bias, qtype);
  return ret_ptr;
}

namespace at {
namespace native {
namespace {

class QLinearPackWeightInt8Mudnn final {
 public:
  static c10::intrusive_ptr<LinearPackedParamsBase> run(
      at::Tensor weight,
      c10::optional<Tensor> bias) {
    c10::musa::MUSAGuard device_guard(weight.device());
    return PackedLinearWeightMudnn::prepack(std::move(weight), std::move(bias));
  }
};

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_prepack"),
      TORCH_FN(QLinearPackWeightInt8Mudnn::run));
}

} // namespace
} // namespace native
} // namespace at