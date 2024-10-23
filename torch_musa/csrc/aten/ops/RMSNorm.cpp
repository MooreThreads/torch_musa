#include <ATen/ATen.h>
#include <ATen/native/layer_norm.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

namespace {

ScalarType InferInvvarDType(ScalarType input_dtype) {
  ScalarType invvar_dtype = input_dtype;
  switch (input_dtype) {
    case ScalarType::Half:
    case ScalarType::BFloat16:
    case ScalarType::Float:
      invvar_dtype = ScalarType::Float;
      break;
    default:
      invvar_dtype = input_dtype;
  }
  return invvar_dtype;
}

Tensor CreateRMSNormFwdInvvar(
    const Tensor& input,
    IntArrayRef normalized_shape) {
  const auto input_shape = input.sizes();
  const auto invvar_ndim = input_shape.size() - normalized_shape.size();
  const auto invvar_dtype = InferInvvarDType(input.scalar_type());

  return at::empty(
      IntArrayRef(input_shape.data(), invvar_ndim),
      input.options().dtype(invvar_dtype));
}

std::vector<int> CreateMuDNNNormAxes(
    const Tensor& input,
    IntArrayRef normalized_shape) {
  const auto norm_ndim = normalized_shape.size();
  std::vector<int> norm_axes(norm_ndim);
  std::iota(
      norm_axes.begin(),
      norm_axes.end(),
      static_cast<int>(input.dim()) - static_cast<int>(norm_ndim));
  return norm_axes;
}

} // anonymous namespace

std::tuple<Tensor, Tensor> FusedRMSNormForward(
    const Tensor& input,
    IntArrayRef normalized_shape,
    double eps,
    const c10::optional<Tensor>& weight_opt) {
  const auto weight = weight_opt.value_or(Tensor());
  (void)native::_check_layer_norm_inputs(
      input, normalized_shape, weight, Tensor());

  const auto input_device = input.device();
  TORCH_CHECK(
      input_device.is_privateuseone(),
      "Device of input tensor of RMSNorm must be MUSA, but now is ",
      input_device);
  TORCH_CHECK(
      !weight.defined() || weight.device() == input_device,
      "Devices of input/weight tensors of RMSNorm must be the same, but now ",
      "input in ",
      input_device,
      ", weight in ",
      weight.device());

  const auto input_dtype = input.scalar_type();
  TORCH_CHECK(
      input_dtype == ScalarType::Float || input_dtype == ScalarType::Half ||
          input_dtype == ScalarType::BFloat16,
      "Dtype of input tensor of RMSNorm only supports Float32, Half and BFloat16, but now is ",
      input_dtype);
  TORCH_CHECK(
      !weight.defined() || weight.scalar_type() == input_dtype,
      "Dtypes of input/weight tensors of RMSNorm must be the same, but now ",
      "input in ",
      input_dtype,
      ", weight in ",
      weight.scalar_type());

  const c10::musa::MUSAGuard device_guard(input_device);
  const auto contig_input = FormatContiguous(input, MemoryFormat::Contiguous);
  const auto contig_gamma = weight.defined()
      ? FormatContiguous(weight, MemoryFormat::Contiguous)
      : Tensor();
  auto contig_output = at::empty_like(contig_input);
  auto contig_invvar = CreateRMSNormFwdInvvar(contig_input, normalized_shape);

  if (C10_UNLIKELY(contig_output.numel() == 0)) {
    return {contig_output, contig_invvar};
  }

  ::musa::dnn::RMSNorm op;
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  const auto norm_axes = CreateMuDNNNormAxes(contig_input, normalized_shape);
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axes.size(), norm_axes.data()), "SetAxis");
  auto mudnn_out = CreateMUTensor(contig_output);
  auto mudnn_invvar = CreateMUTensor(contig_invvar);
  auto mudnn_in = CreateMUTensor(contig_input);
  auto mudnn_gamma = contig_gamma.defined() ? CreateMUTensor(contig_gamma)
                                            : CreateEmptyMUTensor();
  muHandle& h = GetMudnnHandle();
  CHECK_MUDNN_STATUS(
      op.Run(h, mudnn_out, mudnn_invvar, mudnn_in, mudnn_gamma),
      "RunRMSNormFwd");

  return {contig_output, contig_invvar};
}

std::tuple<Tensor, Tensor> FusedRMSNormBackward(
    const Tensor& grad_out,
    const Tensor& invvar,
    const Tensor& input,
    IntArrayRef normalized_shape,
    double eps,
    const c10::optional<Tensor>& weight_opt) {
  const c10::musa::MUSAGuard device_guard(input.device());
  const auto weight = weight_opt.value_or(Tensor());
  const auto contig_input = FormatContiguous(input, MemoryFormat::Contiguous);
  const auto contig_grad_output =
      FormatContiguous(grad_out, MemoryFormat::Contiguous);
  const auto contig_invvar = FormatContiguous(invvar, MemoryFormat::Contiguous);

  auto contig_grad_input = at::empty_like(contig_input);
  const bool has_weight = weight.defined();
  auto contig_gamma = has_weight
      ? FormatContiguous(weight, MemoryFormat::Contiguous)
      : Tensor();
  auto contig_grad_gamma = has_weight ? at::empty_like(contig_gamma) : Tensor();

  ::musa::dnn::RMSNorm op;
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  const auto norm_axes = CreateMuDNNNormAxes(contig_input, normalized_shape);
  CHECK_MUDNN_STATUS(op.SetAxis(norm_axes.size(), norm_axes.data()), "SetAxis");
  auto mudnn_dx = CreateMUTensor(contig_grad_input);
  auto mudnn_dg =
      has_weight ? CreateMUTensor(contig_grad_gamma) : CreateEmptyMUTensor();
  auto mudnn_dy = CreateMUTensor(contig_grad_output);
  auto mudnn_in = CreateMUTensor(contig_input);
  auto mudnn_rstd = CreateMUTensor(contig_invvar);
  auto mudnn_gamma =
      has_weight ? CreateMUTensor(contig_gamma) : CreateEmptyMUTensor();
  muHandle& h = GetMudnnHandle();
  CHECK_MUDNN_STATUS(
      op.RunBwd(
          h,
          mudnn_dx,
          mudnn_dg,
          mudnn_dy,
          mudnn_in,
          mudnn_rstd,
          mudnn_gamma,
          InternalMemAlloc),
      "RunRMSNormBwd");

  return {contig_grad_input, contig_grad_gamma};
}

} // namespace musa
} // namespace at
