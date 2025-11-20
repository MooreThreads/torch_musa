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
#include "torch_musa/csrc/aten/quantized/mudnn/Linear.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

// output Tensor will be a clampped int8 Tensor
// both act and weight will be int8 Tensor
// Numerics are the same as conv (see aten/src/ATen/native/quantized/Conv.cpp):

namespace at::musa {
static void quantized_matmul_per_channel(
    at::Tensor mat1, // act
    double input_scale,
    int64_t input_zero_point,
    at::Tensor mat2, // weight
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<Tensor> bias,
    at::Tensor& result, // output
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype,
    const std::string_view& unary_post_op,
    torch::List<std::optional<at::Scalar>>& unary_post_op_args,
    std::string_view unary_post_op_algorithm) {
  // todo W8A8 matmul kernel
  auto act_fp32 = (mat1.to(c10::kFloat) - input_zero_point) * input_scale;
  auto weight_fp32 =
      (mat2.to(c10::kFloat) - weight_zero_points) * weight_scales;
  result = at::matmul(act_fp32, weight_fp32);
  if (bias.has_value()) {
    result += bias.value();
  }
  if (unary_post_op == "relu") {
    result = at::relu(result);
  } else if (unary_post_op == "tanh") {
    result = at::tanh(result);
  } else if (unary_post_op == "gelu") {
    result = at::gelu(result);
  }
  int64_t clamp_min, clamp_max;
  switch (output_dtype) {
    case c10::ScalarType::Byte:
      clamp_min = 0;
      clamp_max = 255;
      break;
    case c10::ScalarType::Char:
      clamp_min = -128;
      clamp_max = 127;
      break;
    default:
      result = result.to(output_dtype);
      return;
  }
  result = (result / output_scale + output_zero_point)
               .round()
               .clamp(clamp_min, clamp_max)
               .to(output_dtype);
  return;
}

static at::Tensor q_linear_per_tensor(
    at::Tensor act, // act
    double act_scale,
    int64_t act_zero_point,
    at::Tensor weight, // weight
    at::Tensor weight_scales,
    at::Tensor weight_zero_points,
    std::optional<Tensor> bias,
    double output_scale,
    int64_t output_zero_point,
    std::optional<c10::ScalarType> output_dtype,
    std::string_view post_op_name,
    torch::List<std::optional<at::Scalar>> post_op_args,
    std::string_view post_op_algorithm) {
  c10::musa::MUSAGuard device_guard(act.device());
  Tensor b_raw = bias.has_value() ? bias.value() : at::Tensor();
  const int64_t dim = act.dim();
  TORCH_CHECK(dim == 2, "qlinear musa: input dim should be 2, but got", dim);
  int64_t K = act.size(dim - 1);
  int64_t M = act.numel() / K;
  // [M,K] * [K, N]
  int64_t N = weight.size(0);

  std::vector<int64_t> src_dims = {M, K};
  std::vector<int64_t> dst_dims = {M, N};
  auto dst_dtype = output_dtype.value_or(act.scalar_type());

  Tensor qout = at::empty(dst_dims, act.options().dtype(dst_dtype));
  quantized_matmul_per_channel(
      act.contiguous(),
      act_scale,
      act_zero_point,
      weight.contiguous(),
      weight_scales,
      weight_zero_points,
      b_raw,
      qout,
      output_scale,
      output_zero_point,
      dst_dtype,
      post_op_name,
      post_op_args,
      post_op_algorithm);
  return qout;
}
TORCH_LIBRARY_IMPL(quantized, PrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("quantized::linear_per_tensor"),
      TORCH_FN(q_linear_per_tensor));
}
} // namespace at::musa
