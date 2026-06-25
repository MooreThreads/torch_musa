#include <ATen/TensorUtils.h>
#include <ATen/native/GroupedMMUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_grouped_mm_native.h>
#include <ATen/ops/_scaled_grouped_mm_native.h>
#endif

#include "torch_musa/csrc/aten/mudnn/GroupedMatMul.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

namespace {

struct GroupedMatmulSlices {
  std::vector<Tensor> a_slices;
  std::vector<Tensor> b_slices;
  std::vector<Tensor> o_slices;

  std::vector<Tensor> sa_slices;
  std::vector<Tensor> sb_slices;
  std::vector<Tensor> so_slices;
};

Tensor GetScaleSlice(
    const Tensor& scale,
    int64_t group_idx,
    int64_t start,
    int64_t end,
    const char* name) {
  if (scale.numel() == 1) {
    return scale;
  }

  if (scale.dim() == 1) {
    return scale.slice(0, start, end);
  }

  TORCH_CHECK(
      scale.dim() == 2,
      name,
      " must be 1D or 2D for grouped matmul, but got ",
      scale.dim(),
      "D");
  return scale[group_idx];
}

GroupedMatmulSlices BuildGroupedMatmulSlices(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    const std::optional<Tensor>& scale_a,
    const std::optional<Tensor>& scale_b,
    bool transa,
    bool transb,
    Tensor& out) {
  GroupedMatmulSlices slices;
  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;

  // [m, total_k] @ [total_k, n] -> [b, m, n]
  const bool a2_b2 = a_is_2d && b_is_2d;
  // [total_m, k] @ [b, k, n] -> [total_m, n]
  const bool a2_b3 = a_is_2d && (!b_is_2d);
  // [b, m, k] @ [k, total_n] -> [m, total_n]
  const bool a3_b2 = (!a_is_2d) && b_is_2d;
  // [b, m, k] @ [b, k, n] -> [b, m, n]
  const bool a3_b3 = (!a_is_2d) && (!b_is_2d);

  const auto groups =
      a2_b2 ? offs->size(0) : (a2_b3 ? mat_b.size(0) : mat_a.size(0));

  const bool is_scaled = scale_a.has_value();
  TORCH_CHECK(is_scaled == scale_b.has_value());

  slices.a_slices.reserve(groups);
  slices.b_slices.reserve(groups);
  slices.o_slices.reserve(groups);

  if (is_scaled) {
    slices.sa_slices.reserve(groups);
    slices.sb_slices.reserve(groups);
  } else {
    slices.sa_slices.resize(groups, Tensor());
    slices.sb_slices.resize(groups, Tensor());
  }
  slices.so_slices.resize(groups, Tensor());

  Tensor offs_cpu;
  if (offs.has_value()) {
    offs_cpu = offs->cpu();
  }

  int64_t group_start_idx = 0;
  for (int64_t group_idx = 0; group_idx < groups; ++group_idx) {
    if (a3_b3) {
      slices.a_slices.emplace_back(mat_a[group_idx]);
      slices.b_slices.emplace_back(mat_b[group_idx]);
      slices.o_slices.emplace_back(out[group_idx]);
      continue;
    }
    const int64_t group_end_idx = offs_cpu[group_idx].item<int>();
    if (a2_b3) {
      slices.a_slices.emplace_back(
          mat_a.slice(0, group_start_idx, group_end_idx));
      slices.b_slices.emplace_back(mat_b[group_idx]);
      slices.o_slices.emplace_back(
          out.slice(0, group_start_idx, group_end_idx));
    } else if (a3_b2) {
      slices.a_slices.emplace_back(mat_a[group_idx]);
      slices.b_slices.emplace_back(
          mat_b.slice(1, group_start_idx, group_end_idx));
      slices.o_slices.emplace_back(
          out.slice(1, group_start_idx, group_end_idx));
    } else {
      slices.a_slices.emplace_back(
          mat_a.slice(1, group_start_idx, group_end_idx));
      slices.b_slices.emplace_back(
          mat_b.slice(0, group_start_idx, group_end_idx));
      slices.o_slices.emplace_back(out[group_idx]);
    }
    group_start_idx = group_end_idx;
  }

  if (is_scaled) {
    group_start_idx = 0;
    int64_t group_end_idx = 0;
    for (int64_t group_idx = 0; group_idx < groups; ++group_idx) {
      Tensor sa, sb;
      if (a3_b3) {
        sa = GetScaleSlice(*scale_a, group_idx, -1, -1, "scale_a");
        sb = GetScaleSlice(*scale_b, group_idx, -1, -1, "scale_b");
      } else if (a2_b2) {
        group_start_idx = group_idx * mat_a.size(0);
        group_end_idx = group_start_idx + mat_a.size(0);
        sa = GetScaleSlice(
            *scale_a, -1, group_start_idx, group_end_idx, "scale_a");
        group_start_idx = group_idx * mat_b.size(1);
        group_end_idx = group_start_idx + mat_b.size(1);
        sb = GetScaleSlice(
            *scale_b, -1, group_start_idx, group_end_idx, "scale_b");
      } else if (a2_b3) {
        group_end_idx = offs_cpu[group_idx].item<int>();
        sa = GetScaleSlice(
            *scale_a, -1, group_start_idx, group_end_idx, "scale_a");
        sb = GetScaleSlice(*scale_b, group_idx, -1, -1, "scale_b");
      } else {
        group_end_idx = offs_cpu[group_idx].item<int>();
        sa = GetScaleSlice(*scale_a, group_idx, -1, -1, "scale_a");
        sb = GetScaleSlice(
            *scale_b, -1, group_start_idx, group_end_idx, "scale_b");
      }
      group_start_idx = group_end_idx;
      slices.sa_slices.emplace_back(std::move(sa));
      slices.sb_slices.emplace_back(std::move(sb));
    }
  }

  return slices;
}

muTensor PrepareMatrixTensor(const Tensor& tensor, bool transposed) {
  if (transposed) {
    return CreateMUTensor(tensor.transpose(-2, -1));
  }
  return CreateMUTensor(tensor);
}

} // namespace

Tensor ScaledGroupedMM(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const Tensor& scale_a,
    const Tensor& scale_b,
    const std::optional<Tensor>& offs,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& scale_result,
    std::optional<c10::ScalarType> out_dtype,
    bool use_fast_accum) {
  UNUSED(use_fast_accum);
  TORCH_CHECK(
      getMUSAArch() >= 310,
      "scaled_grouped_gemm is only supported for MUSA_ARCH >= 310");

  TORCH_CHECK(
      !native::check_valid_strides_and_return_transposed(mat_a),
      "Expected mat1 to not be transposed");
  TORCH_CHECK(
      native::check_valid_strides_and_return_transposed(mat_b),
      "Expected mat2 to be transposed");
  const bool trans_a = false, trans_b = true;
  TORCH_CHECK(mat_a.dim() == 2 || mat_a.dim() == 3, "mat_a has to be 2 or 3d");
  TORCH_CHECK(mat_b.dim() == 2 || mat_b.dim() == 3, "mat_b has to be 2 or 3d");

  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  if (!a_is_2d || !b_is_2d) {
    TORCH_CHECK(
        mat_a.size(-1) == mat_b.size(-2),
        "contraction dimension of mat_a and mat_b must match");
  }

  TORCH_CHECK(
      mat_a.size(-1) % 16 == 0,
      "Expected trailing dimension of mat_a to be divisible by 16 but got ",
      "mat1 shape: (",
      mat_a.sizes(),
      ").");
  TORCH_CHECK(
      mat_b.size(-2) % 16 == 0 && mat_b.size(-1) % 16 == 0,
      "Expected mat_b shape to be divisible by 16 but got mat_b shape: (",
      mat_b.sizes(),
      ").");

  TORCH_CHECK(!bias.has_value(), "Bias not supported yet");
  TORCH_CHECK(!scale_result.has_value(), "Scale result not supported yet");
  TORCH_CHECK(
      offs.has_value() == (a_is_2d || b_is_2d),
      "Have to provide offsets if there is a 2d matrix");

  if (offs.has_value()) {
    TORCH_CHECK(offs->dim() == 1, "offs has to be 1D");
    TORCH_CHECK(offs->dtype() == at::kInt, "Offsets have to be int32");
  }

  TORCH_CHECK(
      scale_a.scalar_type() == kFloat && scale_b.scalar_type() == kFloat,
      "For MUSA grouped scaled gemm, both scales must be float32 tensors.");
  TORCH_CHECK(
      c10::isFloat8Type(mat_a.scalar_type()),
      "Expected mat_a to be Float8 matrix got ",
      mat_a.scalar_type());
  TORCH_CHECK(
      c10::isFloat8Type(mat_b.scalar_type()),
      "Expected mat_b to be Float8 matrix got ",
      mat_b.scalar_type());

  const auto out_dtype_ = out_dtype.value_or(kBFloat16);
  TORCH_CHECK(
      out_dtype_ == kBFloat16,
      "Only bf16 high precision output types are supported for grouped gemm");

  Tensor bias_ = bias.value_or(Tensor());
  Tensor scale_result_ = scale_result.value_or(Tensor());

  Tensor out =
      native::create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);
  const auto groups =
      a_is_2d ? (b_is_2d ? offs->size(0) : mat_b.size(0)) : mat_a.size(0);
  if (groups == 0 || out.numel() == 0) {
    return out;
  }

  auto slices = BuildGroupedMatmulSlices(
      mat_a, mat_b, offs, scale_a, scale_b, trans_a, trans_b, out);

  auto& h = GetMudnnHandle();
  ::musa::dnn::GroupedMatMul op;
  SetGroupedMatMul(
      op,
      GetMatmulComputeModeFromCtx(mat_a.scalar_type()),
      trans_a,
      trans_b,
      1.0,
      0.0,
      0.0);

  std::vector<muTensor> a_mus(groups);
  std::vector<muTensor> b_mus(groups);
  std::vector<muTensor> o_mus(groups);
  std::vector<muTensor> bias_mus(groups);
  std::vector<::musa::dnn::MatMulLtParam> params(groups);

  int64_t start_idx = 0;
  for (int64_t group_idx = 0; group_idx < groups; ++group_idx) {
    const auto& a_slice = slices.a_slices[group_idx];
    const auto& b_slice = slices.b_slices[group_idx];
    const auto& o_slice = slices.o_slices[group_idx];

    a_mus[group_idx] = PrepareMatrixTensor(a_slice, trans_a);
    b_mus[group_idx] = PrepareMatrixTensor(b_slice, trans_b);
    o_mus[group_idx] = CreateMUTensor(o_slice);

    const auto& sa_slice = slices.sa_slices[group_idx];
    const auto& sb_slice = slices.sb_slices[group_idx];
    const auto& so_slice = slices.so_slices[group_idx];
    auto sa = CreateMUTensor(sa_slice.reshape({-1, 1}));
    auto sb = CreateMUTensor(sb_slice.reshape({-1, 1}));
    auto so = CreateMUTensor(so_slice);
    CHECK_MUDNN_STATUS(
        params[group_idx].SetScale(
            std::move(sa), std::move(sb), muTensor(), std::move(so)),
        "SetScale");
  }

  CHECK_MUDNN_STATUS(
      op.RunLt(
          h,
          o_mus.data(),
          a_mus.data(),
          b_mus.data(),
          o_mus.data(),
          bias_mus.data(),
          params.data(),
          groups,
          InternalMemAlloc),
      "RunLt");

  return out;
}

Tensor GroupedMM(
    const Tensor& mat_a,
    const Tensor& mat_b,
    const std::optional<Tensor>& offs,
    const std::optional<Tensor>& bias,
    std::optional<c10::ScalarType> out_dtype) {
  native::_grouped_mm_validate_inputs(mat_a, mat_b, offs, bias, out_dtype);
  const auto out_dtype_ =
      native::_resolve_grouped_mm_out_dtype(mat_a, mat_b, out_dtype);
  Tensor out =
      native::create_grouped_gemm_output_tensor(mat_a, mat_b, offs, out_dtype_);

  const bool a_is_2d = mat_a.dim() == 2;
  const bool b_is_2d = mat_b.dim() == 2;
  const auto groups =
      a_is_2d ? (b_is_2d ? offs->size(0) : mat_b.size(0)) : mat_a.size(0);
  if (groups == 0 || out.numel() == 0) {
    return out;
  }

  const bool trans_a = native::check_valid_strides_and_return_transposed(mat_a);
  const bool trans_b = native::check_valid_strides_and_return_transposed(mat_b);
  auto slices = BuildGroupedMatmulSlices(
      mat_a, mat_b, offs, std::nullopt, std::nullopt, trans_a, trans_b, out);

  auto& h = GetMudnnHandle();
  ::musa::dnn::GroupedMatMul op;
  SetGroupedMatMul(
      op,
      GetMatmulComputeModeFromCtx(mat_a.scalar_type()),
      trans_a,
      trans_b,
      1.0,
      0.0,
      0.0);

  std::vector<muTensor> a_mus(groups);
  std::vector<muTensor> b_mus(groups);
  std::vector<muTensor> o_mus(groups);
  std::vector<muTensor> bias_mus(groups);
  std::vector<::musa::dnn::MatMulLtParam> params(groups);

  for (int64_t group_idx = 0; group_idx < groups; ++group_idx) {
    a_mus[group_idx] = PrepareMatrixTensor(slices.a_slices[group_idx], trans_a);
    b_mus[group_idx] = PrepareMatrixTensor(slices.b_slices[group_idx], trans_b);
    o_mus[group_idx] = CreateMUTensor(slices.o_slices[group_idx]);
  }

  CHECK_MUDNN_STATUS(
      op.RunLt(
          h,
          o_mus.data(),
          a_mus.data(),
          b_mus.data(),
          o_mus.data(),
          bias_mus.data(),
          params.data(),
          groups,
          InternalMemAlloc),
      "RunLt");

  return out;
}

} // namespace at::musa
