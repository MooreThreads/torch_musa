#include "torch_musa/csrc/aten/ops/ElemwiseHelpers.h"

#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided.h>
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/reciprocal_cpu_dispatch.h>
#endif

#include <c10/core/Contiguity.h>
#include <c10/util/Optional.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

namespace {

optional<Scalar> CPUFastScalarMulScalar(const Scalar& l_s, const Scalar& r_s) {
  if (l_s.isFloatingPoint()) {
    if (r_s.isFloatingPoint()) {
      return Scalar(l_s.toDouble() * r_s.toDouble());
    } else if (r_s.isIntegral(false)) {
      return Scalar(l_s.toDouble() * r_s.toLong());
    }
  } else if (l_s.isIntegral(false)) {
    if (r_s.isFloatingPoint()) {
      return Scalar(l_s.toLong() * r_s.toDouble());
    } else if (r_s.isIntegral(false)) {
      return Scalar(l_s.toLong() * r_s.toLong());
    }
  }
  return nullopt;
}

optional<Scalar> CPUFastScalarReciprocal(const Scalar& input) {
  if (input.isFloatingPoint() || input.isIntegral(false)) {
    return Scalar(static_cast<double>(1.0) / input.toDouble());
  }
  return nullopt;
}

template <typename MUDNN_OP>
void SetAlpha(MUDNN_OP& op, const Scalar& alpha) {
  if (alpha.isFloatingPoint()) {
    CHECK_MUDNN_STATUS(op.SetAlpha(alpha.toDouble()), "SetAlpha");
  } else if (alpha.isBoolean()) {
    const auto v = static_cast<int64_t>(alpha.to<bool>());
    CHECK_MUDNN_STATUS(op.SetAlpha(v), "SetAlpha");
  } else {
    CHECK_MUDNN_STATUS(op.SetAlpha(alpha.toLong()), "SetAlpha");
  }
}

ScalarType UnaryTrueDivSuggestInputType(
    ScalarType i_type,
    ScalarType s_type,
    ScalarType o_type) {
  if (o_type == i_type || o_type != ScalarType::Float ||
      isFloatingType(s_type)) {
    return o_type;
  }
  return isIntegralType(i_type, false) ? i_type : o_type;
}

std::optional<Tensor> BinaryMayContigRHS(MusaTensorIterator& iter) {
  const auto r_shape = iter.shape();
  const auto r_strides = iter.strides(2);
  const size_t ndim = iter.ndim();

  DimVector useful_shape, useful_strides, useful_ids;
  for (const auto i : c10::irange(ndim)) {
    const auto sp_i = r_shape[i];
    const auto st_i = r_strides[i];
    if (sp_i != 1 && st_i != 0) {
      useful_shape.push_back(sp_i);
      useful_strides.push_back(st_i);
      useful_ids.push_back(i);
    }
  }

  const size_t useful_ndim = useful_shape.size();
  const bool is_contig = c10::_compute_contiguous(
      IntArrayRef(useful_shape.data(), useful_ndim),
      IntArrayRef(useful_strides.data(), useful_ndim),
      static_cast<int64_t>(iter.original_input(1).numel()));
  // Device scalar tensor is always contiguous.
  if (is_contig) {
    return std::nullopt;
  }

  Tensor r_contig =
      at::as_strided(iter.tensor(2), useful_shape, useful_strides).contiguous();
  DimVector tmp_strides(ndim, 0);
  for (const auto i : c10::irange(useful_ndim)) {
    tmp_strides[useful_ids[i]] = r_contig.stride(i);
  }

  auto res = at::as_strided(r_contig, r_shape, tmp_strides);
  return res;
}

} // anonymous namespace

Scalar CPUScalarMulScalar(const Tensor& l_t, const Scalar& r_s) {
  const auto opt_res = CPUFastScalarMulScalar(l_t.item(), r_s);
  return opt_res.value_or(
      at::cpu::mul(l_t, native::wrapped_scalar_tensor(r_s)).item());
}

Scalar CPUScalarReciprocal(const Scalar& input) {
  const auto opt_res = CPUFastScalarReciprocal(input);
  return opt_res.value_or(
      at::cpu::reciprocal(native::wrapped_scalar_tensor(input)).item());
}

void UnaryCall(
    MusaTensorIterator& iter,
    UNARY_MODE mode,
    const std::string op_name) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Unary op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  auto out = iter.mu_output(0);
  auto in = iter.mu_input(0);
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run " + op_name);
}

void UnaryAlphaCall(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    UNARY_MODE mode,
    const std::string op_name) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Unary op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  SetAlpha(op, alpha);
  auto out = iter.mu_output(0);
  auto in = iter.mu_input(0);
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run " + op_name);
}

void BinaryCall(
    MusaTensorIterator& iter,
    BINARY_MODE mode,
    const std::string& op_name) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Binary op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  auto out = iter.mu_output(0);
  auto lhs = iter.mu_input(0);

  std::optional<Tensor> maybe_rhs_contig = BinaryMayContigRHS(iter);
  auto rhs = maybe_rhs_contig.has_value()
      ? CreateMUTensor(maybe_rhs_contig.value(), false)
      : iter.mu_input(1);

  // auto rhs = iter.mu_input(1);
  CHECK_MUDNN_STATUS(op.Run(h, out, lhs, rhs), "Run " + op_name);
}

void BinaryAlphaCall(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    BINARY_MODE mode,
    const std::string& op_name) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Binary op;
  SetAlpha(op, alpha);
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  auto out = iter.mu_output(0);
  auto lhs = iter.mu_input(0);

  std::optional<Tensor> maybe_rhs_contig = BinaryMayContigRHS(iter);
  auto rhs = maybe_rhs_contig.has_value()
      ? CreateMUTensor(maybe_rhs_contig.value(), false)
      : iter.mu_input(1);

  // auto rhs = iter.mu_input(1);
  CHECK_MUDNN_STATUS(op.Run(h, out, lhs, rhs), "Run " + op_name);
}

void TernaryCall(
    MusaTensorIterator& iter,
    TERNARY_MODE mode,
    const std::string& op_name) {
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Ternary op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  auto out = iter.mu_output(0);
  auto inp0 = iter.mu_input(0);
  auto inp1 = iter.mu_input(1);
  auto inp2 = iter.mu_input(2);
  CHECK_MUDNN_STATUS(op.Run(h, out, inp0, inp1, inp2), "Run");
}

std::pair<ScalarType, ScalarType> BinaryAddSuggestInputTypes(
    MusaTensorIterator& iter) {
  const auto c_type = iter.common_dtype();
  if (c_type != ScalarType::Float || iter.is_cpu_scalar(1) ||
      iter.is_cpu_scalar(2)) {
    return std::make_pair(c_type, c_type);
  }

  const auto l_type = iter.dtype(1);
  const auto r_type = iter.dtype(2);

  auto is_f16 = [](ScalarType t) -> bool {
    return t == ScalarType::Half || t == ScalarType::BFloat16;
  };

  if (is_f16(l_type) && r_type == ScalarType::Float) {
    return std::make_pair(l_type, r_type);
  }

  if (is_f16(r_type) && l_type == ScalarType::Float) {
    return std::make_pair(l_type, r_type);
  }

  return std::make_pair(c_type, c_type);
}

std::pair<ScalarType, ScalarType> BinaryTrueDivSuggestInputTypes(
    MusaTensorIterator& iter) {
  const auto c_type = iter.common_dtype();
  const auto l_type = iter.dtype(1);
  const auto r_type = iter.dtype(2);
  if (iter.is_cpu_scalar(1)) {
    return std::make_pair(
        l_type, UnaryTrueDivSuggestInputType(r_type, l_type, c_type));
  }
  if (iter.is_cpu_scalar(2)) {
    return std::make_pair(
        UnaryTrueDivSuggestInputType(l_type, r_type, c_type), r_type);
  }
  if (l_type != r_type || l_type == c_type || c_type != ScalarType::Float) {
    return std::make_pair(c_type, c_type);
  }
  if (l_type == ScalarType::Char || l_type == ScalarType::Short) {
    return std::make_pair(l_type, r_type);
  }
  return std::make_pair(c_type, c_type);
}

} // namespace musa
} // namespace at
