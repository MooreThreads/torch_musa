#include "torch_musa/csrc/aten/ops/ElemwiseHelpers.h"

#include <ATen/ScalarOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/reciprocal_cpu_dispatch.h>
#endif

#include <c10/util/Optional.h>

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
  auto rhs = iter.mu_input(1);
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
  auto rhs = iter.mu_input(1);
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
