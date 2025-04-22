#include <ATen/Config.h>
#include <ATen/ScalarOps.h>
#include <ATen/native/BinaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/add_native.h>
#include <ATen/ops/div_cpu_dispatch.h>
#include <ATen/ops/div_native.h>
#include <ATen/ops/empty_like_native.h>
#include <ATen/ops/mul_cpu_dispatch.h>
#include <ATen/ops/mul_musa_dispatch.h>
#include <ATen/ops/mul_native.h>
#include <ATen/ops/reciprocal_musa_dispatch.h>
#include <ATen/ops/sub_native.h>
#endif

#include "torch_musa/csrc/aten/ops/ElemwiseHelpers.h"

namespace at {
namespace musa {

namespace {

void InitBinaryIterator(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs) {
  iter.add_output(out);
  iter.add_input(lhs);
  iter.add_input(rhs);
}

void SetUpBinaryConfig(TensorIteratorConfig& config) {
  config.set_check_mem_overlap(true)
      .allow_cpu_scalars(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true);
}

void SetUpBinaryFloatConfig(TensorIteratorConfig& config) {
  SetUpBinaryConfig(config);
  config.promote_integer_inputs_to_float(true);
}

void AddMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    const Scalar& alpha) {
  InitBinaryIterator(iter, out, lhs, rhs);
  auto dtype_lifter = [](ScalarType t) -> ScalarType {
    auto promote_type = t;
    switch (t) {
      case ScalarType::Bool:
        promote_type = ScalarType::Char;
        break;
      case ScalarType::Byte:
        promote_type = ScalarType::Short;
        break;
      default:
        break;
    }
    return promote_type;
  };
  iter.set_musa_common_dtype_lifter(dtype_lifter);
  {
    TensorIteratorConfig config;
    SetUpBinaryConfig(config);
    iter.build(config);
  }
  native::alpha_check(iter.dtype(), alpha);
}

void SubMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    const Scalar& alpha) {
  native::sub_check(lhs, rhs);
  AddMeta(iter, out, lhs, rhs, alpha);
}

template <UNARY_MODE mode>
void UnaryAddOrSub(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    const std::string& op_name) {
  if (!alpha.equal(0)) {
    UnaryAlphaCall(iter, alpha, mode, op_name);
  } else if (!iter.output(0).is_same(iter.input(0))) {
    UnaryCall(iter, UNARY_MODE::IDENTITY, op_name);
  }
  return;
}

void AddImpl(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    const std::string& op_name) {
  if (iter.is_cpu_scalar(1)) {
    const auto unary_alpha = iter.input(0).item();
    if (!alpha.equal(1)) {
      const auto& rhs = iter.input(1);
      const auto alpha_tensor = at::native::wrapped_scalar_tensor(alpha);
      if (iter.input_is_type_corrected(1)) {
        at::musa::mul_(const_cast<Tensor&>(rhs), alpha_tensor);
      } else {
        Tensor unary_in = at::native::empty_like(rhs);
        at::musa::mul_out(unary_in, rhs, alpha_tensor);
        iter.replace_input(1, unary_in);
      }
    }
    iter.remove_operand(1);
    UnaryAddOrSub<UNARY_MODE::ADD>(iter, unary_alpha, op_name);
  } else if (iter.is_cpu_scalar(2)) {
    const auto unary_alpha = CPUScalarMulScalar(iter.tensor(2), alpha);
    UnaryAddOrSub<UNARY_MODE::ADD>(iter, unary_alpha, op_name);
  } else if (alpha.equal(0)) {
    UnaryAddOrSub<UNARY_MODE::ADD>(iter, alpha, op_name);
  } else if (alpha.equal(1)) {
    BinaryCall(iter, BINARY_MODE::ADD, op_name);
  } else {
    BinaryAlphaCall(iter, alpha, BINARY_MODE::ADD_ALPHA, op_name);
  }
}

void SubImpl(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    const std::string& op_name) {
  if (iter.is_cpu_scalar(1)) {
    const auto unary_alpha = iter.input(0).item();
    if (!alpha.equal(1)) {
      const auto& rhs = iter.input(1);
      const auto alpha_tensor = at::native::wrapped_scalar_tensor(alpha);
      if (iter.input_is_type_corrected(1)) {
        at::musa::mul_(const_cast<Tensor&>(rhs), alpha_tensor);
      } else {
        Tensor unary_in = at::native::empty_like(rhs);
        at::musa::mul_out(unary_in, rhs, alpha_tensor);
        iter.replace_input(1, unary_in);
      }
    }
    iter.remove_operand(1);
    UnaryAlphaCall(iter, unary_alpha, UNARY_MODE::SUB_BY_ALPHA, op_name);
  } else if (iter.is_cpu_scalar(2)) {
    const auto unary_alpha = CPUScalarMulScalar(iter.tensor(2), alpha);
    UnaryAddOrSub<UNARY_MODE::SUB>(iter, unary_alpha, op_name);
  } else if (alpha.equal(0)) {
    UnaryAddOrSub<UNARY_MODE::SUB>(iter, alpha, op_name);
  } else if (alpha.equal(1)) {
    BinaryCall(iter, BINARY_MODE::SUB, op_name);
  } else {
    BinaryAlphaCall(iter, alpha, BINARY_MODE::SUB_ALPHA, op_name);
  }
}

#define GEN_ALPHA_IMPL(TPL, FUNC, META, IMPL) \
  TPL void FUNC(                              \
      MusaTensorIterator& iter,               \
      const Tensor& out,                      \
      const Tensor& lhs,                      \
      const Tensor& rhs,                      \
      const Scalar& alpha,                    \
      const std::string& op_name) {           \
    META(iter, out, lhs, rhs, alpha);         \
    if (iter.numel() != 0) {                  \
      IMPL(iter, alpha, op_name);             \
    }                                         \
    iter.cast_outputs();                      \
  }

GEN_ALPHA_IMPL(, BinaryAdd, AddMeta, AddImpl)
GEN_ALPHA_IMPL(, BinarySub, SubMeta, SubImpl)

#undef GEN_ALPHA_IMPL

void MulMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs) {
  InitBinaryIterator(iter, out, lhs, rhs);
  TensorIteratorConfig config;
  SetUpBinaryConfig(config);
  iter.build(config);
}

void UnaryMul(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    const std::string& op_name) {
  if (!alpha.equal(1)) {
    UnaryAlphaCall(iter, alpha, UNARY_MODE::MUL, op_name);
  } else if (!iter.output(0).is_same(iter.input(0))) {
    UnaryCall(iter, UNARY_MODE::IDENTITY, op_name);
  }
  return;
}

void MulImpl(MusaTensorIterator& iter, const std::string& op_name) {
  if (C10_UNLIKELY(BinaryMulFallThroughCPU(iter))) {
    const auto cpu_out = cpu::mul(iter.tensor(1).cpu(), iter.tensor(2).cpu());
    iter.output().copy_(cpu_out);
    return;
  }
  if (iter.is_cpu_scalar(1)) {
    const auto unary_alpha = iter.input(0).item();
    iter.remove_operand(1);
    UnaryMul(iter, unary_alpha, op_name);
  } else if (iter.is_cpu_scalar(2)) {
    const auto unary_alpha = iter.input(1).item();
    UnaryMul(iter, unary_alpha, op_name);
  } else {
    BinaryCall(iter, BINARY_MODE::MUL, op_name);
  }
}

template <BINARY_MODE div_mode>
void DivMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs) {
  InitBinaryIterator(iter, out, lhs, rhs);
  TensorIteratorConfig config;
  if constexpr (div_mode == BINARY_MODE::TRUEDIV) {
    iter.musa_promote_inputs_to_common_dtype(false);
    SetUpBinaryFloatConfig(config);
  } else {
    SetUpBinaryConfig(config);
  }
  iter.build(config);
}

template <UNARY_MODE div_mode>
void UnaryDiv(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    const std::string& op_name) {
  if (!alpha.equal(1)) {
    UnaryAlphaCall(iter, alpha, div_mode, op_name);
  } else {
    if constexpr (
        div_mode == UNARY_MODE::TRUNCATEDIV ||
        div_mode == UNARY_MODE::FLOORDIV) {
      UnaryAlphaCall(iter, alpha, div_mode, op_name);
    } else if (!iter.output(0).is_same(iter.input(0))) {
      UnaryCall(iter, UNARY_MODE::IDENTITY, op_name);
    }
  }
  return;
}

template <BINARY_MODE div_mode>
void DivImpl(MusaTensorIterator& iter, const std::string& op_name) {
  if (C10_UNLIKELY(BinaryDivFallThroughCPU(iter))) {
    const auto cpu_out = cpu::div(iter.tensor(1).cpu(), iter.tensor(2).cpu());
    iter.output().copy_(cpu_out);
    return;
  }

  const bool l_is_scalar = iter.is_cpu_scalar(1);
  const bool r_is_scalar = iter.is_cpu_scalar(2);

  if constexpr (div_mode == BINARY_MODE::TRUEDIV) {
    const auto [suggest_ltype, suggest_rtype] =
        BinaryTrueDivSuggestInputTypes(iter);
    if (!l_is_scalar && suggest_ltype != iter.dtype(1)) {
      const auto new_lhs = iter.tensor(1).to(suggest_ltype);
      iter.replace_input(0, new_lhs);
    }
    if (!r_is_scalar && suggest_rtype != iter.dtype(2)) {
      const auto new_rhs = iter.tensor(2).to(suggest_rtype);
      iter.replace_input(1, new_rhs);
    }
  }

  if (l_is_scalar) {
    const auto unary_alpha = iter.input(0).item();
    constexpr auto UnaryMode = div_mode == BINARY_MODE::TRUNCATEDIV
        ? UNARY_MODE::TRUNCATEDIV_BY_ALPHA
        : div_mode == BINARY_MODE::FLOORDIV ? UNARY_MODE::FLOORDIV_BY_ALPHA
                                            : UNARY_MODE::TRUEDIV_BY_ALPHA;
    iter.remove_operand(1);
    UnaryAlphaCall(iter, unary_alpha, UnaryMode, op_name);
  } else if (r_is_scalar) {
    const auto unary_alpha = iter.input(1).item();
    constexpr auto UnaryMode = div_mode == BINARY_MODE::TRUNCATEDIV
        ? UNARY_MODE::TRUNCATEDIV
        : div_mode == BINARY_MODE::FLOORDIV ? UNARY_MODE::FLOORDIV
                                            : UNARY_MODE::TRUEDIV;
    UnaryDiv<UnaryMode>(iter, unary_alpha, op_name);
  } else {
    BinaryCall(iter, div_mode, op_name);
  }
}

void FModMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs) {
  InitBinaryIterator(iter, out, lhs, rhs);
  TensorIteratorConfig config;
  SetUpBinaryConfig(config);
  iter.build(config);
}

void FModImpl(MusaTensorIterator& iter, const std::string& op_name) {
  // torch disallow (Number input, Tensor other, *, Tensor out) schema.
  if (iter.is_cpu_scalar(2)) {
    const auto unary_alpha = iter.input(1).item();
    UnaryAlphaCall(iter, unary_alpha, UNARY_MODE::TRUNCATEMOD, op_name);
  } else {
    BinaryCall(iter, BINARY_MODE::TRUNCATEMOD, op_name);
  }
}

#define FOPS_TPL(F_NAME, M_NAME)                                               \
  void F##F_NAME##Meta(                                                        \
      MusaTensorIterator& iter,                                                \
      const Tensor& out,                                                       \
      const Tensor& lhs,                                                       \
      const Tensor& rhs) {                                                     \
    InitBinaryIterator(iter, out, lhs, rhs);                                   \
    TensorIteratorConfig config;                                               \
    SetUpBinaryConfig(config);                                                 \
    iter.build(config);                                                        \
  }                                                                            \
  void F##F_NAME##Impl(MusaTensorIterator& iter, const std::string& op_name) { \
    if (iter.is_cpu_scalar(1)) {                                               \
      const auto unary_alpha = iter.input(0).item();                           \
      UnaryAlphaCall(iter, unary_alpha, UNARY_MODE::M_NAME, op_name);          \
    } else if (iter.is_cpu_scalar(2)) {                                        \
      const auto unary_alpha = iter.input(1).item();                           \
      UnaryAlphaCall(iter, unary_alpha, UNARY_MODE::M_NAME, op_name);          \
    } else {                                                                   \
      BinaryCall(iter, BINARY_MODE::M_NAME, op_name);                          \
    }                                                                          \
  }

FOPS_TPL(Min, MIN)
FOPS_TPL(Max, MAX)

#undef FOPS_TPL

#define GEN_IMPL(TPL, FUNC, META, IMPL) \
  TPL void FUNC(                        \
      MusaTensorIterator& iter,         \
      const Tensor& out,                \
      const Tensor& lhs,                \
      const Tensor& rhs,                \
      const std::string& op_name) {     \
    META(iter, out, lhs, rhs);          \
    if (iter.numel() != 0) {            \
      IMPL(iter, op_name);              \
    }                                   \
    iter.cast_outputs();                \
  }

GEN_IMPL(, BinaryMul, MulMeta, MulImpl)
// clang-format off
GEN_IMPL(
    template <BINARY_MODE div_mode>,
    BinaryDivMode,
    DivMeta<div_mode>,
    DivImpl<div_mode>)
// clang-format on
GEN_IMPL(, BinaryFMod, FModMeta, FModImpl)
GEN_IMPL(, BinaryFMin, FMinMeta, FMinImpl)
GEN_IMPL(, BinaryFMax, FMaxMeta, FMaxImpl)

#undef GEN_IMPL

void BinaryDivDispatch(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& lhs,
    const Tensor& rhs,
    optional<string_view> rounding_mode,
    const std::string& op_name) {
  if (!rounding_mode.has_value()) {
    BinaryDivMode<BINARY_MODE::TRUEDIV>(iter, out, lhs, rhs, op_name);
    return;
  }
  const string_view view = (*rounding_mode);
  if (view == "trunc") {
    BinaryDivMode<BINARY_MODE::TRUNCATEDIV>(iter, out, lhs, rhs, op_name);
  } else if (view == "floor") {
    BinaryDivMode<BINARY_MODE::FLOORDIV>(iter, out, lhs, rhs, op_name);
  } else {
    TORCH_CHECK(
        false,
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' ",
        "but found '",
        static_cast<std::string>(view),
        "'");
  }
}

} // anonymous namespace

#define GEN_ALPHA_FUNCTION(OP, FUNC)                                        \
  Tensor OP(const Tensor& self, const Tensor& other, const Scalar& alpha) { \
    FunctionalTensorIterator iter;                                          \
    FUNC(iter, Tensor(), self, other, alpha, __func__);                     \
    return iter.output();                                                   \
  }                                                                         \
                                                                            \
  Tensor& OP##_(Tensor& self, const Tensor& other, const Scalar& alpha) {   \
    InplaceTensorIterator iter;                                             \
    FUNC(iter, self, self, other, alpha, __func__);                         \
    return self;                                                            \
  }                                                                         \
                                                                            \
  Tensor& OP##Out(                                                          \
      const Tensor& self,                                                   \
      const Tensor& other,                                                  \
      const Scalar& alpha,                                                  \
      Tensor& output) {                                                     \
    OutTensorIterator iter;                                                 \
    FUNC(iter, output, self, other, alpha, __func__);                       \
    return output;                                                          \
  }

GEN_ALPHA_FUNCTION(AddTensor, BinaryAdd)
GEN_ALPHA_FUNCTION(SubTensor, BinarySub)

#undef GEN_ALPHA_FUNCTION

#define GEN_FUNCTION(OP, FUNC)                                               \
  Tensor OP(const Tensor& self, const Tensor& other) {                       \
    FunctionalTensorIterator iter;                                           \
    FUNC(iter, Tensor(), self, other, __func__);                             \
    return iter.output();                                                    \
  }                                                                          \
                                                                             \
  Tensor& OP##_(Tensor& self, const Tensor& other) {                         \
    InplaceTensorIterator iter;                                              \
    FUNC(iter, self, self, other, __func__);                                 \
    return self;                                                             \
  }                                                                          \
                                                                             \
  Tensor& OP##Out(const Tensor& self, const Tensor& other, Tensor& output) { \
    OutTensorIterator iter;                                                  \
    FUNC(iter, output, self, other, __func__);                               \
    return output;                                                           \
  }

GEN_FUNCTION(MulTensor, BinaryMul)
GEN_FUNCTION(DivTensor, BinaryDivMode<BINARY_MODE::TRUEDIV>)
GEN_FUNCTION(FloorDivideTensor, BinaryDivMode<BINARY_MODE::FLOORDIV>)
GEN_FUNCTION(FModTensor, BinaryFMod)
GEN_FUNCTION(FMin, BinaryFMin)
GEN_FUNCTION(FMax, BinaryFMax)

#undef GEN_FUNCTION

Tensor DivTensorMode(
    const Tensor& self,
    const Tensor& other,
    optional<string_view> rounding_mode) {
  FunctionalTensorIterator iter;
  BinaryDivDispatch(iter, Tensor(), self, other, rounding_mode, __func__);
  return iter.output();
}

Tensor& DivTensorMode_(
    Tensor& self,
    const Tensor& other,
    optional<string_view> rounding_mode) {
  InplaceTensorIterator iter;
  BinaryDivDispatch(iter, self, self, other, rounding_mode, __func__);
  return self;
}

Tensor& DivTensorModeOut(
    const Tensor& self,
    const Tensor& other,
    optional<string_view> rounding_mode,
    Tensor& output) {
  OutTensorIterator iter;
  BinaryDivDispatch(iter, output, self, other, rounding_mode, __func__);
  return output;
}

Tensor& RSubTensorOut(
    const Tensor& self,
    const Tensor& other,
    const Scalar& alpha,
    Tensor& out) {
  return SubTensorOut(other, self, alpha, out);
}

Tensor& RSubScalarOut(
    const Tensor& self,
    const Scalar& other,
    const Scalar& alpha,
    Tensor& out) {
  return RSubTensorOut(self, native::wrapped_scalar_tensor(other), alpha, out);
}

} // namespace musa
} // namespace at
