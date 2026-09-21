#include <ATen/Config.h>
#include <ATen/native/UnaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/reciprocal_native.h>
#endif

#include "torch_musa/csrc/aten/ops/ElemwiseHelpers.h"

namespace at::musa {

namespace {

void InitUnaryIterator(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& inp) {
  iter.add_output(out);
  iter.add_input(inp);
}

void SetUpUnaryFloatConfig(TensorIteratorConfig& config) {
  config.set_check_mem_overlap(true)
      .promote_inputs_to_common_dtype(true)
      .cast_common_dtype_to_outputs(true)
      .enforce_safe_casting_to_output(true)
      .promote_integer_inputs_to_float(true);
}

#define UNARY_FLOAT_COMMON_META(FUNC)                                         \
  void FUNC(MusaTensorIterator& iter, const Tensor& out, const Tensor& inp) { \
    InitUnaryIterator(iter, out, inp);                                        \
    TensorIteratorConfig config;                                              \
    SetUpUnaryFloatConfig(config);                                            \
    iter.build(config);                                                       \
  }

UNARY_FLOAT_COMMON_META(ReciprocalMeta)
UNARY_FLOAT_COMMON_META(ErfMeta)
UNARY_FLOAT_COMMON_META(LogMeta)
UNARY_FLOAT_COMMON_META(Log2Meta)
UNARY_FLOAT_COMMON_META(Log10Meta)
UNARY_FLOAT_COMMON_META(SqrtMeta)
UNARY_FLOAT_COMMON_META(RsqrtMeta)
UNARY_FLOAT_COMMON_META(TanhMeta)
UNARY_FLOAT_COMMON_META(TanMeta)
UNARY_FLOAT_COMMON_META(SigmoidMeta)
UNARY_FLOAT_COMMON_META(ExpMeta)
UNARY_FLOAT_COMMON_META(SinMeta)
UNARY_FLOAT_COMMON_META(CosMeta)
UNARY_FLOAT_COMMON_META(AcosMeta)
UNARY_FLOAT_COMMON_META(AtanMeta)

#undef UNARY_FLOAT_COMMON_META

#define UNARY_FLOAT_COMMON_IMPL(FUNC, MODE, STUB)                   \
  void FUNC(MusaTensorIterator& iter, const std::string& op_name) { \
    const auto dtype = iter.common_dtype();                         \
    if (c10::isComplexType(dtype) || dtype == ScalarType::Double) { \
      native::STUB##_stub(iter.device_type(), iter);                \
      return;                                                       \
    }                                                               \
    UnaryCall(iter, MODE, op_name);                                 \
  }

UNARY_FLOAT_COMMON_IMPL(ErfImpl, UNARY_MODE::ERF, erf)
UNARY_FLOAT_COMMON_IMPL(LogImpl, UNARY_MODE::LOG, log)
UNARY_FLOAT_COMMON_IMPL(Log2Impl, UNARY_MODE::LOG2, log2)
UNARY_FLOAT_COMMON_IMPL(Log10Impl, UNARY_MODE::LOG10, log10)
UNARY_FLOAT_COMMON_IMPL(SqrtImpl, UNARY_MODE::SQRT, sqrt)
UNARY_FLOAT_COMMON_IMPL(RsqrtImpl, UNARY_MODE::RSQRT, rsqrt)
UNARY_FLOAT_COMMON_IMPL(TanhImpl, UNARY_MODE::TANH, tanh)
UNARY_FLOAT_COMMON_IMPL(TanImpl, UNARY_MODE::TAN, tan)
UNARY_FLOAT_COMMON_IMPL(SigmoidImpl, UNARY_MODE::SIGMOID, sigmoid)
UNARY_FLOAT_COMMON_IMPL(ExpImpl, UNARY_MODE::EXP, exp)
UNARY_FLOAT_COMMON_IMPL(SinImpl, UNARY_MODE::SIN, sin)
UNARY_FLOAT_COMMON_IMPL(CosImpl, UNARY_MODE::COS, cos)
UNARY_FLOAT_COMMON_IMPL(AcosImpl, UNARY_MODE::ACOS, acos)
UNARY_FLOAT_COMMON_IMPL(AtanImpl, UNARY_MODE::ATAN, atan)

#undef UNARY_FLOAT_COMMON_IMPL

void ReciprocalImpl(MusaTensorIterator& iter, const std::string& op_name) {
  const auto dtype = iter.common_dtype();
  if (c10::isComplexType(dtype) || dtype == ScalarType::Double) {
    native::reciprocal_stub(iter.device_type(), iter);
    return;
  }
  const auto alpha = Scalar(static_cast<double>(-1.0));
  UnaryAlphaCall(iter, alpha, UNARY_MODE::POW, op_name);
}

#define GEN_IMPL(TPL, FUNC, META, IMPL) \
  TPL void FUNC(                        \
      MusaTensorIterator& iter,         \
      const Tensor& out,                \
      const Tensor& inp,                \
      const std::string& op_name) {     \
    META(iter, out, inp);               \
    if (iter.numel() != 0) {            \
      IMPL(iter, op_name);              \
    }                                   \
    iter.cast_outputs();                \
  }

GEN_IMPL(, UnaryReciprocal, ReciprocalMeta, ReciprocalImpl)
GEN_IMPL(, UnaryErf, ErfMeta, ErfImpl)
GEN_IMPL(, UnaryLog, LogMeta, LogImpl)
GEN_IMPL(, UnaryLog2, Log2Meta, Log2Impl)
GEN_IMPL(, UnaryLog10, Log10Meta, Log10Impl)
GEN_IMPL(, UnarySqrt, SqrtMeta, SqrtImpl)
GEN_IMPL(, UnaryRsqrt, RsqrtMeta, RsqrtImpl)
GEN_IMPL(, UnaryTanh, TanhMeta, TanhImpl)
GEN_IMPL(, UnaryTan, TanMeta, TanImpl)
GEN_IMPL(, UnarySigmoid, SigmoidMeta, SigmoidImpl)
GEN_IMPL(, UnaryExp, ExpMeta, ExpImpl)
GEN_IMPL(, UnarySin, SinMeta, SinImpl)
GEN_IMPL(, UnaryCos, CosMeta, CosImpl)
GEN_IMPL(, UnaryAcos, AcosMeta, AcosImpl)
GEN_IMPL(, UnaryAtan, AtanMeta, AtanImpl)

#undef GEN_IMPL

} // anonymous namespace

#define GEN_FUNCTION(OP, FUNC)                          \
  Tensor OP(const Tensor& self) {                       \
    FunctionalTensorIterator iter;                      \
    FUNC(iter, Tensor(), self, __func__);               \
    return iter.output();                               \
  }                                                     \
                                                        \
  Tensor& OP##_(Tensor& self) {                         \
    InplaceTensorIterator iter;                         \
    FUNC(iter, self, self, __func__);                   \
    return self;                                        \
  }                                                     \
                                                        \
  Tensor& OP##Out(const Tensor& self, Tensor& output) { \
    OutTensorIterator iter;                             \
    FUNC(iter, output, self, __func__);                 \
    return output;                                      \
  }

GEN_FUNCTION(Reciprocal, UnaryReciprocal)
GEN_FUNCTION(Erf, UnaryErf)
GEN_FUNCTION(Log, UnaryLog)
GEN_FUNCTION(Log2, UnaryLog2)
GEN_FUNCTION(Log10, UnaryLog10)
GEN_FUNCTION(Sqrt, UnarySqrt)
GEN_FUNCTION(Rsqrt, UnaryRsqrt)
GEN_FUNCTION(Tanh, UnaryTanh)
GEN_FUNCTION(Tan, UnaryTan)
GEN_FUNCTION(Sigmoid, UnarySigmoid)
GEN_FUNCTION(Exp, UnaryExp)
GEN_FUNCTION(Sin, UnarySin)
GEN_FUNCTION(Cos, UnaryCos)
GEN_FUNCTION(Acos, UnaryAcos)
GEN_FUNCTION(Atan, UnaryAtan)

#undef GEN_FUNCTION

} // namespace at::musa
