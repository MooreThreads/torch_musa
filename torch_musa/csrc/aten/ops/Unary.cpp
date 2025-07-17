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

void ReciprocalMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& inp) {
  InitUnaryIterator(iter, out, inp);
  TensorIteratorConfig config;
  SetUpUnaryFloatConfig(config);
  iter.build(config);
}

void ReciprocalImpl(MusaTensorIterator& iter, const std::string& op_name) {
  const auto alpha = Scalar(static_cast<double>(-1.0));
  UnaryAlphaCall(iter, alpha, UNARY_MODE::POW, op_name);
}

void ErfMeta(
    MusaTensorIterator& iter,
    const Tensor& out,
    const Tensor& inp) {
  InitUnaryIterator(iter, out, inp);
  TensorIteratorConfig config;
  SetUpUnaryFloatConfig(config);
  iter.build(config);
}

void ErfImpl(MusaTensorIterator& iter, const std::string& op_name) {
  UnaryCall(iter, UNARY_MODE::ERF, op_name);
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

#undef GEN_FUNCTION

} // namespace at::musa
