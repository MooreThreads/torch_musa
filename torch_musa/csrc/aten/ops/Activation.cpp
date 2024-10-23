#include <ATen/Config.h>
#include <ATen/native/UnaryOps.h>
// clang-format off
// Some classes in NativeFunctions.h require the corrosponding definition in Exception.h
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <torch/torch.h>

// clang-format on
#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/abs_native.h>
#include <ATen/ops/leaky_relu_backward_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <limits>

namespace at {
namespace musa {

namespace {

template <typename Stub>
static inline Tensor& UnaryOpOutImplComplex2Float(
    Tensor& result,
    const Tensor& self,
    Stub& stub) {
  TORCH_CHECK(
      self.is_complex(),
      "Unary op through musa kernel expects complex input dtype, but got ",
      self.scalar_type());

  if (!result.is_complex()) {
    // Checks if the corresponding float type can be cast to the desired dtype
    const auto float_type = c10::toRealValueType(self.scalar_type());
    TORCH_CHECK(
        canCast(float_type, result.scalar_type()),
        "result type ",
        float_type,
        " can't be cast to the desired output type ",
        result.scalar_type());

    // Runs the function complex->complex, as TensorIterator expects
    Tensor complex_result = at::empty({0}, self.options());
    auto iter = TensorIterator::unary_op(complex_result, self);
    stub(iter.device_type(), iter);

    // Copies the complex result to the actual result and returns it
    at::native::resize_output(result, complex_result.sizes());
    result.copy_(at::real(complex_result));
    return result;
  }

  auto iter = TensorIterator::unary_op(result, self);
  stub(iter.device_type(), iter);
  return result;
}

static inline Tensor& MusaAbsOutWithKernelComplex2Float(
    const Tensor& self,
    Tensor& result) {
  return UnaryOpOutImplComplex2Float(result, self, at::native::abs_stub);
}

static inline Tensor MusaAbsWithKernelComplex2Float(const Tensor& self) {
  const auto float_type = c10::toRealValueType(self.scalar_type());
  Tensor result = at::empty_like(self, self.options().dtype(float_type));
  return MusaAbsOutWithKernelComplex2Float(self, result);
}

} // anonymous namespace

using UNARY_MODE = ::musa::dnn::Unary::Mode;

void UnaryCall(
    const std::string& op_name,
    Tensor& o,
    const Tensor& i,
    std::function<void(::musa::dnn::Unary&)> func) {
  if (C10_UNLIKELY(i.numel() == 0)) {
    return;
  }
  auto in = CreateMUTensor(i);
  auto out = CreateMUTensor(o);

  ::musa::dnn::Unary op;
  func(op);

  muHandle& h = GetMudnnHandle();
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run " + op_name);
}

void UnaryBoolCall(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
  UnaryCall(op_name, output, input, [&](::musa::dnn::Unary& op) {
    if (isIntegralType(value.type(), true)) {
      CHECK_MUDNN_STATUS(op.SetAlpha(value.to<int64_t>()), "SetAlpha");
    } else {
      CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    }
    CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  });
}

Tensor UnaryBool(
    const std::string& op_name,
    const Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
  if (C10_UNLIKELY(input.numel() == 0)) {
    return at::empty(
        {input.sizes()}, input.options().dtype(c10::ScalarType::Bool));
  }
  Tensor input_tmp;
  const bool is_input_transpose_contig = IsTranspose(input, false);
  if (is_input_transpose_contig) {
    input_tmp = input.transpose(-1, -2);
  } else {
    input_tmp = input;
  }
  Tensor output = at::empty_like(
      input_tmp,
      input_tmp.options()
          .dtype(ScalarType::Bool)
          .memory_format(input_tmp.suggest_memory_format()));
  UnaryBoolCall(op_name, output, input_tmp, value, mode);
  if (is_input_transpose_contig) {
    output.transpose_(-1, -2);
  }
  return output;
}

void UnaryBool_(
    const std::string& op_name,
    Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
  const bool is_transpose_contig = IsTranspose(input, false);
  if (is_transpose_contig) {
    input.transpose_(-1, -2);
  }
  Tensor output = at::empty_like(
      input,
      input.options()
          .dtype(ScalarType::Bool)
          .memory_format(input.suggest_memory_format()));
  UnaryBoolCall(op_name, output, input, value, mode);
  input.copy_(output);
  if (is_transpose_contig) {
    input.transpose_(-1, -2);
  }
}

void UnaryBoolOut(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
  const bool is_transpose_contig =
      IsTranspose(input, false) && IsTranspose(output, false);
  Tensor input_tmp;
  at::MemoryFormat output_memory_format;
  if (is_transpose_contig) {
    output.transpose_(-1, -2);
    output_memory_format = output.suggest_memory_format();
    input_tmp = input.transpose(-1, -2);
  } else {
    output_memory_format = output.suggest_memory_format();
    input_tmp = input.suggest_memory_format() == output_memory_format
        ? input
        : FormatContiguous(input, output_memory_format);
  }
  UnaryBoolCall(op_name, output, input_tmp, value, mode);
  if (is_transpose_contig) {
    output.transpose_(-1, -2);
  }
}

Tensor Unary(
    const std::string& op_name,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  if (C10_UNLIKELY(input.is_complex())) {
    if (op_name == "Abs") {
      return MusaAbsWithKernelComplex2Float(input);
    }
  }

  MUSA_TENSOR_TYPE_CHECK(input);
  Tensor input_tmp;
  const bool is_input_transpose_contig = IsTranspose(input, false);
  if (is_input_transpose_contig) {
    input_tmp = input.transpose(-1, -2);
  } else {
    input_tmp = input;
  }
  Tensor output = at::empty_like(
      input_tmp,
      input_tmp.options().memory_format(input_tmp.suggest_memory_format()));
  UnaryCall(op_name, output, input_tmp, func);
  if (is_input_transpose_contig) {
    output.transpose_(-1, -2);
  }
  return output;
}

void Unary_(
    const std::string& op_name,
    Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  const bool is_transpose_contig = IsTranspose(input, false);
  if (is_transpose_contig) {
    input.transpose_(-1, -2);
  }
  UnaryCall(op_name, input, input, func);
  if (is_transpose_contig) {
    input.transpose_(-1, -2);
  }
}

void UnaryOut(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  output.resize_as_(input);
  at::MemoryFormat output_memory_format;
  const bool is_transpose_contig =
      IsTranspose(input, false) && IsTranspose(output, false);
  Tensor input_tmp;
  if (is_transpose_contig) {
    output.transpose_(-1, -2);
    output_memory_format = output.suggest_memory_format();
    input_tmp = input.transpose(-1, -2);
  } else {
    output_memory_format = output.suggest_memory_format();
    input_tmp = input.suggest_memory_format() == output_memory_format
        ? input
        : FormatContiguous(input, output_memory_format);
  }
  UnaryCall(op_name, output, input_tmp, func);
  if (is_transpose_contig) {
    output.transpose_(-1, -2);
  }
}

#define DEFINE_ACTIVATE_OP_ARGS(op_name, mode, alpha, beta)        \
  Tensor op_name(const Tensor& input) {                            \
    const c10::musa::MUSAGuard device_guard(input.device());       \
    return Unary(__func__, input, [](::musa::dnn::Unary& op) {     \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
      CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");          \
      CHECK_MUDNN_STATUS(op.SetBeta(beta), "SetBeta");             \
    });                                                            \
  }                                                                \
                                                                   \
  Tensor& op_name##_(Tensor& input) {                              \
    const c10::musa::MUSAGuard device_guard(input.device());       \
    Unary_(__func__, input, [](::musa::dnn::Unary& op) {           \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
      CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");          \
      CHECK_MUDNN_STATUS(op.SetBeta(beta), "SetBeta");             \
    });                                                            \
    return input;                                                  \
  }                                                                \
                                                                   \
  Tensor& op_name##Out(const Tensor& input, Tensor& output) {      \
    const c10::musa::MUSAGuard device_guard(input.device());       \
    UnaryOut(__func__, output, input, [](::musa::dnn::Unary& op) { \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
      CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");          \
      CHECK_MUDNN_STATUS(op.SetBeta(beta), "SetBeta");             \
    });                                                            \
    return output;                                                 \
  }

#define DEFINE_ACTIVATE_OP(op_name, mode) \
  DEFINE_ACTIVATE_OP_ARGS(op_name, mode, 0., 0.)

DEFINE_ACTIVATE_OP(Relu, ::musa::dnn::Unary::Mode::RELU)
DEFINE_ACTIVATE_OP(Silu, ::musa::dnn::Unary::Mode::SILU)
DEFINE_ACTIVATE_OP(Sqrt, ::musa::dnn::Unary::Mode::SQRT)
DEFINE_ACTIVATE_OP(Round, ::musa::dnn::Unary::Mode::ROUND)
DEFINE_ACTIVATE_OP(Rsqrt, ::musa::dnn::Unary::Mode::RSQRT)
DEFINE_ACTIVATE_OP(HardSwish, ::musa::dnn::Unary::Mode::HARDSWISH)
DEFINE_ACTIVATE_OP(Tanh, ::musa::dnn::Unary::Mode::TANH)
DEFINE_ACTIVATE_OP(Tan, ::musa::dnn::Unary::Mode::TAN)
DEFINE_ACTIVATE_OP(Sigmoid, ::musa::dnn::Unary::Mode::SIGMOID)
DEFINE_ACTIVATE_OP(Exp, ::musa::dnn::Unary::Mode::EXP)
DEFINE_ACTIVATE_OP(Sin, ::musa::dnn::Unary::Mode::SIN)
DEFINE_ACTIVATE_OP(Cos, ::musa::dnn::Unary::Mode::COS)
DEFINE_ACTIVATE_OP(Abs, ::musa::dnn::Unary::Mode::ABS)
DEFINE_ACTIVATE_OP(Acos, ::musa::dnn::Unary::Mode::ACOS)
DEFINE_ACTIVATE_OP(Atan, ::musa::dnn::Unary::Mode::ATAN)
DEFINE_ACTIVATE_OP(Ceil, ::musa::dnn::Unary::Mode::CEIL)
DEFINE_ACTIVATE_OP(Log, ::musa::dnn::Unary::Mode::LOG)
DEFINE_ACTIVATE_OP(Log10, ::musa::dnn::Unary::Mode::LOG10)
DEFINE_ACTIVATE_OP(Log2, ::musa::dnn::Unary::Mode::LOG2)
DEFINE_ACTIVATE_OP(Floor, ::musa::dnn::Unary::Mode::FLOOR)
DEFINE_ACTIVATE_OP_ARGS(
    HardSigmoid,
    ::musa::dnn::Unary::Mode::HARDSIGMOID,
    0.166667,
    0.5)

#define SCALAR_COMPARISON(op_name, mode)                         \
  Tensor& op_name##Out(                                          \
      const Tensor& self, const Scalar& value, Tensor& output) { \
    const c10::musa::MUSAGuard device_guard(self.device());      \
    UnaryBoolOut(__func__, output, self, value, mode);           \
    return output;                                               \
  }                                                              \
                                                                 \
  Tensor op_name(const Tensor& self, const Scalar& value) {      \
    const c10::musa::MUSAGuard device_guard(self.device());      \
    return UnaryBool(__func__, self, value, mode);               \
  }                                                              \
                                                                 \
  Tensor& op_name##_(Tensor& self, const Scalar& value) {        \
    const c10::musa::MUSAGuard device_guard(self.device());      \
    UnaryBool_(__func__, self, value, mode);                     \
    return self;                                                 \
  }

SCALAR_COMPARISON(LeScalar, UNARY_MODE::LE)
SCALAR_COMPARISON(LtScalar, UNARY_MODE::LT)
SCALAR_COMPARISON(GeScalar, UNARY_MODE::GE)
SCALAR_COMPARISON(GtScalar, UNARY_MODE::GT)
SCALAR_COMPARISON(EqScalar, UNARY_MODE::EQ)
SCALAR_COMPARISON(NeScalar, UNARY_MODE::NE)

/**
 * auto poscoef = scale.to<float>(); // default this to 1
 * auto negiptcoef = input_scale.to<float>();
 * Original PyTorch implementation follows:
 * out = out <= 0 ? (exp(input * negiptcoef) - 1) * negcoef : input *
 * postcoef
 * this equation is a little different with the ELU definition: out
 * = out <= 0 ? (exp(input) - 1) * alpha : input Unfortunately, mudnn didn't
 * support the upper one. Fortunately, Elu will always be valid.
 */
at::Tensor Elu(
    const at::Tensor& self,
    const c10::Scalar& alpha,
    const c10::Scalar& scale,
    const c10::Scalar& input_scale) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    auto negcoef = alpha.to<float>() * scale.to<float>();
    op.SetAlpha(negcoef);
    op.SetMode(::musa::dnn::Unary::Mode::ELU);
  });
}
at::Tensor& Elu_(
    at::Tensor& self,
    const c10::Scalar& alpha,
    const c10::Scalar& scale,
    const c10::Scalar& input_scale) {
  const c10::musa::MUSAGuard device_guard(self.device());
  Unary_(__func__, self, [&](::musa::dnn::Unary& op) {
    auto negcoef = alpha.to<float>() * scale.to<float>();
    op.SetAlpha(negcoef);
    op.SetMode(::musa::dnn::Unary::Mode::ELU);
  });
  return self;
}

at::Tensor& EluOut(
    const at::Tensor& self,
    const c10::Scalar& alpha,
    const c10::Scalar& scale,
    const c10::Scalar& input_scale,
    at::Tensor& result) {
  const c10::musa::MUSAGuard device_guard(self.device());
  UnaryOut(__func__, result, self, [&](::musa::dnn::Unary& op) {
    auto negcoef = alpha.to<float>() * scale.to<float>();
    op.SetAlpha(negcoef);
    op.SetMode(::musa::dnn::Unary::Mode::ELU);
  });
  return result;
}

Tensor Gelu(const Tensor& self, c10::string_view approximate) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  auto mode = approximate_type == at::native::GeluType::None
      ? ::musa::dnn::Unary::Mode::GELU
      : ::musa::dnn::Unary::Mode::GELU_TANH;

  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  });
}

Tensor& Gelu_(Tensor& self, c10::string_view approximate) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  auto mode = approximate_type == at::native::GeluType::None
      ? ::musa::dnn::Unary::Mode::GELU
      : ::musa::dnn::Unary::Mode::GELU_TANH;
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  });
  return self;
}

Tensor& GeluOut(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& output) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  auto mode = approximate_type == at::native::GeluType::None
      ? ::musa::dnn::Unary::Mode::GELU
      : ::musa::dnn::Unary::Mode::GELU_TANH;
  const c10::musa::MUSAGuard device_guard(self.device());
  UnaryOut(__func__, output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  });
  return output;
}

Tensor Reciprocal(const Tensor& self) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
}

Tensor& Reciprocal_(Tensor& self) {
  const c10::musa::MUSAGuard device_guard(self.device());
  Unary_(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return self;
}

Tensor& ReciprocalOut(const Tensor& self, Tensor& output) {
  const c10::musa::MUSAGuard device_guard(self.device());
  UnaryOut(__func__, output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return output;
}

void NegCall(
    const std::string& op_name,
    Tensor& out,
    const Tensor& self,
    const c10::optional<Scalar>& val,
    bool self_output = false) {
  const auto t_type = self.scalar_type();
  Tensor input;
  const bool isT =
      IsTranspose(self, false) && (self_output || IsTranspose(out, false));
  at::MemoryFormat output_memory_format;

  if (isT) {
    out.transpose_(-1, -2);
    input = self_output ? self : self.transpose(-1, -2);
    output_memory_format = out.suggest_memory_format();
  } else {
    output_memory_format = out.suggest_memory_format();
    input =
        (self_output || (self.suggest_memory_format() == output_memory_format))
        ? self
        : FormatContiguous(self, output_memory_format);
  }
  switch (t_type) {
    case ScalarType::Float:
    case ScalarType::Half:
    case ScalarType::BFloat16: {
      const double alpha = val.value().to<double>();
      UnaryCall(op_name, out, input, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Int:
    case ScalarType::Long: {
      const int64_t alpha = val.value().to<int64_t>();
      UnaryCall(op_name, out, input, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype in Neg: ", t_type);
  }
  if (isT) {
    out.transpose_(-1, -2);
  }
}

Tensor Neg(const Tensor& self) {
  const c10::musa::MUSAGuard device_guard(self.device());
  MUSA_TENSOR_TYPE_CHECK(self);
  Tensor output =
      at::empty_like(self, self.options(), self.suggest_memory_format());
  Scalar val = -1;
  NegCall(__func__, output, self, val);
  return output;
}

Tensor& Neg_(Tensor& self) {
  const c10::musa::MUSAGuard device_guard(self.device());
  MUSA_TENSOR_TYPE_CHECK(self);
  Scalar val = -1;
  NegCall(__func__, self, self, val, true);
  return self;
}

Tensor& NegOut(const Tensor& self, Tensor& out) {
  const c10::musa::MUSAGuard device_guard(self.device());
  MUSA_TENSOR_TYPE_CHECK(self);
  Scalar val = -1;
  out.resize_as_(self);
  NegCall(__func__, out, self, val);
  return out;
}

Tensor LogicalNot(const Tensor& self) {
  return at::eq(self, 0).to(ScalarType::Bool);
}

Tensor& LogicalNot_(Tensor& self) {
  // TODO(@mt-ai): use inplace op to avoid overhead
  auto out_tmp = LogicalNot(self);
  self.resize_as_(out_tmp);
  self.copy_(out_tmp);
  return self;
}

Tensor& LogicalNotOut(const Tensor& self, Tensor& out) {
  // TODO(@mt-ai): use inplace op to avoid overhead
  auto out_tmp = LogicalNot(self).to(out.scalar_type());
  out.resize_as_(out_tmp);
  out.copy_(out_tmp);
  return out;
}

Tensor PowScalar(const Tensor& self, const Scalar& value) {
  const c10::musa::MUSAGuard device_guard(self.device());
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
}

Tensor& PowScalar_(Tensor& self, const Scalar& value) {
  const c10::musa::MUSAGuard device_guard(self.device());
  Unary_("pow_.Scalar", self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return self;
}

Tensor& PowScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  const c10::musa::MUSAGuard device_guard(self.device());
  UnaryOut("pow.Tensor_Scalar_out", output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return output;
}

Tensor LeakyRelu(const Tensor& input, const Scalar& neg_slope) {
  const c10::musa::MUSAGuard device_guard(input.device());
  return Unary(__func__, input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
}

Tensor& LeakyRelu_(Tensor& input, const Scalar& neg_slope) {
  const c10::musa::MUSAGuard device_guard(input.device());
  Unary_("leaky_relu_", input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
  return input;
}

Tensor& LeakyReluOut(
    const Tensor& input,
    const Scalar& neg_slope,
    Tensor& output) {
  const c10::musa::MUSAGuard device_guard(input.device());
  UnaryOut("leaky_relu.out", output, input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
  return output;
}

at::Tensor HardTanh(
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::hardtanh(self, min_val, max_val);
}

at::Tensor& HardTanh_(
    at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::hardtanh_(self, min_val, max_val);
}

at::Tensor& HardTanhOut(
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::hardtanh_out(self, min_val, max_val, out);
}

at::Tensor HardTanhBackward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::hardtanh_backward(grad_output, self, min_val, max_val);
}

at::Tensor& HardTanhBackwardOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& min_val,
    const at::Scalar& max_val,
    at::Tensor& grad_input) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::hardtanh_backward_out(
      grad_output, self, min_val, max_val, grad_input);
}

at::Tensor PRelu(const at::Tensor& self, const at::Tensor& weight) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "PRelu", "self");
  c10::impl::check_and_update_common_device(
      common_device, weight, "PRelu", "weight");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_prelu_kernel(self, weight);
}

::std::tuple<at::Tensor, at::Tensor> PReluBackward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Tensor& weight) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, grad_output, "PReluBackward", "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "PReluBackward", "self");
  c10::impl::check_and_update_common_device(
      common_device, weight, "PReluBackward", "weight");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::_prelu_kernel_backward(grad_output, self, weight);
}

at::Tensor IsNan(const at::Tensor& self) {
  // DeviceGuard omitted
  return self != self;
}

Tensor HardSwishBwd(const Tensor& grad_output, const Tensor& self) {
  return at::native::hardswish_backward(grad_output, self);
}

} // namespace musa
} // namespace at
