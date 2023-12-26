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
#include "torch_musa/csrc/utils/register_wrapper.h"

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
    if (input.scalar_type() == ScalarType::Long ||
        input.scalar_type() == ScalarType::Byte ||
        input.scalar_type() == ScalarType::Bool) {
      CHECK_MUDNN_STATUS(op.SetAlpha(value.to<int64_t>()), "SetAlpha");
    } else {
      CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    }
    CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  });
}

// TODO(songtao.liu): muDNN requires bool tensor for lt/le... output,
// which cannot be autocast in muDNN now.
Tensor UnaryBool(
    const std::string& op_name,
    const Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
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
  UnaryBoolCall(op_name, input, input, value, mode);
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

Tensor GELU(const Tensor& self, c10::string_view approximate) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
}

Tensor& GELU_(Tensor& self, c10::string_view approximate) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
  return self;
}

Tensor& GELUOut(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& output) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  const c10::musa::MUSAGuard device_guard(self.device());
  UnaryOut(__func__, output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
  return output;
}

void ClampCall(
    const std::string& op_name,
    Tensor& out,
    const Tensor& self,
    bool has_min,
    const c10::optional<Scalar>& min,
    bool has_max,
    const c10::optional<Scalar>& max) {
  const auto t_type = self.scalar_type();
  switch (t_type) {
    case ScalarType::Float: {
      // DBL_MIN = 2.22507e-308 which is positive, so we must use lowest or
      // (-max) there !!!
      const double min_val = has_min ? min.value().to<double>()
                                     : std::numeric_limits<double>::lowest();

      const double max_val = has_max ? max.value().to<double>()
                                     : std::numeric_limits<double>::max();
      UnaryCall(op_name, out, self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
      break;
    }
    case ScalarType::Long: {
      // LONG_MIN = -9223372036854775808, LONG_MAX = 9223372036854775807
      const int64_t min_val = has_min ? min.value().to<int64_t>()
                                      : std::numeric_limits<int64_t>::min();
      const int64_t max_val = has_max ? max.value().to<int64_t>()
                                      : std::numeric_limits<int64_t>::max();
      UnaryCall(op_name, out, self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
      break;
    }
    case ScalarType::Int: {
      // TODO(@fan.mo): mudnn currently doesn't support INT32 CLIP
      // INT_MIN = - 2**32, INT_MAX = 2**32 - 1
      const int32_t min_val = has_min ? min.value().to<int32_t>()
                                      : std::numeric_limits<int32_t>::min();
      const int32_t max_val = has_max ? max.value().to<int32_t>()
                                      : std::numeric_limits<int32_t>::max();
      int64_t min_val_ = (int64_t)min_val;
      int64_t max_val_ = (int64_t)max_val;
      const Tensor self_ = self.to(ScalarType::Long);
      Tensor out_ = out.to(ScalarType::Long);
      UnaryCall(op_name, out_, self_, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val_), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val_), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
      out = out_.to(ScalarType::Int);
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype: ", t_type);
      throw;
  }
}

Tensor Clamp(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  const bool has_min = (min.has_value());
  const bool has_max = (max.has_value());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: either min, max or both scalars must be defined")
  const c10::musa::MUSAGuard device_guard(self.device());
  // TODO(jing.li): eliminate fp32 conversion workaround after muDNN supports
  // fp16 calculation.
  const bool self_fp16 = (self.scalar_type() == ScalarType::Half);
  Tensor input = self_fp16 ? self.to(ScalarType::Float) : self;
  const bool is_transpose_contig = IsTranspose(input, false);
  if (is_transpose_contig) {
    input.transpose_(-1, -2);
  }
  Tensor output = at::empty_like(
      self,
      c10::TensorOptions(input.suggest_memory_format())
          .dtype(input.scalar_type()));

  MUSA_TENSOR_TYPE_CHECK(self);
  ClampCall(__func__, output, input, has_min, min, has_max, max);

  if (is_transpose_contig) {
    output.transpose_(-1, -2);
  }
  if (self_fp16) {
    return output.to(ScalarType::Half);
  }
  return output;
}

Tensor& MudnnClamp_(
    Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max) {
  const bool has_min = (min.has_value());
  const bool has_max = (max.has_value());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: either min, max or both scalars must be defined")
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  ClampCall(__func__, self, self, has_min, min, has_max, max);
  return self;
}

namespace {
struct structured_clamp_out_inplace final
    : public at::native::structured_clamp_out {
  structured_clamp_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_clamp_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_clamp_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor& Clamp_(
    at::Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max) {
  // No device check
  structured_clamp_out_inplace op(self);
  op.meta(
      self,
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef()),
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef()));
  op.impl(
      self,
      (min.has_value() ? at::OptionalScalarRef(&(min.value()))
                       : at::OptionalScalarRef()),
      (max.has_value() ? at::OptionalScalarRef(&(max.value()))
                       : at::OptionalScalarRef()),
      op.outputs_[0]);
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}

Tensor& ClampOut(
    const Tensor& self,
    const c10::optional<Scalar>& min,
    const c10::optional<Scalar>& max,
    Tensor& out) {
  const bool has_min = (min.has_value());
  const bool has_max = (max.has_value());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: either min, max or both scalars must be defined")
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  out.resize_as_(self);
  at::MemoryFormat output_memory_format;
  const bool is_transpose_contig =
      IsTranspose(self, false) && IsTranspose(out, false);
  Tensor input;
  if (is_transpose_contig) {
    out.transpose_(-1, -2);
    output_memory_format = out.suggest_memory_format();
    input = self.transpose(-1, -2);
  } else {
    output_memory_format = out.suggest_memory_format();
    input = self.suggest_memory_format() == output_memory_format
        ? self
        : FormatContiguous(self, output_memory_format);
  }
  ClampCall(__func__, out, input, has_min, min, has_max, max);
  if (is_transpose_contig) {
    out.transpose_(-1, -2);
  }
  return out;
}

Tensor& ClampTensorOut(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max,
    Tensor& out) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Clamp must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      out.device().type() == kMUSA,
      "Device of output tensor of Clamp.TensorOut must be MUSA, but now is ",
      out.device());
  auto has_min = min.has_value() && min->defined();
  auto has_max = max.has_value() && max->defined();
  const c10::musa::MUSAGuard device_guard(self.device());
  Tensor cpu_min;
  Tensor cpu_max;
  if (has_min) {
    TORCH_CHECK(
        min->device().type() == kMUSA,
        "Device of min tensor of Clamp.TensorOut must be MUSA, but now is ",
        min->device());
    cpu_min = min->to("cpu");
  }
  if (has_max) {
    TORCH_CHECK(
        max->device().type() == kMUSA,
        "Device of max tensor of Clamp.TensorOut must be MUSA, but now is ",
        max->device());
    cpu_max = max->to("cpu");
  }
  out.resize_(self.sizes());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: At least one of 'min' or 'max' must not be None");
  auto cpu_self = self.to("cpu");
  const auto cpu_min_ref =
      has_min ? c10::optional<Tensor>(cpu_min) : c10::optional<Tensor>();
  const auto cpu_max_ref =
      has_max ? c10::optional<Tensor>(cpu_max) : c10::optional<Tensor>();
  auto cpu_result = clamp(cpu_self, cpu_min_ref, cpu_max_ref);
  out.copy_(cpu_result);
  return out;
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

namespace {
struct structured_softplus_out_functional final
    : public at::native::structured_softplus_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_softplus_out_out final
    : public at::native::structured_softplus_out {
  structured_softplus_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_softplus_backward_out_functional final
    : public at::native::structured_softplus_backward_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_softplus_backward_out_out final
    : public at::native::structured_softplus_backward_out {
  structured_softplus_backward_out_out(Tensor& out0)
      : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_softplus_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor& SoftPlusOut(
    const at::Tensor& self,
    const c10::Scalar& beta,
    const c10::Scalar& threshold,
    at::Tensor& out) {
  structured_softplus_out_out op(out);
  op.meta(self, beta, threshold);
  op.impl(self, beta, threshold, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

at::Tensor SoftPlus(
    const at::Tensor& self,
    const c10::Scalar& beta,
    const c10::Scalar& threshold) {
  structured_softplus_out_functional op;
  op.meta(self, beta, threshold);
  op.impl(self, beta, threshold, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& wrapper_MUSA_softplus_backward_out_grad_input(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_input,
      "wrapper_MUSA_softplus_backward_out_grad_input",
      "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "wrapper_MUSA_softplus_backward_out_grad_input",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device,
      self,
      "wrapper_MUSA_softplus_backward_out_grad_input",
      "self");
  structured_softplus_backward_out_out op(grad_input);
  op.meta(grad_output, self, beta, threshold);
  op.impl(grad_output, self, beta, threshold, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

at::Tensor wrapper_MUSA_softplus_backward(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& beta,
    const at::Scalar& threshold) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "wrapper_MUSA_softplus_backward",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_MUSA_softplus_backward", "self");
  structured_softplus_backward_out_functional op;
  op.meta(grad_output, self, beta, threshold);
  op.impl(grad_output, self, beta, threshold, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
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
    case ScalarType::Float: {
      const double alpha = val.value().to<double>();
      UnaryCall(op_name, out, input, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Half: {
      const double alpha = val.value().to<double>();
      UnaryCall(op_name, out, input, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Int: {
      const int64_t alpha = val.value().to<int64_t>();
      UnaryCall(op_name, out, input, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
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
  auto out_tmp = LogicalNot(self);
  self.resize_as_(out_tmp);
  self.copy_(out_tmp);
  return self;
}

Tensor& LogicalNotOut(const Tensor& self, Tensor& out) {
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

Tensor LeakyRelu(const Tensor& input, const Scalar& neg_slope = 0.01) {
  const c10::musa::MUSAGuard device_guard(input.device());
  return Unary(__func__, input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
}

Tensor& LeakyRelu_(Tensor& input, const Scalar& neg_slope = 0.01) {
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

namespace {
struct structured_leaky_relu_backward_out_out final
    : public at::native::structured_leaky_relu_backward_out {
  structured_leaky_relu_backward_out_out(Tensor& out0)
      : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_leaky_relu_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_leaky_relu_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor& LeakyReluBackwardOutGradInput(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    const at::Scalar& negative_slope,
    bool self_is_result,
    at::Tensor& grad_input) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, grad_input, "LeakyReluBackwardOutGradInput", "grad_input");
  c10::impl::check_and_update_common_device(
      common_device,
      grad_output,
      "LeakyReluBackwardOutGradInput",
      "grad_output");
  c10::impl::check_and_update_common_device(
      common_device, self, "LeakyReluBackwardOutGradInput", "self");
  structured_leaky_relu_backward_out_out op(grad_input);
  op.meta(grad_output, self, negative_slope, self_is_result);
  op.impl(
      grad_output,
      self,
      negative_slope,
      self_is_result,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

namespace {
struct structured_clamp_min_out_out final
    : public at::native::structured_clamp_min_out {
  structured_clamp_min_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_clamp_min_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    at::native::structured_clamp_min_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

Tensor& ClampMinOut(const Tensor& self, const Scalar& min, Tensor& out) {
  // No device check
  structured_clamp_min_out_out op(out);
  op.meta(self, min);
  op.impl(self, min, op.maybe_get_output(0));
  return out;
}

namespace {
struct structured_bitwise_not_out_functional final
    : public at::native::structured_bitwise_not_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_not_out_inplace final
    : public at::native::structured_bitwise_not_out {
  structured_bitwise_not_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_bitwise_not_out_out final
    : public at::native::structured_bitwise_not_out {
  structured_bitwise_not_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);

    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_bitwise_not_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor BitwiseNot(const at::Tensor& self) {
  // No device check
  structured_bitwise_not_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& BitwiseNot_(at::Tensor& self) {
  // No device check
  structured_bitwise_not_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  return self;
}

at::Tensor& BitwiseNotOut(const at::Tensor& self, at::Tensor& out) {
  // No device check
  structured_bitwise_not_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  return out;
}

namespace {
struct structured_sgn_out_functional final
    : public at::native::structured_sgn_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_sgn_out_inplace final
    : public at::native::structured_sgn_out {
  structured_sgn_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_sgn_out_out final : public at::native::structured_sgn_out {
  structured_sgn_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }

    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sgn_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor Sgn(const at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(common_device, self, "Sgn", "self");
  structured_sgn_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& Sgn_(at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "Sgn_", "self");
  structured_sgn_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  return self;
}

Tensor& SgnOut(const Tensor& self, Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "SgnOut", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "SgnOut", "self");
  structured_sgn_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  return out;
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

namespace {
struct structured_sign_out_out final : public at::native::structured_sign_out {
  structured_sign_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_sign_out_inplace final
    : public at::native::structured_sign_out {
  structured_sign_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    auto maybe_proxy = maybe_create_proxy(out, sizes, strides, options);
    if (C10_UNLIKELY(maybe_proxy.has_value())) {
      proxy_outputs_[output_idx] =
          c10::ExclusivelyOwned<Tensor>(std::move(maybe_proxy).value());
    }
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    check_inplace(out, sizes, options);
    if (!names.empty()) {
      namedinference::propagate_names(outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_sign_out_functional final
    : public at::native::structured_sign_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_sign_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor Sign(const at::Tensor& self) {
  // No device check
  structured_sign_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& Sign_(at::Tensor& self) {
  // No device check
  structured_sign_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}

at::Tensor& SignOut(const at::Tensor& self, at::Tensor& out) {
  // No device check
  structured_sign_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

namespace {
struct structured_hardsigmoid_backward_out_functional final
    : public at::native::structured_hardsigmoid_backward_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_hardsigmoid_backward_out_out final
    : public at::native::structured_hardsigmoid_backward_out {
  structured_hardsigmoid_backward_out_out(Tensor& out0)
      : outputs_{std::ref(out0)} {}
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    resize_out(out, sizes, strides, options);
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_hardsigmoid_backward_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};
} // namespace

at::Tensor HardSigmoidBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self) {
  structured_hardsigmoid_backward_out_functional op;
  op.meta(grad_output, self);
  op.impl(grad_output, self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& HardSigmoidBwdOut(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    at::Tensor& grad_input) {
  structured_hardsigmoid_backward_out_out op(grad_input);
  op.meta(grad_output, self);
  op.impl(grad_output, self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

Tensor HardSwishBwd(const Tensor& grad_output, const Tensor& self) {
  return at::native::hardswish_backward(grad_output, self);
}

ADVANCED_REGISTER(aten, PrivateUse1, "abs", Abs)
ADVANCED_REGISTER(aten, PrivateUse1, "abs_", Abs_)
ADVANCED_REGISTER(aten, PrivateUse1, "abs.out", AbsOut)

ADVANCED_REGISTER(aten, PrivateUse1, "sgn", Sgn)
ADVANCED_REGISTER(aten, PrivateUse1, "sgn_", Sgn_)
ADVANCED_REGISTER(aten, PrivateUse1, "sgn.out", SgnOut)

ADVANCED_REGISTER(aten, PrivateUse1, "sign", Sign)
ADVANCED_REGISTER(aten, PrivateUse1, "sign_", Sign_)
ADVANCED_REGISTER(aten, PrivateUse1, "sign.out", SignOut)

ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_not", BitwiseNot)
ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_not_", BitwiseNot_)
ADVANCED_REGISTER(aten, PrivateUse1, "bitwise_not.out", BitwiseNotOut)

ADVANCED_REGISTER(aten, PrivateUse1, "logical_not", LogicalNot)
ADVANCED_REGISTER(aten, PrivateUse1, "logical_not_", LogicalNot_)
ADVANCED_REGISTER(aten, PrivateUse1, "logical_not.out", LogicalNotOut)

ADVANCED_REGISTER(aten, PrivateUse1, "eq.Scalar", EqScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "eq_.Scalar", EqScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "eq.Scalar_out", EqScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "relu", Relu)
ADVANCED_REGISTER(aten, PrivateUse1, "relu_", Relu_)

ADVANCED_REGISTER(aten, PrivateUse1, "lt.Scalar", LtScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "lt_.Scalar", LtScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "lt.Scalar_out", LtScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "le.Scalar", LeScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "le_.Scalar", LeScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "le.Scalar_out", LeScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "ne.Scalar", NeScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "ne_.Scalar", NeScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "ne.Scalar_out", NeScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "gt.Scalar", GtScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "gt_.Scalar", GtScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "gt.Scalar_out", GtScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "ge.Scalar", GeScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "ge_.Scalar", GeScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "ge.Scalar_out", GeScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "sqrt", Sqrt)
ADVANCED_REGISTER(aten, PrivateUse1, "sqrt_", Sqrt_)
ADVANCED_REGISTER(aten, PrivateUse1, "sqrt.out", SqrtOut)

ADVANCED_REGISTER(aten, PrivateUse1, "round", Round)
ADVANCED_REGISTER(aten, PrivateUse1, "round_", Round_)
ADVANCED_REGISTER(aten, PrivateUse1, "round.out", RoundOut)

ADVANCED_REGISTER(aten, PrivateUse1, "rsqrt", Rsqrt)
ADVANCED_REGISTER(aten, PrivateUse1, "rsqrt_", Rsqrt_)
ADVANCED_REGISTER(aten, PrivateUse1, "rsqrt.out", RsqrtOut)

ADVANCED_REGISTER(aten, PrivateUse1, "hardswish", HardSwish)
ADVANCED_REGISTER(aten, PrivateUse1, "hardswish_", HardSwish_)
ADVANCED_REGISTER(aten, PrivateUse1, "hardswish.out", HardSwishOut)
ADVANCED_REGISTER(aten, PrivateUse1, "hardswish_backward", HardSwishBwd)

ADVANCED_REGISTER(aten, PrivateUse1, "hardsigmoid", HardSigmoid)
ADVANCED_REGISTER(aten, PrivateUse1, "hardsigmoid_", HardSigmoid_)
ADVANCED_REGISTER(aten, PrivateUse1, "hardsigmoid.out", HardSigmoidOut)
ADVANCED_REGISTER(aten, PrivateUse1, "hardsigmoid_backward", HardSigmoidBwd)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "hardsigmoid_backward.grad_input",
    HardSigmoidBwdOut)

ADVANCED_REGISTER(aten, PrivateUse1, "acos", Acos)
ADVANCED_REGISTER(aten, PrivateUse1, "acos_", Acos_)
ADVANCED_REGISTER(aten, PrivateUse1, "acos.out", AcosOut)

ADVANCED_REGISTER(aten, PrivateUse1, "tanh", Tanh)
ADVANCED_REGISTER(aten, PrivateUse1, "tanh_", Tanh_)
ADVANCED_REGISTER(aten, PrivateUse1, "tanh.out", TanhOut)

ADVANCED_REGISTER(aten, PrivateUse1, "tan", Tan)
ADVANCED_REGISTER(aten, PrivateUse1, "tan_", Tan_)
ADVANCED_REGISTER(aten, PrivateUse1, "tan.out", TanOut)

ADVANCED_REGISTER(aten, PrivateUse1, "atan", Atan)
ADVANCED_REGISTER(aten, PrivateUse1, "atan_", Atan_)
ADVANCED_REGISTER(aten, PrivateUse1, "atan.out", AtanOut)

ADVANCED_REGISTER(aten, PrivateUse1, "log", Log)
ADVANCED_REGISTER(aten, PrivateUse1, "log_", Log_)
ADVANCED_REGISTER(aten, PrivateUse1, "log.out", LogOut)

ADVANCED_REGISTER(aten, PrivateUse1, "log2", Log2)
ADVANCED_REGISTER(aten, PrivateUse1, "log2_", Log2_)
ADVANCED_REGISTER(aten, PrivateUse1, "log2.out", Log2Out)

ADVANCED_REGISTER(aten, PrivateUse1, "gelu", GELU)
ADVANCED_REGISTER(aten, PrivateUse1, "gelu_", GELU_)
ADVANCED_REGISTER(aten, PrivateUse1, "gelu.out", GELUOut)

ADVANCED_REGISTER(aten, PrivateUse1, "clamp", Clamp)
ADVANCED_REGISTER(aten, PrivateUse1, "clamp_", Clamp_)
ADVANCED_REGISTER(aten, PrivateUse1, "clamp.out", ClampOut)
ADVANCED_REGISTER(aten, PrivateUse1, "clamp.Tensor_out", ClampTensorOut)
ADVANCED_REGISTER(aten, PrivateUse1, "clamp_min.out", ClampMinOut)

ADVANCED_REGISTER(aten, PrivateUse1, "reciprocal", Reciprocal)
ADVANCED_REGISTER(aten, PrivateUse1, "reciprocal_", Reciprocal_)
ADVANCED_REGISTER(aten, PrivateUse1, "reciprocal.out", ReciprocalOut)

ADVANCED_REGISTER(aten, PrivateUse1, "sigmoid", Sigmoid)
ADVANCED_REGISTER(aten, PrivateUse1, "sigmoid_", Sigmoid_)
ADVANCED_REGISTER(aten, PrivateUse1, "sigmoid.out", SigmoidOut)

ADVANCED_REGISTER(aten, PrivateUse1, "ceil", Ceil)
ADVANCED_REGISTER(aten, PrivateUse1, "ceil_", Ceil_)
ADVANCED_REGISTER(aten, PrivateUse1, "ceil.out", CeilOut)

ADVANCED_REGISTER(aten, PrivateUse1, "exp", Exp)
ADVANCED_REGISTER(aten, PrivateUse1, "exp_", Exp_)
ADVANCED_REGISTER(aten, PrivateUse1, "exp.out", ExpOut)

ADVANCED_REGISTER(aten, PrivateUse1, "silu", Silu)
ADVANCED_REGISTER(aten, PrivateUse1, "silu_", Silu_)
ADVANCED_REGISTER(aten, PrivateUse1, "silu.out", SiluOut)

ADVANCED_REGISTER(aten, PrivateUse1, "cos", Cos)
ADVANCED_REGISTER(aten, PrivateUse1, "cos_", Cos_)
ADVANCED_REGISTER(aten, PrivateUse1, "cos.out", CosOut)

ADVANCED_REGISTER(aten, PrivateUse1, "sin", Sin)
ADVANCED_REGISTER(aten, PrivateUse1, "sin_", Sin_)
ADVANCED_REGISTER(aten, PrivateUse1, "sin.out", SinOut)

ADVANCED_REGISTER(aten, PrivateUse1, "neg", Neg)
ADVANCED_REGISTER(aten, PrivateUse1, "neg_", Neg_)
ADVANCED_REGISTER(aten, PrivateUse1, "neg.out", NegOut)

ADVANCED_REGISTER(aten, PrivateUse1, "pow.Tensor_Scalar", PowScalar)
ADVANCED_REGISTER(aten, PrivateUse1, "pow_.Scalar", PowScalar_)
ADVANCED_REGISTER(aten, PrivateUse1, "pow.Tensor_Scalar_out", PowScalarOut)

ADVANCED_REGISTER(aten, PrivateUse1, "leaky_relu", LeakyRelu)
ADVANCED_REGISTER(aten, PrivateUse1, "leaky_relu_", LeakyRelu_)
ADVANCED_REGISTER(aten, PrivateUse1, "leaky_relu.out", LeakyReluOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "leaky_relu_backward.grad_input",
    LeakyReluBackwardOutGradInput)

ADVANCED_REGISTER(aten, PrivateUse1, "log10", Log10)
ADVANCED_REGISTER(aten, PrivateUse1, "log10_", Log10_)
ADVANCED_REGISTER(aten, PrivateUse1, "log10.out", Log10Out)

ADVANCED_REGISTER(aten, PrivateUse1, "floor", Floor)
ADVANCED_REGISTER(aten, PrivateUse1, "floor_", Floor_)
ADVANCED_REGISTER(aten, PrivateUse1, "floor.out", FloorOut)

ADVANCED_REGISTER(aten, PrivateUse1, "elu", Elu)
ADVANCED_REGISTER(aten, PrivateUse1, "elu_", Elu_)
ADVANCED_REGISTER(aten, PrivateUse1, "elu.out", EluOut)

ADVANCED_REGISTER(aten, PrivateUse1, "hardtanh", HardTanh)
ADVANCED_REGISTER(aten, PrivateUse1, "hardtanh_", HardTanh_)
ADVANCED_REGISTER(aten, PrivateUse1, "hardtanh.out", HardTanhOut)
ADVANCED_REGISTER(aten, PrivateUse1, "hardtanh_backward", HardTanhBackward)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "hardtanh_backward.grad_input",
    HardTanhBackwardOut)

ADVANCED_REGISTER(aten, PrivateUse1, "_prelu_kernel", PRelu)
ADVANCED_REGISTER(aten, PrivateUse1, "_prelu_kernel_backward", PReluBackward)

ADVANCED_REGISTER(aten, PrivateUse1, "softplus", SoftPlus)
ADVANCED_REGISTER(aten, PrivateUse1, "softplus.out", SoftPlusOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "softplus_backward",
    wrapper_MUSA_softplus_backward)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "softplus_backward.grad_input",
    wrapper_MUSA_softplus_backward_out_grad_input)

ADVANCED_REGISTER(aten, PrivateUse1, "isnan", IsNan)

} // namespace musa
} // namespace at
