#include <ATen/Config.h>
// clang-format off
// Some classes in NativeFunctions.h require the corrosponding definition in Exception.h
#include <c10/util/Exception.h>
// clang-format on
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Activation.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
using UNARY_MODE = ::musa::dnn::Unary::Mode;

void UnaryCall(
    const std::string& op_name,
    Tensor& o,
    const Tensor& i,
    std::function<void(::musa::dnn::Unary&)> func) {
  c10::musa::MUSAGuard device_guard(i.device());
  muHandle& h = GetMudnnHandle();
  auto in = CreateMUTensor(i);
  auto out = CreateMUTensor(o);

  ::musa::dnn::Unary op;
  func(op);
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run " + op_name);
}

void UnaryBoolOut(
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
  c10::musa::MUSAGuard device_guard(input.device());
  // as le/lt/ne/eq/gt/ge... ops return bool type
  Tensor output = at::empty_like(
      input,
      input.options()
          .dtype(ScalarType::Bool)
          .memory_format(at::MemoryFormat::Contiguous));
  auto contiguous_input = input.contiguous();
  UnaryBoolOut(op_name, output, contiguous_input, value, mode);
  return output;
}

Tensor Unary(
    const std::string& op_name,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  c10::musa::MUSAGuard device_guard(input.device());
  Tensor output = at::empty_like(input);
  auto contiguous_input = input.contiguous();
  MUSA_TENSOR_TYPE_CHECK(input);
  UnaryCall(op_name, output, contiguous_input, func);
  return output;
}

void Unary_(
    const std::string& op_name,
    Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  UnaryCall(op_name, input, input, func);
}

void UnaryOut(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  output.resize_as_(input);
  UnaryCall(op_name, output, input, func);
}

#define DEFINE_ACTIVATE_OP(op_name, mode)                          \
  Tensor op_name(const Tensor& input) {                            \
    return Unary(__func__, input, [](::musa::dnn::Unary& op) {     \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
    });                                                            \
  }                                                                \
                                                                   \
  Tensor& op_name##_(Tensor& input) {                              \
    Unary_(__func__, input, [](::musa::dnn::Unary& op) {           \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
    });                                                            \
    return input;                                                  \
  }                                                                \
                                                                   \
  Tensor& op_name##Out(const Tensor& input, Tensor& output) {      \
    UnaryOut(__func__, output, input, [](::musa::dnn::Unary& op) { \
      CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");             \
    });                                                            \
    return output;                                                 \
  }

DEFINE_ACTIVATE_OP(Relu, ::musa::dnn::Unary::Mode::RELU)
DEFINE_ACTIVATE_OP(Silu, ::musa::dnn::Unary::Mode::SILU)
DEFINE_ACTIVATE_OP(Sqrt, ::musa::dnn::Unary::Mode::SQRT)
DEFINE_ACTIVATE_OP(Round, ::musa::dnn::Unary::Mode::ROUND)
DEFINE_ACTIVATE_OP(Rsqrt, ::musa::dnn::Unary::Mode::RSQRT)
DEFINE_ACTIVATE_OP(HardSwish, ::musa::dnn::Unary::Mode::HARDSWISH)
// TODO(chen.feng): use muDNN when the output issue is fixed
// DEFINE_ACTIVATE_OP(HardSigmoid, ::musa::dnn::Unary::Mode::HARDSIGMOID)
DEFINE_ACTIVATE_OP(Tanh, ::musa::dnn::Unary::Mode::TANH)
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

#define SCALAR_COMPARISON(op_name, mode)                          \
  Tensor& op_name##Out(                                           \
      const Tensor& self, const Scalar& value, Tensor& output) {  \
    auto contiguous_self = self.contiguous();                     \
    UnaryBoolOut(__func__, output, contiguous_self, value, mode); \
    return output;                                                \
  }                                                               \
                                                                  \
  Tensor op_name(const Tensor& self, const Scalar& value) {       \
    return UnaryBool(__func__, self, value, mode);                \
  }                                                               \
                                                                  \
  Tensor& op_name##_(Tensor& self, const Scalar& value) {         \
    auto out_tmp = UnaryBool(__func__, self, value, mode);        \
    self.copy_(out_tmp);                                          \
    return self;                                                  \
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
  UnaryOut(__func__, result, self, [&](::musa::dnn::Unary& op) {
    auto negcoef = alpha.to<float>() * scale.to<float>();
    op.SetAlpha(negcoef);
    op.SetMode(::musa::dnn::Unary::Mode::ELU);
  });
  return result;
}

Tensor GELU(const Tensor& self, c10::string_view approximate) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  TORCH_CHECK(
      approximate_type == at::native::GeluType::None,
      "Musa GELU op only support approximate is None now!");
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
}

Tensor& GELUOut(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& output) {
  auto approximate_type = at::native::get_gelutype_enum(approximate);
  TORCH_CHECK(
      approximate_type == at::native::GeluType::None,
      "Musa GELU op only support approximate is None now!");
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
  auto t_type = self.scalar_type();

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
  Tensor output = at::empty_like(self, at::MemoryFormat::Contiguous);
  MUSA_TENSOR_TYPE_CHECK(self);

  ClampCall(__func__, output, self, has_min, min, has_max, max);
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
  ClampCall(__func__, self, self, has_min, min, has_max, max);
  return self;
}

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
  out.resize_as_(self);
  ClampCall(__func__, out, self, has_min, min, has_max, max);
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
  c10::musa::MUSAGuard device_guard(self.device());
  auto has_min = min.has_value() && min->defined();
  auto has_max = max.has_value() && max->defined();
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
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
}

Tensor& Reciprocal_(Tensor& self) {
  Unary_(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return self;
}

Tensor& ReciprocalOut(const Tensor& self, Tensor& output) {
  UnaryOut(__func__, output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(-1.), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return output;
}

namespace {
// copied from RegisterCUDA.cpp
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

void NegCall(
    const std::string& op_name,
    Tensor& out,
    const Tensor& self,
    const c10::optional<Scalar>& val) {
  auto t_type = self.scalar_type();
  auto contiguous_self = self.contiguous();
  switch (t_type) {
    case ScalarType::Float: {
      const double alpha = val.value().to<double>();
      UnaryCall(op_name, out, contiguous_self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Half: {
      const double alpha = val.value().to<double>();
      UnaryCall(op_name, out, contiguous_self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Int: {
      const int64_t alpha = val.value().to<int64_t>();
      UnaryCall(op_name, out, contiguous_self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    case ScalarType::Long: {
      const int64_t alpha = val.value().to<int64_t>();
      UnaryCall(op_name, out, contiguous_self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::MUL), "SetMode");
      });
      break;
    }
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype in Neg: ", t_type);
      throw;
  }
}

Tensor Neg(const Tensor& self) {
  Tensor output = at::empty_like(self, at::MemoryFormat::Contiguous);
  MUSA_TENSOR_TYPE_CHECK(self);
  Scalar val = -1;
  NegCall(__func__, output, self, val);
  return output;
}

Tensor& Neg_(Tensor& self) {
  MUSA_TENSOR_TYPE_CHECK(self);
  Scalar val = -1;
  NegCall(__func__, self, self, val);
  return self;
}

Tensor& NegOut(const Tensor& self, Tensor& out) {
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
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
}

Tensor& PowScalar_(Tensor& self, const Scalar& value) {
  Unary_("pow_.Scalar", self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return self;
}

Tensor& PowScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryOut("pow.Tensor_Scalar_out", output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(value.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::POW), "SetMode");
  });
  return output;
}

Tensor LeakyRelu(const Tensor& input, const Scalar& neg_slope = 0.01) {
  return Unary(__func__, input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
}

Tensor& LeakyRelu_(Tensor& input, const Scalar& neg_slope = 0.01) {
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
  UnaryOut("leaky_relu.out", output, input, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetAlpha(neg_slope.to<double>()), "SetAlpha");
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Unary::Mode::LEAKY_RELU), "SetMode");
  });
  return output;
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

namespace {
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

Tensor& ClampMinOut(const Tensor& self, const Scalar& min, Tensor& out) {
  // No device check
  structured_clamp_min_out_out op(out);
  op.meta(self, min);
  op.impl(self, min, op.maybe_get_output(0));
  return out;
}

Tensor& BitwiseNotOut(const Tensor& self, Tensor& out) {
  // No device check

  structured_bitwise_not_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  return out;
}

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

at::Tensor BitwiseNot(const at::Tensor& self) {
  // No device check

  structured_bitwise_not_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

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

at::Tensor& BitwiseNot_(at::Tensor& self) {
  // No device check

  structured_bitwise_not_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  return self;
}

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

at::Tensor Sgn(const at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(common_device, self, "Sgn", "self");
  structured_sgn_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

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

at::Tensor& Sgn_(at::Tensor& self) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "Sgn_", "self");
  structured_sgn_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  return self;
}

// TODO(chen.feng): hardsigmoid porting
namespace {
struct structured_hardsigmoid_out_functional final
    : public at::native::structured_hardsigmoid_out {
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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }

  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }

  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_hardsigmoid_out_inplace final
    : public at::native::structured_hardsigmoid_out {
  structured_hardsigmoid_out_inplace(Tensor& self) : outputs_{std::ref(self)} {}

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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
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

struct structured_hardsigmoid_out_out final
    : public at::native::structured_hardsigmoid_out {
  structured_hardsigmoid_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}

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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
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
    at::native::structured_hardsigmoid_out::set_output_raw_strided(
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

at::Tensor HardSigmoid(const at::Tensor& self) {
  // No device check
  structured_hardsigmoid_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

at::Tensor& HardSigmoid_(at::Tensor& self) {
  // No device check
  structured_hardsigmoid_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}

at::Tensor& HardSigmoidOut(const at::Tensor& self, at::Tensor& out) {
  // No device check
  structured_hardsigmoid_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}
// hardsigmoid

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

at::Tensor& SignOut(const at::Tensor& self, at::Tensor& out) {
  // No device check
  structured_sign_out_out op(out);
  op.meta(self);
  op.impl(self, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

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

at::Tensor& Sign_(at::Tensor& self) {
  // No device check
  structured_sign_out_inplace op(self);
  op.meta(self);
  op.impl(self, op.outputs_[0]);
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return self;
}

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

at::Tensor Sign(const at::Tensor& self) {
  // No device check
  structured_sign_out_functional op;
  op.meta(self);
  op.impl(self, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("abs", &Abs);
  m.impl("abs_", &Abs_);
  m.impl("abs.out", &AbsOut);

  m.impl("sgn", &Sgn);
  m.impl("sgn_", &Sgn_);
  m.impl("sgn.out", &SgnOut);

  m.impl("sign", &Sign);
  m.impl("sign_", &Sign_);
  m.impl("sign.out", &SignOut);

  m.impl("bitwise_not", &BitwiseNot);
  m.impl("bitwise_not_", &BitwiseNot_);
  m.impl("bitwise_not.out", &BitwiseNotOut);

  m.impl("logical_not", &LogicalNot);
  m.impl("logical_not_", &LogicalNot_);
  m.impl("logical_not.out", &LogicalNotOut);

  m.impl("eq.Scalar", &EqScalar);
  m.impl("eq_.Scalar", &EqScalar_);
  m.impl("eq.Scalar_out", &EqScalarOut);

  m.impl("relu", &Relu);
  m.impl("relu_", &Relu_);

  m.impl("lt.Scalar", &LtScalar);
  m.impl("lt_.Scalar", &LtScalar_);
  m.impl("lt.Scalar_out", &LtScalarOut);

  m.impl("le.Scalar", &LeScalar);
  m.impl("le_.Scalar", &LeScalar_);
  m.impl("le.Scalar_out", &LeScalarOut);

  m.impl("ne.Scalar", &NeScalar);
  m.impl("ne_.Scalar", &NeScalar_);
  m.impl("ne.Scalar_out", &NeScalarOut);

  m.impl("gt.Scalar", &GtScalar);
  m.impl("gt_.Scalar", &GtScalar_);
  m.impl("gt.Scalar_out", &GtScalarOut);

  m.impl("ge.Scalar", &GeScalar);
  m.impl("ge_.Scalar", &GeScalar_);
  m.impl("ge.Scalar_out", &GeScalarOut);

  m.impl("sqrt", &Sqrt);
  m.impl("sqrt.out", &SqrtOut);

  m.impl("round", &Round);
  m.impl("round_", &Round_);
  m.impl("round.out", &RoundOut);

  m.impl("rsqrt", &Rsqrt);
  m.impl("rsqrt.out", &RsqrtOut);

  m.impl("hardswish", &HardSwish);
  m.impl("hardswish_", &HardSwish_);
  m.impl("hardswish.out", &HardSwishOut);

  m.impl("hardsigmoid", &HardSigmoid);
  m.impl("hardsigmoid_", &HardSigmoid_);
  m.impl("hardsigmoid.out", &HardSigmoidOut);

  m.impl("acos", &Acos);
  m.impl("acos_", &Acos_);
  m.impl("acos.out", &AcosOut);

  m.impl("tanh", &Tanh);
  m.impl("tanh_", &Tanh_);
  m.impl("tanh.out", &TanhOut);

  m.impl("atan", &Atan);
  m.impl("atan_", &Atan_);
  m.impl("atan.out", &AtanOut);

  m.impl("log", &Log);
  m.impl("log_", &Log_);
  m.impl("log.out", &LogOut);

  m.impl("log2", &Log2);
  m.impl("log2_", &Log2_);
  m.impl("log2.out", &Log2Out);

  m.impl("gelu", &GELU);
  m.impl("gelu.out", &GELUOut);

  m.impl("clamp", &Clamp);
  m.impl("clamp_", &Clamp_);
  m.impl("clamp.out", &ClampOut);
  m.impl("clamp.Tensor_out", &ClampTensorOut);
  m.impl("clamp_min.out", &ClampMinOut);

  m.impl("reciprocal", &Reciprocal);
  m.impl("reciprocal_", &Reciprocal_);
  m.impl("reciprocal.out", &ReciprocalOut);

  m.impl("sigmoid", &Sigmoid);
  m.impl("sigmoid_", &Sigmoid_);
  m.impl("sigmoid.out", &SigmoidOut);

  m.impl("ceil", &Ceil);
  m.impl("ceil_", &Ceil_);
  m.impl("ceil.out", &CeilOut);

  m.impl("exp", &Exp);
  m.impl("exp_", &Exp_);
  m.impl("exp.out", &ExpOut);

  m.impl("silu", &Silu);
  m.impl("silu_", &Silu_);
  m.impl("silu.out", &SiluOut);

  m.impl("cos", &Cos);
  m.impl("cos_", &Cos_);
  m.impl("cos.out", &CosOut);

  m.impl("sin", &Sin);
  m.impl("sin_", &Sin_);
  m.impl("sin.out", &SinOut);

  m.impl("neg", &Neg);
  m.impl("neg_", &Neg_);
  m.impl("neg.out", &NegOut);

  m.impl("pow.Tensor_Scalar", &PowScalar);
  m.impl("pow_.Scalar", &PowScalar_);
  m.impl("pow.Tensor_Scalar_out", &PowScalarOut);

  m.impl("leaky_relu", &LeakyRelu);
  m.impl("leaky_relu_", &LeakyRelu_);
  m.impl("leaky_relu.out", &LeakyReluOut);

  m.impl("log10", &Log10);
  m.impl("log10_", &Log10_);
  m.impl("log10.out", &Log10Out);

  m.impl("floor", &Floor);
  m.impl("floor_", &Floor_);
  m.impl("floor.out", &FloorOut);

  m.impl("elu", &Elu);
  m.impl("elu_", &Elu_);
  m.impl("elu.out", &EluOut);

  m.impl("hardtanh", &HardTanh);
  m.impl("hardtanh_", &HardTanh_);
  m.impl("hardtanh.out", &HardTanhOut);
  m.impl("hardtanh_backward", &HardTanhBackward);
  m.impl("hardtanh_backward.grad_input", &HardTanhBackwardOut);

  m.impl("_prelu_kernel", &PRelu);
  m.impl("_prelu_kernel_backward", &PReluBackward);

  m.impl("softplus", &SoftPlus);
  m.impl("softplus.out", &SoftPlusOut);

  m.impl("isnan", &IsNan);
}

} // namespace musa
} // namespace at
