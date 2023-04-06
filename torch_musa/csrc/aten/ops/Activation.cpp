#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>

// Restore disabled warnings
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {
namespace musa {

using UNARY_MODE = ::musa::dnn::Unary::Mode;

void UnaryCall(
    const std::string& op_name,
    Tensor& o,
    const Tensor& i,
    std::function<void(::musa::dnn::Unary&)> func) {
  muHandle h;
  auto in = CreateMUTensor(i);
  auto out = CreateMUTensor(o);

  ::musa::dnn::Unary op;
  func(op);
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run " + op_name);
}

// TODO (zaixing.wang): we should modify this function if muDNN supports other
// dytpe
void UnaryBoolOut(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    const Scalar& value,
    UNARY_MODE mode) {
  Tensor contiguous_input = Contiguous(input);
  UnaryCall(op_name, output, contiguous_input, [&](::musa::dnn::Unary& op) {
    if (contiguous_input.scalar_type() == ScalarType::Long ||
        contiguous_input.scalar_type() == ScalarType::Byte) {
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
  // as le/lt/ne/eq/gt/ge... ops return bool type
  Tensor output = empty_mtgpu(
      input.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  UnaryBoolOut(op_name, output, input, value, mode);
  return output;
}

Tensor Unary(
    const std::string& op_name,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  Tensor contiguous_input = Contiguous(input);
  Tensor output = at::empty_like(contiguous_input);
  MUSA_TENSOR_TYPE_CHECK(contiguous_input);
  UnaryCall(op_name, output, contiguous_input, func);
  return output;
}

void Unary_(
    const std::string& op_name,
    Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  CheckContiguousWithName(op_name, input);
  MUSA_TENSOR_TYPE_CHECK(input);
  UnaryCall(op_name, input, input, func);
}

void UnaryOut(
    const std::string& op_name,
    Tensor& output,
    const Tensor& input,
    std::function<void(::musa::dnn::Unary&)> func) {
  CheckContiguousWithName(op_name, output);
  Tensor contiguous_input = Contiguous(input);
  MUSA_TENSOR_TYPE_CHECK(contiguous_input);
  UnaryCall(op_name, output, contiguous_input, func);
}

Tensor& BinaryCall(
    const Tensor& input,
    const Tensor& other,
    ::musa::dnn::Binary::Mode mode,
    Tensor& output) {
  ::musa::dnn::Handle h;
  ::musa::dnn::Binary binary_op;
  auto mt_input = CreateMUTensor(input);
  auto mt_other = CreateMUTensor(other);
  auto mt_output = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(binary_op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(binary_op.Run(h, mt_output, mt_input, mt_other), "Run");
  return output;
}

Tensor Relu(const Tensor& input) {
  return Unary(__func__, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::RELU), "SetMode");
  });
}

Tensor& Relu_(Tensor& input) {
  Unary_(__func__, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::RELU), "SetMode");
  });
  return input;
}

#define SCALAR_COMPARISON(op_name, mode)                         \
  Tensor& op_name##Out(                                          \
      const Tensor& self, const Scalar& value, Tensor& output) { \
    UnaryBoolOut(__func__, output, self, value, mode);           \
    return output;                                               \
  }                                                              \
                                                                 \
  Tensor op_name(const Tensor& self, const Scalar& value) {      \
    Tensor output = empty_mtgpu(                                 \
        self.sizes().vec(),                                      \
        ScalarType::Bool,                                        \
        c10::nullopt,                                            \
        kMUSA,                                                   \
        c10::nullopt,                                            \
        at::MemoryFormat::Contiguous);                           \
    op_name##Out(self, value, output);                           \
    return output;                                               \
  }                                                              \
                                                                 \
  Tensor& op_name##_(Tensor& self, const Scalar& value) {        \
    self = UnaryBool(__func__, self, value, mode);               \
    return self;                                                 \
  }

SCALAR_COMPARISON(LeScalar, UNARY_MODE::LE)
SCALAR_COMPARISON(LtScalar, UNARY_MODE::LT)
SCALAR_COMPARISON(GeScalar, UNARY_MODE::GE)
SCALAR_COMPARISON(GtScalar, UNARY_MODE::GT)
SCALAR_COMPARISON(EqScalar, UNARY_MODE::EQ)
SCALAR_COMPARISON(NeScalar, UNARY_MODE::NE)

Tensor& ThresholdGradInputBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of ThresholdBackward must be MUSA, ",
      "but now is ",
      grad_output.device());
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of self tensor of ThresholdBackward must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of ThresholdBackward must be MUSA, ",
      "but now is ",
      grad_input.device());
  grad_input.resize_(self.sizes());

  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of ThresholdBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of self tensor of ThresholdBackward only support Float32, ",
      "but now it is ",
      self.scalar_type());
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_self = Contiguous(self);

  muHandle h;
  ::musa::dnn::Binary binary_op;
  auto mt_grad_output = CreateMUTensor(contiguous_grad_output);
  auto mt_self = CreateMUTensor(contiguous_self);
  auto mt_output = CreateMUTensor(grad_input);
  CHECK_MUDNN_STATUS(
      binary_op.SetMode(::musa::dnn::Binary::Mode::THRESHOLD_BW), "SetMode");
  // only support float32 now
  AT_DISPATCH_ALL_MTGPU_TYPES_AND_HALF(
      self.scalar_type(), "threshold_backward", [&] {
        auto threshold_value = threshold.to<scalar_t>();
        if (self.scalar_type() == at::ScalarType::Double ||
            self.scalar_type() == at::ScalarType::Float ||
            self.scalar_type() == at::ScalarType::Half) {
          CHECK_MUDNN_STATUS(
              binary_op.SetAlpha(static_cast<double>(threshold_value)),
              "SetAlpha");
        } else {
          CHECK_MUDNN_STATUS(
              binary_op.SetAlpha(static_cast<int64_t>(threshold_value)),
              "SetAlpha");
        }
        CHECK_MUDNN_STATUS(
            binary_op.Run(h, mt_output, mt_grad_output, mt_self), "Run");
      });
  return grad_input;
}

Tensor ThresholdBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold) {
  auto grad_input = at::empty(self.sizes(), self.options());
  ThresholdGradInputBwd(grad_output, self, threshold, grad_input);
  return grad_input;
}

Tensor Sqrt(const Tensor& input) {
  return Unary(__func__, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::SQRT), "SetMode");
  });
}

Tensor& SqrtOut(const Tensor& input, Tensor& output) {
  UnaryOut(__func__, output, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::SQRT), "SetMode");
  });
  return output;
}

Tensor Tanh(const Tensor& input) {
  return Unary(__func__, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::TANH), "SetMode");
  });
}

Tensor& Tanh_(Tensor& input) {
  Tensor contiguous_input = Contiguous(input);
  Unary_(__func__, contiguous_input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::TANH), "SetMode");
  });
  input.copy_(contiguous_input);
  return input;
}

Tensor& TanhOut(const Tensor& input, Tensor& output) {
  UnaryOut(__func__, output, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::TANH), "SetMode");
  });
  return output;
}

at::Tensor& TanhGradInputBwd(
    const at::Tensor& grad_output,
    const at::Tensor& output,
    at::Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of TanhBackward must be MUSA, but now is ",
      grad_output.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of TanhBackward must be MUSA, but now is ",
      output.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of TanhBackward must be MUSA, but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of TanhBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float,
      "Dtype of output tensor of TanhBackward only support Float32, ",
      "but now it is ",
      output.scalar_type());
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_output = Contiguous(output);

  grad_input.resize_((contiguous_grad_output).sizes());
  return BinaryCall(
      contiguous_grad_output,
      contiguous_output,
      ::musa::dnn::Binary::Mode::TANH_BW,
      grad_input);
}

at::Tensor TanhBwd(const at::Tensor& grad_output, const at::Tensor& output) {
  auto result = at::empty_like(grad_output);
  Tensor grad_output_ = Contiguous(grad_output);
  Tensor output_ = Contiguous(output);
  Tensor result_ = Contiguous(result);
  TanhGradInputBwd(grad_output_, output_, result_);
  return result_;
}

Tensor GELU(const Tensor& self, c10::string_view approximate) {
  auto approximate_type = get_gelutype_enum(approximate);
  TORCH_CHECK(
      approximate_type == GeluType::None,
      "Musa GELU op only support approximate is None now!");
  return Unary(__func__, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
}

Tensor& GELUOut(
    const Tensor& self,
    c10::string_view approximate,
    Tensor& output) {
  auto approximate_type = get_gelutype_enum(approximate);
  TORCH_CHECK(
      approximate_type == GeluType::None,
      "Musa GELU op only support approximate is None now!");
  UnaryOut(__func__, output, self, [&](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::GELU), "SetMode");
  });
  return output;
}

at::Tensor& GELUGradInputBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate,
    at::Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of GELUBackward must be MUSA, but now is ",
      grad_output.device());
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of self tensor of GELUBackward must be MUSA, but now is ",
      self.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of GELUBackward must be MUSA, but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of GELUBackward only support Float32, ",
      "but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of GELUBackward only support Float32, ",
      "but now it is ",
      self.scalar_type());
  auto contiguous_grad_output = Contiguous(grad_output);
  auto contiguous_self = Contiguous(self);

  grad_input.resize_(self.sizes());
  auto approximate_type = get_gelutype_enum(approximate);
  if (approximate_type == GeluType::None) {
    return BinaryCall(
        contiguous_grad_output,
        contiguous_self,
        ::musa::dnn::Binary::Mode::GELU_NONE_BW,
        grad_input);
  } else {
    return BinaryCall(
        contiguous_grad_output,
        contiguous_self,
        ::musa::dnn::Binary::Mode::GELU_TANH_BW,
        grad_input);
  }
}

at::Tensor GELUBwd(
    const at::Tensor& grad_output,
    const at::Tensor& self,
    c10::string_view approximate) {
  auto result = ::at::empty(self.sizes(), self.options());
  GELUGradInputBwd(grad_output, self, approximate, result);
  return result;
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
  auto self_ = Contiguous(self);

  switch (t_type) {
    case ScalarType::Float: {
      // DBL_MIN = 2.22507e-308 which is positive, so we must use lowest or
      // (-max) there !!!
      const double min_val = has_min ? min.value().to<double>()
                                     : std::numeric_limits<double>::lowest();

      const double max_val = has_max ? max.value().to<double>()
                                     : std::numeric_limits<double>::max();
      UnaryCall(op_name, out, self_, [&](::musa::dnn::Unary& op) {
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
      UnaryCall(op_name, out, self_, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
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
  Tensor output = at::empty_like(self);
  MUSA_TENSOR_TYPE_CHECK(self);

  ClampCall(__func__, output, self, has_min, min, has_max, max);
  return output;
}

Tensor& Clamp_(
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
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

  m.impl("threshold_backward.grad_input", &ThresholdGradInputBwd);
  m.impl("threshold_backward", &ThresholdBwd);

  m.impl("sqrt", &Sqrt);
  m.impl("sqrt.out", &SqrtOut);

  m.impl("tanh", &Tanh);
  m.impl("tanh_", &Tanh_);
  m.impl("tanh.out", &TanhOut);
  m.impl("tanh_backward.grad_input", &TanhGradInputBwd);
  m.impl("tanh_backward", &TanhBwd);

  m.impl("gelu", &GELU);
  m.impl("gelu.out", &GELUOut);
  m.impl("gelu_backward", &GELUBwd);
  m.impl("gelu_backward.grad_input", &GELUGradInputBwd);

  m.impl("clamp", &Clamp);
  m.impl("clamp_", &Clamp_);
  m.impl("clamp.out", &ClampOut);
  m.impl("clamp.Tensor_out", &ClampTensorOut);

  m.impl("reciprocal", &Reciprocal);
  m.impl("reciprocal_", &Reciprocal_);
  m.impl("reciprocal.out", &ReciprocalOut);
}

} // namespace musa
} // namespace native
} // namespace at
