#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Activation.h>
#include <torch/library.h>
#include <torch/torch.h>
#include <limits>
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

// TODO(songtao.liu): muDNN requres bool tensor for lt/le... output,
// which cannot be autocast in muDNN now.
Tensor UnaryBool(const Tensor& input, const Scalar& value, UNARY_MODE mode) {
  // as le/lt/ne/eq/gt/ge... ops return bool type
  Tensor output = empty_mtgpu(
      input.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  UnaryBoolOut(__func__, output, input, value, mode);
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

Tensor Relu(const Tensor& input) {
  return Unary(__func__, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::RELU), "SetMode");
  });
}

Tensor& Relu_(Tensor& input) {
  Unary_("relu_", input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::RELU), "SetMode");
  });
  return input;
}

Tensor& LeScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::LE);
  return output;
}

Tensor LeScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  LeScalarOut(self, value, output);
  return output;
}

Tensor& LeScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::LE);
  return self;
}

Tensor& LtScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::LT);
  return output;
}

Tensor LtScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  LtScalarOut(self, value, output);
  return output;
}

Tensor& LtScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::LT);
  return self;
}

Tensor& GeScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::GE);
  return output;
}

Tensor GeScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  GeScalarOut(self, value, output);
  return output;
}

Tensor& GeScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::GE);
  return self;
}

Tensor& GtScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::GT);
  return output;
}

Tensor GtScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  GtScalarOut(self, value, output);
  return output;
}

Tensor& GtScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::GT);
  return self;
}

Tensor& NeScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::NE);
  return output;
}

Tensor NeScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  NeScalarOut(self, value, output);
  return output;
}

Tensor& NeScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::NE);
  return self;
}

Tensor& EqScalarOut(const Tensor& self, const Scalar& value, Tensor& output) {
  UnaryBoolOut(__func__, output, self, value, UNARY_MODE::EQ);
  return output;
}

Tensor EqScalar(const Tensor& self, const Scalar& value) {
  Tensor output = empty_mtgpu(
      self.sizes().vec(),
      ScalarType::Bool,
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
  EqScalarOut(self, value, output);
  return output;
}

Tensor& EqScalar_(Tensor& self, const Scalar& value) {
  self = UnaryBool(self, value, UNARY_MODE::EQ);
  return self;
}

Tensor& ThresholdGradInputBwd(
    const Tensor& grad_output,
    const Tensor& self,
    const Scalar& threshold,
    Tensor& grad_input) {
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of ThresholdBackward must be MTGPU, ",
      "but now is ",
      grad_output.device());
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of self tensor of ThresholdBackward must be MTGPU, but now is ",
      self.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of ThresholdBackward must be MTGPU, ",
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
  auto contiguous_grad_output = grad_output.expect_contiguous();
  auto contiguous_self = self.expect_contiguous();

  muHandle h;
  ::musa::dnn::Binary binary_op;
  auto mt_grad_output = CreateMUTensor(*contiguous_grad_output);
  auto mt_self = CreateMUTensor(*contiguous_self);
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
  UnaryOut("sqrt.out", output, input, [](::musa::dnn::Unary& op) {
    CHECK_MUDNN_STATUS(op.SetMode(::musa::dnn::Unary::Mode::SQRT), "SetMode");
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
}
} // namespace musa
} // namespace native
} // namespace at
