#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
namespace musa {
using SOFTMAX_MODE = ::musa::dnn::Softmax::Mode;

const char* H2FMsg = "softmax with half to float conversion is not supported!";

void SoftMaxCall(
    Tensor& output,
    const int64_t dim,
    const Tensor& input,
    SOFTMAX_MODE mode) {
  muHandle h;
  ::musa::dnn::Softmax softmax;
  auto input_m = CreateMUTensor(input);
  auto output_m = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(softmax.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(softmax.SetDim(static_cast<int>(dim)), "SetDim");
  CHECK_MUDNN_STATUS(
      softmax.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE),
      "SetAlgorithm");
  CHECK_MUDNN_STATUS(softmax.Run(h, output_m, input_m), "Run");
}

inline void CheckDimParams(const Tensor& input, const int64_t dim) {
  int64_t dim_ = maybe_wrap_dim(dim, input.dim());
  int64_t input_dim = input.dim() > 0 ? input.dim() : 1;
  TORCH_CHECK(
      dim_ >= 0 && dim_ < input_dim,
      "dim must be non-negative and less than input dimensions");
}

Tensor OpInternal(const Tensor& input, const int64_t dim, SOFTMAX_MODE mode) {
  auto contiguous_input = Contiguous(input);
  CheckDimParams(contiguous_input, dim);
  auto output = at::empty_like(contiguous_input);
  SoftMaxCall(output, dim, contiguous_input, mode);
  return output;
}

void OpInternalOut(
    Tensor& output,
    const Tensor& input,
    const int64_t dim,
    SOFTMAX_MODE mode) {
  auto contiguous_input = Contiguous(input);
  CheckDimParams(contiguous_input, dim);
  TORCH_CHECK(output.is_contiguous(), "check contiguous failed for unary op!");
  SoftMaxCall(output, dim, contiguous_input, mode);
}

Tensor LogSoftmax(const Tensor& self, int64_t dim, bool half_to_float) {
  TORCH_CHECK(!half_to_float, H2FMsg);
  return OpInternal(self, dim, SOFTMAX_MODE::LOGSOFTMAX);
}

Tensor& LogSoftmaxOut(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& output) {
  TORCH_CHECK(!half_to_float, H2FMsg);
  OpInternalOut(output, self, dim, SOFTMAX_MODE::LOGSOFTMAX);
  return output;
}

Tensor LogSoftmaxInt(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype = c10::nullopt) {
  bool half_to_float =
      self.scalar_type() == ScalarType::Half && dtype == ScalarType::Float;

  Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;
  Tensor result = LogSoftmax(converted, dim, half_to_float);

  namedinference::propagate_names(result, self);
  return result;
}

Tensor LogSoftmaxDimname(
    const Tensor& self,
    Dimname dim,
    c10::optional<at::ScalarType> dtype = c10::nullopt) {
  return LogSoftmaxInt(self, dimname_to_position(self, dim), dtype);
}

Tensor Softmax(const Tensor& self, int64_t dim, bool half_to_float) {
  TORCH_CHECK(!half_to_float, H2FMsg);
  return OpInternal(self, dim, SOFTMAX_MODE::SOFTMAX);
}

Tensor& SoftmaxOut(
    const Tensor& self,
    int64_t dim,
    bool half_to_float,
    Tensor& output) {
  TORCH_CHECK(!half_to_float, H2FMsg);
  OpInternalOut(output, self, dim, SOFTMAX_MODE::SOFTMAX);
  return output;
}

Tensor SoftmaxInt(
    const Tensor& self,
    int64_t dim,
    c10::optional<at::ScalarType> dtype = c10::nullopt) {
  bool half_to_float =
      self.scalar_type() == ScalarType::Half && dtype == ScalarType::Float;

  Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;
  Tensor result = Softmax(converted, dim, half_to_float);

  namedinference::propagate_names(result, self);
  return result;
}

Tensor SoftmaxDimname(
    const Tensor& self,
    Dimname dim,
    c10::optional<at::ScalarType> dtype = c10::nullopt) {
  return SoftmaxInt(self, dimname_to_position(self, dim), dtype);
}

Tensor& SoftmaxBwdInternal(
    const Tensor& grad_output,
    const Tensor& output,
    const int64_t dim,
    ScalarType input_dtype,
    const char* op_name,
    SOFTMAX_MODE mode,
    Tensor& grad_input) {
  TORCH_CHECK(
      input_dtype == ScalarType::Float, "input_dtype only support float32");
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of ",
      std::string(op_name),
      " must be MTGPU, but now is ",
      grad_output.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of ",
      std::string(op_name),
      " must be MTGPU, but now is ",
      output.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of ",
      std::string(op_name),
      " must be MTGPU, but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float,
      "Dtype of grad_output tensor of ",
      std::string(op_name),
      " only support Float32, but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float,
      "Dtype of output tensor of ",
      std::string(op_name),
      " only support Float32, but now it is ",
      output.scalar_type());

  grad_input.resize_(grad_output.sizes());

  auto contiguous_grad_output = grad_output.expect_contiguous();
  auto contiguous_output = output.expect_contiguous();

  const TensorArg grad_arg{grad_output, "grad", 0};
  const TensorArg output_arg{output, "output", 1};
  checkSameSize(op_name, grad_arg, output_arg);
  CheckDimParams(grad_output, dim);

  if (output.numel() == 0) {
    return grad_input;
  }
  muHandle h;
  ::musa::dnn::Softmax softmax;
  auto mt_grad_output = CreateMUTensor(*contiguous_grad_output);
  auto mt_output = CreateMUTensor(*contiguous_output);
  auto mt_grad_input = CreateMUTensor(grad_input);
  CHECK_MUDNN_STATUS(softmax.SetDim(static_cast<int>(dim)), "SetDim");
  CHECK_MUDNN_STATUS(
      softmax.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE),
      "SetAlgorithm");
  CHECK_MUDNN_STATUS(softmax.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(
      softmax.RunBwd(h, mt_grad_input, mt_output, mt_grad_output), "Run");
  return grad_input;
}

Tensor& SoftmaxOutBwd(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& grad_input) {
  return SoftmaxBwdInternal(
      grad_output,
      output,
      dim,
      input_dtype,
      "softmax_backward",
      SOFTMAX_MODE::SOFTMAX,
      grad_input);
}

Tensor SoftmaxBwd(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  auto grad_input = at::empty_like(grad_output);
  SoftmaxOutBwd(grad_output, output, dim, input_dtype, grad_input);
  return grad_input;
}

Tensor& LogSoftmaxDataOutBwd(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype,
    Tensor& out) {
  return SoftmaxBwdInternal(
      grad_output,
      output,
      dim,
      input_dtype,
      "logsoftmax_backward",
      SOFTMAX_MODE::LOGSOFTMAX,
      out);
}

Tensor LogSoftmaxDataBwd(
    const Tensor& grad_output,
    const Tensor& output,
    int64_t dim,
    ScalarType input_dtype) {
  Tensor result = at::empty(grad_output.sizes(), grad_output.options());
  LogSoftmaxDataOutBwd(grad_output, output, dim, input_dtype, result);
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("log_softmax.Dimname", &LogSoftmaxDimname);
  m.impl("_log_softmax", &LogSoftmax);
  m.impl("_log_softmax.out", &LogSoftmaxOut);
  m.impl("_log_softmax_backward_data", &LogSoftmaxDataBwd);
  m.impl("_log_softmax_backward_data.out", &LogSoftmaxDataOutBwd);

  m.impl("softmax.Dimname", &SoftmaxDimname);
  m.impl("_softmax", &Softmax);
  m.impl("_softmax.out", &SoftmaxOut);

  m.impl("_softmax_backward_data.out", &SoftmaxOutBwd);
  m.impl("_softmax_backward_data", &SoftmaxBwd);
}

} // namespace musa
} // namespace native
} // namespace at
