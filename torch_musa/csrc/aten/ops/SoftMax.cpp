#include <ATen/Config.h>
#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using SOFTMAX_MODE = ::musa::dnn::Softmax::Mode;

const char* H2FMsg = "softmax with half to float conversion is not supported!";

void SoftMaxCall(
    Tensor& output,
    const int64_t dim,
    const Tensor& input,
    SOFTMAX_MODE mode) {
  c10::musa::MUSAGuard device_guard(input.device());
  muHandle& h = GetMudnnHandle();
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
  auto contiguous_input = input.contiguous();
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
  auto contiguous_input = input.contiguous();
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
    c10::optional<at::ScalarType> dtype) {
  bool half_to_float = (self.scalar_type() == ScalarType::Half ||
                        self.scalar_type() == ScalarType::BFloat16) &&
      dtype == ScalarType::Float;

  Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;
  Tensor result = LogSoftmax(converted, dim, half_to_float);

  namedinference::propagate_names(result, self);
  return result;
}

Tensor LogSoftmaxDimname(
    const Tensor& self,
    Dimname dim,
    c10::optional<at::ScalarType> dtype) {
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
    c10::optional<at::ScalarType> dtype) {
  bool half_to_float = (self.scalar_type() == ScalarType::Half ||
                        self.scalar_type() == ScalarType::BFloat16) &&
      dtype == ScalarType::Float;

  Tensor converted = dtype.has_value() ? self.toType(dtype.value()) : self;
  Tensor result = Softmax(converted, dim, half_to_float);

  namedinference::propagate_names(result, self);
  return result;
}

Tensor SoftmaxDimname(
    const Tensor& self,
    Dimname dim,
    c10::optional<at::ScalarType> dtype) {
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
      input_dtype == ScalarType::Float || input_dtype == ScalarType::Half ||
          input_dtype == ScalarType::BFloat16,
      "input_dtype of SoftmaxBwd only support Float/Half/BFloat16");
  TORCH_CHECK(
      grad_output.device().type() == kMUSA,
      "Device of grad_output tensor of ",
      std::string(op_name),
      " must be MUSA, but now is ",
      grad_output.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of ",
      std::string(op_name),
      " must be MUSA, but now is ",
      output.device());
  TORCH_CHECK(
      grad_input.device().type() == kMUSA,
      "Device of grad_input tensor of ",
      std::string(op_name),
      " must be MUSA, but now is ",
      grad_input.device());
  TORCH_CHECK(
      grad_output.scalar_type() == at::ScalarType::Float ||
          grad_output.scalar_type() == at::ScalarType::Half ||
          grad_output.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of grad_output tensor of ",
      std::string(op_name),
      " only support Float/Half/BFloat16, but now it is ",
      grad_output.scalar_type());
  TORCH_CHECK(
      output.scalar_type() == at::ScalarType::Float ||
          output.scalar_type() == at::ScalarType::Half ||
          output.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of output tensor of ",
      std::string(op_name),
      " only support Float/Half/BFloat16, but now it is ",
      output.scalar_type());

  grad_input.resize_(grad_output.sizes());

  c10::musa::MUSAGuard device_guard(grad_output.device());
  auto contiguous_grad_output = grad_output.contiguous();
  auto contiguous_output = output.contiguous();

  const TensorArg grad_arg{grad_output, "grad", 0};
  const TensorArg output_arg{output, "output", 1};
  checkSameSize(op_name, grad_arg, output_arg);
  CheckDimParams(grad_output, dim);

  if (output.numel() == 0) {
    return grad_input;
  }
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Softmax softmax;
  auto mt_grad_output = CreateMUTensor(contiguous_grad_output);
  auto mt_output = CreateMUTensor(contiguous_output);
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
  auto grad_input = at::empty_like(grad_output, at::MemoryFormat::Contiguous);
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

Tensor& LogSumExpOutImpl(Tensor& result, const Tensor& self, IntArrayRef dims) {
  auto run_logsumexp = [](Tensor& result, const Tensor& self, int dim) {
    c10::musa::MUSAGuard device_guard(self.device());

    Tensor self_contig = FormatContiguous(self, at::MemoryFormat::Contiguous);
    Tensor result_contig =
        FormatContiguous(result, at::MemoryFormat::Contiguous);
    muHandle& h = GetMudnnHandle();
    ::musa::dnn::Softmax desc;
    auto input_m = CreateMUTensor(self_contig);
    auto output_m = CreateMUTensor(result_contig);
    CHECK_MUDNN_STATUS(
        desc.SetMode(::musa::dnn::Softmax::Mode::LOGSUMEXP), "SetMode");
    CHECK_MUDNN_STATUS(desc.SetDim(dim), "SetDim");
    CHECK_MUDNN_STATUS(
        desc.SetAlgorithm(::musa::dnn::Softmax::Algorithm::ACCURATE),
        "SetAlgorithm");
    CHECK_MUDNN_STATUS(desc.Run(h, output_m, input_m), "Run");
    if (!result.is_contiguous()) {
      result.copy_(result_contig);
    }
  };

  if (dims.size() > 1) {
    Tensor cur_self = self;
    Tensor cur_result;
    int dims_size = dims.size();
    for (int i = 0; i < dims_size; ++i) {
      DimVector cur_dim({dims[i]});
      maybe_wrap_dims(cur_dim, self.dim());
      if (i == (dims_size - 1)) {
        cur_result = result;
      } else {
        auto cur_shape =
            at::meta::get_reduction_shape(cur_self, cur_dim, /*keepdim=*/true);
        cur_result = at::empty(
            cur_shape,
            result.options().memory_format(at::MemoryFormat::Contiguous));
      }
      run_logsumexp(cur_result, cur_self, static_cast<int>(cur_dim[0]));
      cur_self = cur_result;
    }
  } else {
    run_logsumexp(result, self, static_cast<int>(dims[0]));
  }

  return result;
}

Tensor& LogSumExpOut(
    const Tensor& self,
    IntArrayRef dims,
    bool keepdim,
    Tensor& result) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_MUSA_CHECK_FLOATING_TYPES(result.scalar_type(), "LogSumExpOut");
  namedinference::propagate_names_for_reduction(result, self, dims, keepdim);

  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // for integral inputs, promote input to default floating type.
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    LogSumExpOutImpl(result, self.to(default_dtype), dims);
  } else {
    LogSumExpOutImpl(result, self, dims);
  }

  return result;
}

Tensor LogSumExp(const Tensor& self, IntArrayRef dims, bool keepdim) {
  TensorOptions result_options;
  DimVector dims_vec(dims);
  maybe_wrap_dims(dims_vec, self.dim());
  auto shape = at::meta::get_reduction_shape(self, dims_vec, keepdim);

  if (at::isIntegralType(self.scalar_type(), /*includeBool=*/true)) {
    // even for integral inputs, result is floating dtype
    auto default_dtype = at::typeMetaToScalarType(c10::get_default_dtype());
    result_options = self.options().dtype(default_dtype);
  } else {
    result_options = self.options();
  }
  auto result = at::empty(
      shape, result_options.memory_format(at::MemoryFormat::Contiguous));
  return LogSumExpOut(self, dims_vec, keepdim, result);
}

} // namespace musa
} // namespace at
