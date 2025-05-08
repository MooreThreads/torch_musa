#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/full_like.h>
#include <ATen/ops/result_type.h>
#include <ATen/ops/result_type_native.h>
#include <ATen/ops/where_native.h>
#endif

#include <ATen/TensorIterator.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>
#include "Binary.h"

namespace at {
namespace musa {
using TERNARY_MODE = ::musa::dnn::Ternary::Mode;

void CallCLIP(
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

void ClampScalarCall(
    const std::string& op_name,
    Tensor& out,
    const Tensor& self,
    bool has_min,
    const c10::optional<Scalar>& min,
    bool has_max,
    const c10::optional<Scalar>& max) {
  if (C10_UNLIKELY(self.numel() == 0)) {
    TORCH_CHECK(
        out.numel() == 0,
        "clamp output should be empty if input a empty tensor");
    return;
  }
  const auto t_type = self.scalar_type();
  switch (t_type) {
    case ScalarType::Float:
    case ScalarType::Half:
    case ScalarType::BFloat16: {
      // DBL_MIN = 2.22507e-308 which is positive, so we must use lowest or
      // (-max) there !!!
      const double min_val = has_min ? min.value().to<double>()
                                     : std::numeric_limits<double>::lowest();

      const double max_val = has_max ? max.value().to<double>()
                                     : std::numeric_limits<double>::max();
      CallCLIP(op_name, out, self, [&](::musa::dnn::Unary& op) {
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
      CallCLIP(op_name, out, self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
      break;
    }
    case ScalarType::Int: {
      // INT_MIN = - 2**32, INT_MAX = 2**32 - 1
      const int32_t min_val = has_min ? min.value().to<int32_t>()
                                      : std::numeric_limits<int32_t>::min();
      const int32_t max_val = has_max ? max.value().to<int32_t>()
                                      : std::numeric_limits<int32_t>::max();
      int64_t min_val_ = (int64_t)min_val;
      int64_t max_val_ = (int64_t)max_val;
      CallCLIP(op_name, out, self, [&](::musa::dnn::Unary& op) {
        CHECK_MUDNN_STATUS(op.SetAlpha(min_val_), "SetAlpha");
        CHECK_MUDNN_STATUS(op.SetBeta(max_val_), "SetBeta");
        CHECK_MUDNN_STATUS(
            op.SetMode(::musa::dnn::Unary::Mode::CLIP), "SetMode");
      });
      break;
    }
    default:
      TORCH_CHECK(false, "ClampScalar Unsupported tensor dtype: ", t_type);
      throw;
  }
}

void ClampTensorCall(
    Tensor& output,
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    TERNARY_MODE m = TERNARY_MODE::CLAMP) {
  if (C10_UNLIKELY(
          input1.numel() == 0 || input2.numel() == 0 || self.numel() == 0)) {
    return;
  }
  c10::musa::MUSAGuard device_guard(self.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Ternary top;

  auto input1_mt = CreateMUTensor(input1);
  auto input2_mt = CreateMUTensor(input2);
  auto self_mt = CreateMUTensor(self);
  // output should be contiguous, caller should be responsible for this
  auto om_mt = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(top.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(top.Run(h, om_mt, self_mt, input1_mt, input2_mt), "Run");
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
  Tensor output = at::empty_like(
      self,
      c10::TensorOptions(self.suggest_memory_format())
          .dtype(self.scalar_type()));

  MUSA_TENSOR_TYPE_CHECK(self);
  ClampScalarCall(__func__, output, self, has_min, min, has_max, max);

  return output;
}

Tensor& Clamp_(
    Tensor& self,
    const c10::optional<at::Scalar>& min,
    const c10::optional<at::Scalar>& max) {
  const bool has_min = (min.has_value());
  const bool has_max = (max.has_value());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: either min, max or both scalars must be defined");
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  ClampScalarCall(__func__, self, self, has_min, min, has_max, max);

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
  at::MemoryFormat output_memory_format = out.suggest_memory_format();
  Tensor input = self.suggest_memory_format() == output_memory_format
      ? self
      : FormatContiguous(self, output_memory_format);
  ClampScalarCall(__func__, out, input, has_min, min, has_max, max);

  return out;
}

Tensor ClampMin(const Tensor& self, const Scalar& min) {
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self,
      c10::TensorOptions(self.suggest_memory_format())
          .dtype(self.scalar_type()));
  ClampScalarCall(
      __func__,
      output,
      self,
      true,
      c10::optional<Scalar>(min),
      false,
      c10::optional<Scalar>());
  return output;
}

Tensor& ClampMinOut(const Tensor& self, const Scalar& min, Tensor& out) {
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());

  out.resize_as_(self);
  at::MemoryFormat output_memory_format = out.suggest_memory_format();
  Tensor input = self.suggest_memory_format() == output_memory_format
      ? self
      : FormatContiguous(self, output_memory_format);
  ClampScalarCall(
      __func__,
      out,
      input,
      true,
      c10::optional<Scalar>(min),
      false,
      c10::optional<Scalar>());
  return out;
}

Tensor ClampMax(const Tensor& self, const Scalar& max) {
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());
  Tensor output = at::empty_like(
      self,
      c10::TensorOptions(self.suggest_memory_format())
          .dtype(self.scalar_type()));
  ClampScalarCall(
      __func__,
      output,
      self,
      false,
      c10::optional<Scalar>(),
      true,
      c10::optional<Scalar>(max));
  return output;
}

Tensor& ClampMaxOut(const Tensor& self, const Scalar& max, Tensor& out) {
  MUSA_TENSOR_TYPE_CHECK(self);
  const c10::musa::MUSAGuard device_guard(self.device());

  out.resize_as_(self);
  at::MemoryFormat output_memory_format = out.suggest_memory_format();
  Tensor input = self.suggest_memory_format() == output_memory_format
      ? self
      : FormatContiguous(self, output_memory_format);
  ClampScalarCall(
      __func__,
      out,
      input,
      false,
      c10::optional<Scalar>(),
      true,
      c10::optional<Scalar>(max));
  return out;
}

Tensor& ClampTensorOut(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max,
    Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  MUSA_TENSOR_TYPE_CHECK(self);
  const auto self_device = self.device();
  const bool has_min = (min.has_value());
  const bool has_max = (max.has_value());
  TORCH_CHECK(
      has_min || has_max,
      "torch.clamp: either min, max or both tensors must be defined");
  if (has_min) {
    TORCH_CHECK(
        min->device() == self_device,
        "Device of min tensor of ClampTensor must be the same as self, "
        "but now is ",
        min->device());
    MUSA_TENSOR_TYPE_CHECK(min.value());
  }
  if (has_max) {
    TORCH_CHECK(
        max->device() == self_device,
        "Device of max tensor of ClampTensor must be the same as self, "
        "but now is ",
        max->device());
    MUSA_TENSOR_TYPE_CHECK(max.value());
  }
  // set output shape, must be consistent with self's sizes
  if (!output.sizes().equals(self.sizes())) {
    output.resize_(self.sizes());
  }
  if (!output.numel()) {
    return output;
  }
  if (!has_min) {
    // if the min is not provided, call binary's minimum op
    return MinimumTensorOut(self, max.value(), output);
  }
  if (!has_max) {
    // if the max is not provided, call binary's maximum op
    return MaximumTensorOut(self, min.value(), output);
  }
  if (min->numel() <= 1 && max->numel() <= 1) {
    // if min and max only have one element, convert it to the scalar to
    // process.
    return ClampOut(self, min.value().item(), max.value().item(), output);
  }
  // TODO(@kang.chen): mudnn now only support the inputs of same dtypes.
  Tensor min_ = min.value().to(self.scalar_type());
  Tensor max_ = max.value().to(self.scalar_type());
  ClampTensorCall(output, self, min_, max_);
  return output;
}

at::Tensor& ClampTensor_(
    Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  at::Tensor output = at::empty_like(
      self, self.options().memory_format(self.suggest_memory_format()));

  ClampTensorOut(self, min, max, output);

  self.copy_(output);
  return self;
}

Tensor ClampTensor(
    const Tensor& self,
    const c10::optional<Tensor>& min,
    const c10::optional<Tensor>& max) {
  at::Tensor output = at::empty_like(
      self, self.options().memory_format(self.suggest_memory_format()));

  ClampTensorOut(self, min, max, output);
  return output;
}

} // namespace musa
} // namespace at
