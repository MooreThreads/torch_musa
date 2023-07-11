#include <ATen/ExpandUtils.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {
using TERNARY_MODE = ::musa::dnn::Ternary::Mode;

Tensor& WhereSelfOut_Float16(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::where_self_out(condition, self, other, out);
}

Tensor WhereSelf_Float16(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  c10::musa::MUSAGuard device_guard(self.device());
  return at::native::where(condition, self, other);
}

namespace {
struct structured_addcdiv_out_out final
    : public at::native::structured_addcdiv_out {
  structured_addcdiv_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
    at::native::structured_addcdiv_out::set_output_raw_strided(
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
    at::native::structured_addcdiv_out::set_output_raw_strided(
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

struct structured_addcmul_out_out final
    : public at::native::structured_addcmul_out {
  structured_addcmul_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
    at::native::structured_addcmul_out::set_output_raw_strided(
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
    at::native::structured_addcmul_out::set_output_raw_strided(
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

Tensor& AddcDivOut_Float16(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  structured_addcdiv_out_out op(out);
  op.meta(self, tensor1, tensor2, value);
  op.impl(self, tensor1, tensor2, value, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}
Tensor& AddcMulOut_Float16(
    const Tensor& self,
    const Tensor& tensor1,
    const Tensor& tensor2,
    const Scalar& value,
    Tensor& out) {
  structured_addcmul_out_out op(out);
  op.meta(self, tensor1, tensor2, value);
  op.impl(self, tensor1, tensor2, value, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

void TernaryCall(
    Tensor& output,
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    TERNARY_MODE m,
    const Scalar& alpha_scalar) {
  c10::musa::MUSAGuard device_guard(self.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Ternary top;

  if (!alpha_scalar.equal(1)) {
    if (m == TERNARY_MODE::ADDCMUL) {
      m = TERNARY_MODE::ADDCMUL_ALPHA;
    } else if (m == TERNARY_MODE::ADDCDIV) {
      m = TERNARY_MODE::ADDCDIV_ALPHA;
    }
    if (self.is_floating_point()) {
      CHECK_MUDNN_STATUS(top.SetAlpha(alpha_scalar.toDouble()), "SetAlpha");
    } else {
      CHECK_MUDNN_STATUS(top.SetAlpha(alpha_scalar.toLong()), "SetAlpha");
    }
  }

  auto contiguous_input1 = at::musa::Contiguous(input1);
  auto contiguous_input2 = at::musa::Contiguous(input2);
  auto contiguous_self = at::musa::Contiguous(self);
  auto input1_mt = CreateMUTensor(contiguous_input1);
  auto input2_mt = CreateMUTensor(contiguous_input2);
  auto self_mt = CreateMUTensor(contiguous_self);
  // output should be contiguous, caller should be responsible for this
  auto om_mt = CreateMUTensor(output);
  CHECK_MUDNN_STATUS(top.SetMode(m), "SetMode");
  CHECK_MUDNN_STATUS(top.Run(h, om_mt, self_mt, input1_mt, input2_mt), "Run");
}

/**
 * @brief Ternary operations convention. All of the function params that are
 * tensors could be non-contiguous, be careful.
 *
 * @param self input tensor
 * @param input1 first of the two operands
 * @param input2 second of the two operands
 * @param alpha_scalar scaling factor
 * @param output output tensor to write result
 * @param m Ternary Mode Type.
 */
void TernarycommonDtypeCall(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    Scalar const& alpha_scalar,
    Tensor& output,
    TERNARY_MODE m) {
  auto common_dtype = at::result_type(input1, input2);
  at::native::alpha_check(common_dtype, alpha_scalar);
  // WARN: output created by torch, which could be non-contiguous.
  output = at::musa::Contiguous(output);
  Tensor contiguous_self = Contiguous(self.to(common_dtype));
  Tensor contiguous_input1 = Contiguous(input1.to(common_dtype));
  Tensor contiguous_input2 = Contiguous(input2.to(common_dtype));
  TernaryCall(
      output,
      contiguous_self,
      contiguous_input1,
      contiguous_input2,
      m,
      alpha_scalar);
}

Tensor& TernaryOut(
    Tensor& output,
    const Tensor& cond,
    const Tensor& self,
    const Tensor& other,
    TERNARY_MODE m,
    const Scalar& alpha_scalar) {
  TORCH_CHECK(
      self.scalar_type() == other.scalar_type(),
      "input scalar type must the same");

  Tensor contiguous_out = Contiguous(output);
  Tensor contiguous_cond = Contiguous(cond);
  Tensor contiguous_self = Contiguous(self);
  Tensor contiguous_other = Contiguous(other);

  // 1. deal with other and self tensor shape isn't same
  if (other.dim() == 0) {
    contiguous_other = at::full_like(contiguous_self, other.item());
  }
  if (self.dim() == 0) {
    contiguous_self = at::full_like(contiguous_other, self.item());
  }
  TernaryCall(
      contiguous_out,
      contiguous_cond,
      contiguous_self,
      contiguous_other,
      m,
      alpha_scalar);
  return output;
}

Tensor& WhereSelfOut(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == at::ScalarType::Half ||
      other.scalar_type() == at::ScalarType::Half ||
      out.scalar_type() == at::ScalarType::Half) {
    return WhereSelfOut_Float16(condition, self, other, out);
  }
  Tensor contiguous_self, contiguous_other;
  auto result_type = at::native::result_type(self, other);
  if (self.dtype() != other.dtype()) {
    contiguous_self = self.to(result_type);
    contiguous_other = other.to(result_type);
  } else {
    contiguous_self = self;
    contiguous_other = other;
  }
  if (condition.scalar_type() == ScalarType::Byte) {
    TORCH_WARN_ONCE(
        "where received a uint8 condition tensor. This behavior is deprecated "
        "and will be removed in a future version of PyTorch. Use a boolean "
        "condition instead.");
  } else {
    TORCH_CHECK(
        condition.scalar_type() == ScalarType::Bool,
        "where expected condition to be a boolean tensor, but got a "
        "tensor with dtype ",
        condition.scalar_type());
  }
  Tensor cond_bool = condition.scalar_type() == ScalarType::Byte
      ? condition.to(ScalarType::Bool)
      : condition;
  // compute output shape
  DimVector output_shape;
  std::vector<std::vector<int64_t>> operands_shape = {
      condition.sizes().vec(), self.sizes().vec(), other.sizes().vec()};
  for (const auto& shape : operands_shape) {
    if (output_shape.empty()) {
      output_shape = DimVector(shape.begin(), shape.end());
    }
    if (output_shape != DimVector(shape.begin(), shape.end())) {
      output_shape = infer_size_dimvector(output_shape, shape);
    }
  }

  if (!out.sizes().equals(output_shape)) {
    out.resize_(output_shape);
  }
  if (!out.numel()) {
    return out;
  }
  return TernaryOut(out, cond_bool, self, other, TERNARY_MODE::SELECT, 1);
}

Tensor WhereSelf(
    const Tensor& condition,
    const Tensor& self,
    const Tensor& other) {
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == at::ScalarType::Half ||
      other.scalar_type() == at::ScalarType::Half) {
    return WhereSelf_Float16(condition, self, other);
  }
  auto result_type = at::native::result_type(self, other);
  Tensor output = at::empty(
      other.sizes(),
      self.options()
          .dtype(result_type)
          .memory_format(at::MemoryFormat::Contiguous));
  WhereSelfOut(condition, self, other, output);
  return output;
}

Tensor& AddcMulOut(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar& alpha_scalar,
    Tensor& output) {
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == at::ScalarType::Half &&
      input1.scalar_type() == at::ScalarType::Half &&
      input2.scalar_type() == at::ScalarType::Half) {
    return AddcMulOut_Float16(self, input1, input2, alpha_scalar, output);
  }
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of addcmul must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input1.device().type() == kMUSA,
      "Device of input1 tensor of addcmul must be MUSA, but now it is ",
      input1.device());
  TORCH_CHECK(
      input2.device().type() == kMUSA,
      "Device of input2 tensor of addcmul must be MUSA, but now it is ",
      input2.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of addcmul must be MUSA, but now it is ",
      output.device());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCMUL);
  return output;
}

Tensor& AddcDivOut(
    const Tensor& self,
    const Tensor& input1,
    const Tensor& input2,
    const Scalar& alpha_scalar,
    Tensor& output) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of addcdiv must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input1.device().type() == kMUSA,
      "Device of input1 tensor of addcdiv must be MUSA, but now it is ",
      self.device());
  TORCH_CHECK(
      input2.device().type() == kMUSA,
      "Device of input2 tensor of addcdiv must be MUSA, but now it is ",
      input2.device());
  TORCH_CHECK(
      output.device().type() == kMUSA,
      "Device of output tensor of addcdiv must be MUSA, but now it is ",
      output.device());
  c10::musa::MUSAGuard device_guard(self.device());
  if (self.scalar_type() == at::ScalarType::Half &&
      input1.scalar_type() == at::ScalarType::Half &&
      input2.scalar_type() == at::ScalarType::Half) {
    return AddcDivOut_Float16(self, input1, input2, alpha_scalar, output);
  }
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of addcdiv only support Float32, but now it is ",
      self.scalar_type());
  TORCH_CHECK(
      input1.scalar_type() == at::ScalarType::Float,
      "Dtype of input1 tensor of addcdiv only support Float32, but now it is ",
      input1.scalar_type());
  TORCH_CHECK(
      input2.scalar_type() == at::ScalarType::Float,
      "Dtype of input2 tensor of addcdiv only support Float32, but now it is ",
      input2.scalar_type());
  TORCH_CHECK(
      output.dtype() == at::ScalarType::Float,
      "Dtype of output tensor of addcdiv only support Float32, but now it is ",
      output.dtype());
  TernarycommonDtypeCall(
      self, input1, input2, alpha_scalar, output, TERNARY_MODE::ADDCDIV);
  return output;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("where.self", &WhereSelf);
  m.impl("where.self_out", &WhereSelfOut);

  m.impl("addcdiv.out", &AddcDivOut);
  m.impl("addcmul.out", &AddcMulOut);
}

} // namespace musa
} // namespace at
