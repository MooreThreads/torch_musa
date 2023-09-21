#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/ReduceOpsUtils.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/musa/ReduceOps.h>

#include <ATen/Context.h>
#include <ATen/TensorIterator.h>
#include <ATen/TensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/NamedTensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/amax_native.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/aminmax_native.h>
#include <ATen/ops/full.h>
#include <ATen/ops/imag.h>
#include <ATen/ops/kthvalue_native.h>
#include <ATen/ops/median_native.h>
#include <ATen/ops/nanmedian_native.h>
#include <ATen/ops/where.h>
#endif

#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/musa/ReduceOps.h>
#include <ATen/ops/aminmax.h>
#include <ATen/ops/aminmax_native.h>
#include <torch/library.h>

#include <tuple>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace native {
namespace {

void NormKernel(TensorIterator& iter, const Scalar& val) {
  double p;
  if (val.isIntegral(false)) {
    p = val.to<int64_t>();
  } else if (val.isFloatingPoint()) {
    p = val.to<double>();
  } else {
    TORCH_CHECK(
        false, "norm_launch_kernel expects norm to be integer or float");
  }
  if (iter.numel() == 0) {
    iter.output().fill_((p < 0) ? INFINITY : 0);
    return;
  }
  norm_launch_kernel(iter, p);

  if (isComplexType(iter.output().scalar_type())) {
    at::imag(iter.output()).zero_();
  }
}

void AMinMaxKernel(
    const Tensor& self,
    int64_t dim,
    bool keepdim,
    Tensor& min_result,
    Tensor& max_result) {
  at::TensorIterator iter = make_reduction(
      "aminmax_musa",
      min_result,
      max_result,
      self,
      dim,
      keepdim,
      self.scalar_type());
  aminmax_launch_kernel(iter);
}

void AMinMaxAllReduceKernel(
    const Tensor& input,
    Tensor& min_result,
    Tensor& max_result) {
  auto dtype = input.scalar_type();
  auto iter = make_reduction(
      "aminmax_musa",
      min_result,
      max_result,
      input,
      IntArrayRef{},
      false,
      dtype);
  TORCH_CHECK(
      iter.numel() > 0, "min_max on a tensor with no elements is not defined.");
  aminmax_allreduce_launch_kernel(iter);
}

} // namespace

REGISTER_MUSA_DISPATCH(norm_stub, &NormKernel);
REGISTER_MUSA_DISPATCH(aminmax_stub, &AMinMaxKernel);
REGISTER_MUSA_DISPATCH(aminmax_allreduce_stub, &AMinMaxAllReduceKernel);

} // namespace native

namespace musa {
namespace {

struct structured_amax_out_out final : public at::native::structured_amax_out {
  structured_amax_out_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
    at::musa::resize_out(out, sizes, strides, options);
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
    at::musa::resize_out(out, sizes, strides, options);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_aminmax_out_out final
    : public at::native::structured_aminmax_out {
  structured_aminmax_out_out(Tensor& out0, Tensor& out1)
      : outputs_{std::ref(out0), std::ref(out1)} {}
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
          "structured kernels don't suport multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    const auto& out = outputs_[output_idx].get();
    at::musa::resize_out(out, sizes, strides, options);
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
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 2> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

} // namespace

at::Tensor& AMaxOut(
    const at::Tensor& self,
    at::IntArrayRef dim,
    bool keepdim,
    at::Tensor& out) {
  structured_amax_out_out op(out);
  op.meta(self, dim, keepdim);
  op.impl(self, dim, keepdim, op.maybe_get_output(0));
  return out;
}

std::tuple<at::Tensor&, at::Tensor&> AMinMaxOut(
    const at::Tensor& self,
    c10::optional<int64_t> dim,
    bool keepdim,
    at::Tensor& min,
    at::Tensor& max) {
  structured_aminmax_out_out op(min, max);
  op.meta(self, dim, keepdim);
  op.impl(self, dim, keepdim, op.maybe_get_output(0), op.maybe_get_output(1));
  return std::forward_as_tuple(min, max);
}

ADVANCED_REGISTER(aten, PrivateUse1, "amax.out", AMaxOut)
ADVANCED_REGISTER(aten, PrivateUse1, "aminmax.out", AMinMaxOut)

} // namespace musa
} // namespace at
