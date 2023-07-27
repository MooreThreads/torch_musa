#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include <ATen/NamedTensorUtils.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

struct structured_cat_out_cuda_functional final
    : public at::native::structured_cat_out_cuda {
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
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_cat_out_cuda_out final
    : public at::native::structured_cat_out_cuda {
  structured_cat_out_cuda_out(Tensor& out0) : outputs_{std::ref(out0)} {}
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
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }
  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

Tensor Cat(const at::ITensorListRef& tensors, int64_t dim = 0) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, tensors, "wrapper_CUDA_cat", "tensors");
  structured_cat_out_cuda_functional op;
  auto precompute = op.meta(tensors, dim);
  (void)precompute;
  op.impl(
      tensors,
      precompute.dim,
      precompute.valid,
      precompute.all_contiguous,
      precompute.all_same_dtype,
      precompute.all_same_sizes_and_stride,
      precompute.memory_format,
      *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

Tensor& CatOut(const at::ITensorListRef& tensors, int64_t dim, Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_CUDA_cat_out_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, tensors, "wrapper_CUDA_cat_out_out", "tensors");
  structured_cat_out_cuda_out op(out);
  auto precompute = op.meta(tensors, dim);
  (void)precompute;
  op.impl(
      tensors,
      precompute.dim,
      precompute.valid,
      precompute.all_contiguous,
      precompute.all_same_dtype,
      precompute.all_same_sizes_and_stride,
      precompute.memory_format,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("cat", &Cat);
  m.impl("_cat", &Cat);
  m.impl("cat.out", &CatOut);
}

} // namespace musa
} // namespace at
