#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UpSample.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor& UpSampleNearest2dOut(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  MUSA_TENSOR_TYPE_CHECK(self);
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.dim() == 4,
      "UpSampleNearest2dOut needs input to be a 4-D tensor, which is ",
      self.dim());
  TORCH_CHECK(
      result.dim() == 4,
      "UpSampleNearest2dOut needs output to be a 4-D tensor, which is ",
      result.dim());
  TORCH_CHECK(
      self.scalar_type() == c10::ScalarType::Float ||
          self.scalar_type() == c10::ScalarType::Half,
      "UpSampleNearest2dOut needs input to be a float or half dtype tensor, which is",
      self.scalar_type());

  if (self.numel() == 0) {
    return result;
  }

  if (self.sizes() == result.sizes()) {
    result.copy_(self);
    return result;
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = self.size(2);
  int input_width = self.size(3);

  const float height_scale = at::native::compute_scales_value<float>(
      scales_h, input_height, output_height);
  const float width_scale = at::native::compute_scales_value<float>(
      scales_w, input_width, output_width);

  Tensor in_;
  Tensor out_;
  at::musa::muTensor in;
  at::musa::muTensor out;
  bool result_out = false;

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
  CHECK_MUDNN_STATUS(
      op.SetScaleInfo({height_scale, width_scale}), "SetScaleInfo");

  const auto input_memory_format = self.suggest_memory_format();
  const bool is_input_nhwc =
      (input_memory_format == at::MemoryFormat::ChannelsLast);
  in_ = FormatContiguous(self, input_memory_format);
  if (is_input_nhwc) {
    // if result tensor comes from UpSampleNearest2d, then it already has the
    // same format as input, otherwise we should make both of them the same
    // format
    if (result.suggest_memory_format() != at::MemoryFormat::ChannelsLast) {
      out_ = result.to(input_memory_format);
    } else {
      out_ = result;
      result_out = true;
    }
  } else {
    if (!result.is_contiguous()) {
      out_ = FormatContiguous(result, input_memory_format);
    } else {
      out_ = result;
      result_out = true;
    }
  }

  in = CreateMUTensor(in_);
  out = CreateMUTensor(out_);
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
  if (is_input_nhwc) {
    out_ = out_.permute({0, 3, 1, 2});
  }

  if (!result_out) {
    result.copy_(out_);
  }
  return result;
}

Tensor UpSampleNearest2d(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto result = at::empty(
      at::native::upsample_2d_common_check(self.sizes(), output_size),
      self.options().memory_format(self.suggest_memory_format()));
  UpSampleNearest2dOut(self, output_size, scales_h, scales_w, result);
  return result;
}

Tensor& UpSampleNearest2dBwdOut(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef contiguous_inputsize,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  MUSA_TENSOR_TYPE_CHECK(grad_output);
  c10::musa::MUSAGuard device_guard(grad_output.device());

  int output_height = output_size[0];
  int output_width = output_size[1];

  int contiguous_inputheight = contiguous_inputsize[2];
  int contiguous_inputwidth = contiguous_inputsize[3];

  float height_scale = at::native::compute_scales_value<float>(
      scales_h, contiguous_inputheight, output_height);
  float width_scale = at::native::compute_scales_value<float>(
      scales_w, contiguous_inputwidth, output_width);
  const auto grad_input_memory_format = grad_input.suggest_memory_format();

  // mtDNN use the reverse scale in bwd.
  const float h_scale = 1. / height_scale;
  const float w_scale = 1. / width_scale;

  Tensor contiguous_grad_output =
      FormatContiguous(grad_output, grad_input_memory_format);

  muHandle& h = GetMudnnHandle();
  auto in = CreateMUTensor(contiguous_grad_output);
  auto out = CreateMUTensor(grad_input);

  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
  CHECK_MUDNN_STATUS(op.SetScaleInfo({h_scale, w_scale}), "SetScaleInfo");

  CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");
  return grad_input;
}

Tensor UpSampleNearest2dBwd(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef contiguous_inputsize,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::empty(
      contiguous_inputsize,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));
  UpSampleNearest2dBwdOut(
      grad_output,
      output_size,
      contiguous_inputsize,
      scales_h,
      scales_w,
      grad_input);
  return grad_input;
}

// TODO(chen.feng): cuda porting when align_corners = true
namespace {
struct structured_upsample_bilinear2d_out_musa_out final
    : public at::native::structured_upsample_bilinear2d_out_cuda {
  structured_upsample_bilinear2d_out_musa_out(Tensor& out0)
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
    return proxy_outputs_[output_idx].has_value() ? **proxy_outputs_[output_idx]
                                                  : outputs_[output_idx].get();
  }

  std::array<std::reference_wrapper<Tensor>, 1> outputs_;
  std::array<c10::optional<c10::ExclusivelyOwned<Tensor>>, 1> proxy_outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

struct structured_upsample_bilinear2d_backward_out_musa_out final
    : public at::native::structured_upsample_bilinear2d_backward_out_cuda {
  structured_upsample_bilinear2d_backward_out_musa_out(Tensor& out0)
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
    at::musa::resize_out(out, sizes, strides, options);
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
    at::musa::resize_out(out, sizes, strides, options);
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
} // namespace

at::Tensor& UpsampleBilinear2dOutPorting(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out) {
  structured_upsample_bilinear2d_out_musa_out op(out);
  op.meta(self, output_size, align_corners, scales_h, scales_w);
  op.impl(
      self,
      output_size,
      align_corners,
      scales_h,
      scales_w,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

at::Tensor& UpsampleBilinear2dBwdOutPorting(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  structured_upsample_bilinear2d_backward_out_musa_out op(grad_input);
  op.meta(
      grad_output, output_size, input_size, align_corners, scales_h, scales_w);
  op.impl(
      grad_output,
      output_size,
      input_size,
      align_corners,
      scales_h,
      scales_w,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return grad_input;
}

Tensor& UpSampleBilinear2dOut(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  MUSA_TENSOR_TYPE_CHECK(self);
  c10::musa::MUSAGuard device_guard(self.device());

  // TODO(chen.feng): cuda porting when align_corners = true
  if (align_corners) {
    return UpsampleBilinear2dOutPorting(
        self, output_size, align_corners, scales_h, scales_w, result);
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int contiguous_inputheight = self.size(2);
  int contiguous_inputwidth = self.size(3);

  const float height_scale = at::native::compute_scales_value<float>(
      scales_h, contiguous_inputheight, output_height);
  const float width_scale = at::native::compute_scales_value<float>(
      scales_w, contiguous_inputwidth, output_width);

  const auto output_memory_format = result.suggest_memory_format();

  // if shapes are the same, just copy and return.
  if (self.sizes() == result.sizes()) {
    result.copy_(self);
  } else if (self.numel() > 0) { // else result should be empty to return
    Tensor contiguous_input = FormatContiguous(self, output_memory_format);

    muHandle& h = GetMudnnHandle();
    auto in = CreateMUTensor(contiguous_input);
    auto out = CreateMUTensor(result);

    ::musa::dnn::Interpolate op;
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Interpolate::Mode::LINEAR), "SetMode");
    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({height_scale, width_scale}), "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.SetAlignCorners(align_corners), "SetAlignCorners");

    CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
  }
  return result;
}

Tensor UpSampleBilinear2d(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto result = at::empty(
      at::native::upsample_2d_common_check(self.sizes(), output_size),
      self.options().memory_format(self.suggest_memory_format()));
  UpSampleBilinear2dOut(
      self, output_size, align_corners, scales_h, scales_w, result);
  return result;
}

Tensor& UpSampleBilinear2dBwdOut(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef contiguous_inputsize,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& grad_input) {
  MUSA_TENSOR_TYPE_CHECK(grad_output);
  c10::musa::MUSAGuard device_guard(grad_output.device());

  // TODO(chen.feng): cuda porting when align_corners = true
  if (align_corners) {
    return UpsampleBilinear2dBwdOutPorting(
        grad_output,
        output_size,
        contiguous_inputsize,
        align_corners,
        scales_h,
        scales_w,
        grad_input);
  }

  int output_height = output_size[0];
  int output_width = output_size[1];

  int contiguous_inputheight = contiguous_inputsize[2];
  int contiguous_inputwidth = contiguous_inputsize[3];

  float height_scale = at::native::compute_scales_value<float>(
      scales_h, contiguous_inputheight, output_height);
  float width_scale = at::native::compute_scales_value<float>(
      scales_w, contiguous_inputwidth, output_width);

  const auto grad_input_memory_format = grad_input.suggest_memory_format();

  // mtDNN use the reverse scale in bwd.
  const float h_scale = 1. / height_scale;
  const float w_scale = 1. / width_scale;

  Tensor contiguous_grad_output =
      FormatContiguous(grad_output, grad_input_memory_format);

  muHandle& h = GetMudnnHandle();
  auto in = CreateMUTensor(contiguous_grad_output);
  auto out = CreateMUTensor(grad_input);

  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::LINEAR), "SetMode");
  CHECK_MUDNN_STATUS(op.SetScaleInfo({h_scale, w_scale}), "SetScaleInfo");
  CHECK_MUDNN_STATUS(op.SetAlignCorners(align_corners), "SetAlignCorners");

  CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");
  return grad_input;
}

Tensor UpSampleBilinear2dBwd(
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef contiguous_inputsize,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  auto grad_input = at::empty(
      contiguous_inputsize,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));
  UpSampleBilinear2dBwdOut(
      grad_output,
      output_size,
      contiguous_inputsize,
      align_corners,
      scales_h,
      scales_w,
      grad_input);
  return grad_input;
}

struct structured_upsample_bicubic2d_out_cuda_out final
    : public at::native::structured_upsample_bicubic2d_out_cuda {
  structured_upsample_bicubic2d_out_cuda_out(Tensor& out0)
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

at::Tensor& UpSampleBicubic2dOut(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "UpSampleBicubic2dOut", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "UpSampleBicubic2dOut", "self");
  structured_upsample_bicubic2d_out_cuda_out op(out);
  op.meta(self, output_size, align_corners, scales_h, scales_w);
  op.impl(
      self,
      output_size,
      align_corners,
      scales_h,
      scales_w,
      op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

struct structured_upsample_linear1d_out_cuda_out final
    : public at::native::structured_upsample_linear1d_out_cuda {
  structured_upsample_linear1d_out_cuda_out(Tensor& out0)
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

at::Tensor& UpSampleLinear1dOut(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "UpSampleLinear1dOut", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "UpSampleLinear1dOut", "self");
  structured_upsample_linear1d_out_cuda_out op(out);
  op.meta(self, output_size, align_corners, scales);
  op.impl(self, output_size, align_corners, scales, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

struct structured_upsample_nearest3d_out_cuda_out final
    : public at::native::structured_upsample_nearest3d_out_cuda {
  structured_upsample_nearest3d_out_cuda_out(Tensor& out0)
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

at::Tensor& UpSampleNearest3dOut(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, out, "UpSampleNearest3dOut", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "UpSampleNearest3dOut", "self");
  structured_upsample_nearest3d_out_cuda_out op(out);
  op.meta(self, output_size, scales_d, scales_h, scales_w);
  op.impl(
      self, output_size, scales_d, scales_h, scales_w, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

struct structured_upsample_nearest1d_out_cuda_out final
    : public at::native::structured_upsample_nearest1d_out_cuda {
  structured_upsample_nearest1d_out_cuda_out(Tensor& out0)
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

at::Tensor& UpSampleNearest1dOut(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales,
    at::Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "UpSampleNearest1dOut", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "UpSampleNearest1dOut", "self");
  structured_upsample_nearest1d_out_cuda_out op(out);
  op.meta(self, output_size, scales);
  op.impl(self, output_size, scales, op.maybe_get_output(0));
  if (op.proxy_outputs_[0].has_value())
    op.outputs_[0].get().copy_(**op.proxy_outputs_[0]);
  return out;
}

ADVANCED_REGISTER(aten, PrivateUse1, "upsample_nearest2d", UpSampleNearest2d)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_nearest2d.out",
    UpSampleNearest2dOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_nearest2d_backward",
    UpSampleNearest2dBwd)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_nearest2d_backward.grad_input",
    UpSampleNearest2dBwdOut)
ADVANCED_REGISTER(aten, PrivateUse1, "upsample_bilinear2d", UpSampleBilinear2d)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_bilinear2d.out",
    UpSampleBilinear2dOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_bilinear2d_backward",
    UpSampleBilinear2dBwd)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_bilinear2d_backward.grad_input",
    UpSampleBilinear2dBwdOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_bicubic2d.out",
    UpSampleBicubic2dOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_linear1d.out",
    UpSampleLinear1dOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_nearest3d.out",
    UpSampleNearest3dOut)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "upsample_nearest1d.out",
    UpSampleNearest1dOut)

} // namespace musa
} // namespace at
