#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/UpSample.h>
#include <c10/util/Exception.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

namespace {

void UpSampleNdCommonDtypeCheck(
    const char* ref_from,
    const Tensor& ref,
    const char* test_from,
    const Tensor& test,
    const char* func_from) {
  const auto ref_dtype = ref.scalar_type();
  const auto test_dtype = test.scalar_type();
  if (C10_UNLIKELY(ref_dtype != test_dtype)) {
    TORCH_CHECK(
        false,
        func_from,
        " expects the same dtype for ",
        ref_from,
        " and ",
        test_from,
        ", but got ",
        ref_dtype,
        " and ",
        test_dtype);
  }
}

void UpSampleNdCommonDeviceCheck(
    const char* ref_from,
    const Tensor& ref,
    const char* test_from,
    const Tensor& test,
    const char* func_from) {
  const auto ref_device = ref.device();
  const auto test_device = test.device();
  if (C10_UNLIKELY(ref_device != test_device)) {
    TORCH_CHECK(
        false,
        func_from,
        " expects the same device for ",
        ref_from,
        " and ",
        test_from,
        ", but got ",
        ref_device,
        " and ",
        test_device);
  }
}

C10_ALWAYS_INLINE float ComputeScalesValueBwd(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.)
      ? static_cast<float>(scale.value())
      : static_cast<float>(src_size) / static_cast<float>(dst_size);
}

} // anonymous namespace

Tensor& UpSampleNearest2dOut(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  TORCH_MUSA_CHECK_FLOATING_TYPES_AND_N(
      self.scalar_type(), "UpSampleNearest2dOut", ScalarType::Byte);
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(
      self.dim() == 4,
      "UpSampleNearest2dOut needs input to be a 4-D tensor, which is ",
      self.dim());
  TORCH_CHECK(
      result.dim() == 4,
      "UpSampleNearest2dOut needs output to be a 4-D tensor, which is ",
      result.dim());

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
  TORCH_MUSA_CHECK_FLOATING_TYPES_AND_N(
      grad_output.scalar_type(), "UpSampleNearest2dBwdOut", ScalarType::Byte);
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

Tensor& UpSampleBilinear2dOut(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  TORCH_MUSA_CHECK_FLOATING_TYPES(self.scalar_type(), "UpSampleBilinear2dOut");
  c10::musa::MUSAGuard device_guard(self.device());

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
  TORCH_MUSA_CHECK_FLOATING_TYPES(
      grad_output.scalar_type(), "UpSampleBilinear2dBwdOut");
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

Tensor& UpSampleNearest3dOut(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& output) {
  UpSampleNdCommonDeviceCheck("self", self, "output", output, __func__);
  TORCH_MUSA_CHECK_FLOATING_TYPES_AND_N(
      self.scalar_type(), "UpSampleNearest3dOut", ScalarType::Byte);
  UpSampleNdCommonDtypeCheck("self", self, "output", output, __func__);

  if (C10_UNLIKELY(self.numel() == 0)) {
    return output;
  }
  const c10::musa::MUSAGuard device_guard(self.device());
  if (C10_UNLIKELY(self.sizes() == output.sizes())) {
    output.copy_(self);
    return output;
  }

  const auto input_depth = self.size(2);
  const auto output_depth = output_size[0];
  const auto depth_scale =
      native::compute_scales_value<float>(scales_d, input_depth, output_depth);

  const auto input_height = self.size(3);
  const auto output_height = output_size[1];
  const auto height_scale = native::compute_scales_value<float>(
      scales_h, input_height, output_height);

  const auto input_width = self.size(4);
  const auto output_width = output_size[2];
  const auto width_scale =
      native::compute_scales_value<float>(scales_w, input_width, output_width);

  const auto output_format = output.suggest_memory_format();
  const bool is_output_format_contig = output.is_contiguous(output_format);

  const auto contig_input = FormatContiguous(self, output_format);
  const auto contig_output = FormatContiguous(output, output_format);

  auto in = CreateMUTensor(contig_input);
  auto out = CreateMUTensor(contig_output);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
  CHECK_MUDNN_STATUS(
      op.SetScaleInfo({depth_scale, height_scale, width_scale}),
      "SetScaleInfo");
  CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");

  if (C10_UNLIKELY(!is_output_format_contig)) {
    output.copy_(contig_output);
  }
  return output;
}

Tensor UpSampleNearest3d(
    const at::Tensor& self,
    at::IntArrayRef output_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  const auto input_sizes = self.sizes();
  // Allow for empty batch size but not other dimensions
  TORCH_CHECK(
      self.numel() != 0 ||
          c10::multiply_integers(input_sizes.cbegin() + 1, input_sizes.cend()),
      "Non-empty 5D data tensor expected but got a tensor with sizes ",
      input_sizes);

  auto output = at::empty(
      at::native::upsample_3d_common_check(input_sizes, output_size),
      self.options().memory_format(self.suggest_memory_format()));
  UpSampleNearest3dOut(self, output_size, scales_d, scales_h, scales_w, output);
  return output;
}

Tensor& UpSampleNearest3dBwdOut(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  UpSampleNdCommonDeviceCheck(
      "grad_output", grad_output, "grad_input", grad_input, __func__);
  TORCH_MUSA_CHECK_FLOATING_TYPES_AND_N(
      grad_output.scalar_type(), "UpSampleNearest3dOut", ScalarType::Byte);
  UpSampleNdCommonDtypeCheck(
      "grad_output", grad_output, "grad_input", grad_input, __func__);

  if (C10_UNLIKELY(grad_input.numel() == 0)) {
    return grad_input;
  }
  const c10::musa::MUSAGuard device_guard(grad_output.device());

  const int64_t output_depth = output_size[0];
  const int64_t input_depth = input_size[2];
  const auto depth_scale =
      ComputeScalesValueBwd(scales_d, output_depth, input_depth);

  const int64_t output_height = output_size[1];
  const int64_t input_height = input_size[3];
  const auto height_scale =
      ComputeScalesValueBwd(scales_h, output_height, input_height);

  const int64_t output_width = output_size[2];
  const int64_t input_width = input_size[4];
  const auto width_scale =
      ComputeScalesValueBwd(scales_w, output_width, input_width);

  const auto grad_input_format = grad_input.suggest_memory_format();
  const bool is_grad_input_format_contig =
      grad_input.is_contiguous(grad_input_format);
  const auto contig_grad_input =
      FormatContiguous(grad_input, grad_input_format);
  const auto contig_grad_output =
      FormatContiguous(grad_output, grad_input_format);

  auto in = CreateMUTensor(contig_grad_output);
  auto out = CreateMUTensor(contig_grad_input);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
  CHECK_MUDNN_STATUS(
      op.SetScaleInfo({depth_scale, height_scale, width_scale}),
      "SetScaleInfo");
  CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");

  if (C10_UNLIKELY(!is_grad_input_format_contig)) {
    grad_input.copy_(contig_grad_input);
  }
  return grad_input;
}

Tensor UpSampleNearest3dBwd(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  const auto grad_output_dim = grad_output.dim();
  TORCH_CHECK(
      grad_output_dim == 5,
      "Expected grad_output to be a tensor of dimension 5 but got: dimension ",
      grad_output_dim);

  const auto full_output_sizes =
      at::native::upsample_3d_common_check(input_size, output_size);
  for (auto i : c10::irange(5)) {
    const auto grad_output_dim_i = grad_output.size(i);
    const auto full_output_dim_i = full_output_sizes[i];
    TORCH_CHECK(
        grad_output_dim_i == full_output_dim_i,
        "Expected grad_output to have the same shape as output;",
        " output.size(",
        i,
        ") = ",
        full_output_dim_i,
        " but got grad_output.size(",
        i,
        ") = ",
        grad_output_dim_i);
  }

  auto grad_input = at::empty(
      input_size,
      grad_output.options().memory_format(grad_output.suggest_memory_format()));
  UpSampleNearest3dBwdOut(
      grad_output,
      output_size,
      input_size,
      scales_d,
      scales_h,
      scales_w,
      grad_input);
  return grad_input;
}

} // namespace musa
} // namespace at
