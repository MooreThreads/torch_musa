#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/UpSample.h>
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bilinear2d_aa_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact1d_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact2d_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_backward_native.h>
#include <ATen/ops/_upsample_nearest_exact3d_native.h>
#include <ATen/ops/upsample_bicubic2d_backward_native.h>
#include <ATen/ops/upsample_bicubic2d_native.h>
#include <ATen/ops/upsample_bilinear2d_backward_native.h>
#include <ATen/ops/upsample_bilinear2d_native.h>
#include <ATen/ops/upsample_linear1d_backward_native.h>
#include <ATen/ops/upsample_linear1d_native.h>
#include <ATen/ops/upsample_nearest1d_backward_native.h>
#include <ATen/ops/upsample_nearest1d_native.h>
#include <ATen/ops/upsample_nearest2d_backward_native.h>
#include <ATen/ops/upsample_nearest2d_native.h>
#include <ATen/ops/upsample_nearest3d_backward_native.h>
#include <ATen/ops/upsample_nearest3d_native.h>
#include <ATen/ops/upsample_trilinear3d_backward_native.h>
#include <ATen/ops/upsample_trilinear3d_native.h>
#endif

#include <c10/util/Exception.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

using INTERPOLATE_MODE = ::musa::dnn::Interpolate::Mode;

namespace {

C10_ALWAYS_INLINE float ComputeScalesValueBwd(
    const c10::optional<double> scale,
    int64_t src_size,
    int64_t dst_size) {
  return (scale.has_value() && scale.value() > 0.)
      ? static_cast<float>(scale.value())
      : static_cast<float>(src_size) / static_cast<float>(dst_size);
}

template <int Nd, INTERPOLATE_MODE mode>
Tensor& UpSampleNdOut(
    const Tensor& self,
    IntArrayRef output_size,
    bool align_corners,
    bool antialias,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  if (self.numel() == 0) {
    return result;
  }
  c10::musa::MUSAGuard device_guard(self.device());

  if (self.sizes() == result.sizes()) {
    result.copy_(self);
    return result;
  }

  const auto input_memory_format = self.suggest_memory_format();
  const bool is_input_channel_last = self.dim() <= 4
      ? (input_memory_format == at::MemoryFormat::ChannelsLast)
      : (input_memory_format == at::MemoryFormat::ChannelsLast3d);
  bool result_out = false;

  Tensor in_ = FormatContiguous(self, input_memory_format);
  Tensor out_;
  if (is_input_channel_last) {
    // if result tensor comes from UpSampleNearest2d, then it already has the
    // same format as input, otherwise we should make both of them the same
    // format
    if (result.suggest_memory_format() != at::MemoryFormat::ChannelsLast ||
        result.suggest_memory_format() != at::MemoryFormat::ChannelsLast3d) {
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

  muTensor in = CreateMUTensor(in_);
  muTensor out = CreateMUTensor(out_);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(op.SetAntialias(antialias), "SetAntialias");
  if constexpr (
      mode == INTERPOLATE_MODE::LINEAR || mode == INTERPOLATE_MODE::BICUBIC) {
    CHECK_MUDNN_STATUS(op.SetAlignCorners(align_corners), "SetAlignCorners");
  }

  if constexpr (Nd == 1) {
    // Since pytorch only support channelslast for 4D tensor and
    // channelslast3d for 5D tensor, we don't use channelslast
    // for 3D tensor in 1D upsample
    CHECK_MUDNN_STATUS(in.SetFormat(muTensor::Format::NCW), "SetFormat");
    CHECK_MUDNN_STATUS(out.SetFormat(muTensor::Format::NCW), "SetFormat");

    int input_width = self.size(2);
    int output_width = output_size[0];
    const float width_scale = at::native::compute_scales_value<float>(
        scales_w, input_width, output_width);

    CHECK_MUDNN_STATUS(op.SetScaleInfo({width_scale}), "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
  } else if constexpr (Nd == 2) {
    int input_height = self.size(2);
    int input_width = self.size(3);
    int output_height = output_size[0];
    int output_width = output_size[1];
    const float height_scale = at::native::compute_scales_value<float>(
        scales_h, input_height, output_height);
    const float width_scale = at::native::compute_scales_value<float>(
        scales_w, input_width, output_width);

    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({height_scale, width_scale}), "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");

  } else if constexpr (Nd == 3) {
    int input_depth = self.size(2);
    int input_height = self.size(3);
    int input_width = self.size(4);
    int output_depth = output_size[0];
    int output_height = output_size[1];
    int output_width = output_size[2];

    const auto depth_scale = native::compute_scales_value<float>(
        scales_d, input_depth, output_depth);
    const auto height_scale = native::compute_scales_value<float>(
        scales_h, input_height, output_height);
    const auto width_scale = native::compute_scales_value<float>(
        scales_w, input_width, output_width);

    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({depth_scale, height_scale, width_scale}),
        "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");

  } else {
    TORCH_CHECK(false, "UpSample only support for 1D, 2D and 3D");
  }

  if (!result_out) {
    result.copy_(out_);
  }
  return result;
}

template <int Nd, INTERPOLATE_MODE mode>
Tensor& UpSampleNdBwdOut(
    const at::Tensor& grad_output,
    at::IntArrayRef output_size,
    at::IntArrayRef input_size,
    bool align_corners,
    bool antialias,
    c10::optional<double> scales_d,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    at::Tensor& grad_input) {
  if (C10_UNLIKELY(grad_input.numel() == 0)) {
    return grad_input;
  }
  const c10::musa::MUSAGuard device_guard(grad_output.device());

  const auto grad_input_format = grad_input.suggest_memory_format();
  const bool is_grad_input_format_contig =
      grad_input.is_contiguous(grad_input_format);
  const auto contig_grad_input =
      FormatContiguous(grad_input, grad_input_format);
  const auto contig_grad_output =
      FormatContiguous(grad_output, grad_input_format);

  muTensor in = CreateMUTensor(contig_grad_output);
  muTensor out = CreateMUTensor(contig_grad_input);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Interpolate op;
  CHECK_MUDNN_STATUS(
      op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
  if constexpr (mode == INTERPOLATE_MODE::LINEAR) {
    CHECK_MUDNN_STATUS(op.SetAlignCorners(align_corners), "SetAlignCorners");
  }

  if constexpr (Nd == 1) {
    // Since pytorch only support channelslast for 4D tensor and
    // channelslast3d for 5D tensor, we don't use channelslast
    // for 3D tensor in 1D upsample
    CHECK_MUDNN_STATUS(in.SetFormat(muTensor::Format::NCW), "SetFormat");
    CHECK_MUDNN_STATUS(out.SetFormat(muTensor::Format::NCW), "SetFormat");

    int64_t output_width = output_size[0];
    int64_t input_width = input_size[2];

    const auto width_scale =
        ComputeScalesValueBwd(scales_w, output_width, input_width);

    CHECK_MUDNN_STATUS(op.SetScaleInfo({width_scale}), "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");
  } else if constexpr (Nd == 2) {
    int64_t output_height = output_size[0];
    int64_t output_width = output_size[1];

    int64_t input_height = input_size[2];
    int64_t input_width = input_size[3];

    const auto height_scale =
        ComputeScalesValueBwd(scales_h, output_height, input_height);
    const auto width_scale =
        ComputeScalesValueBwd(scales_w, output_width, input_width);

    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({height_scale, width_scale}), "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");
  } else if constexpr (Nd == 3) {
    int64_t output_depth = output_size[0];
    int64_t output_height = output_size[1];
    int64_t output_width = output_size[2];

    int64_t input_depth = input_size[2];
    int64_t input_height = input_size[3];
    int64_t input_width = input_size[4];

    const auto depth_scale =
        ComputeScalesValueBwd(scales_d, output_depth, input_depth);
    const auto height_scale =
        ComputeScalesValueBwd(scales_h, output_height, input_height);
    const auto width_scale =
        ComputeScalesValueBwd(scales_w, output_width, input_width);

    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({depth_scale, height_scale, width_scale}),
        "SetScaleInfo");
    CHECK_MUDNN_STATUS(op.RunBackward(h, out, in), "RunBackward");
  }

  if (C10_UNLIKELY(!is_grad_input_format_contig)) {
    grad_input.copy_(contig_grad_input);
  }
  return grad_input;
}

} // anonymous namespace

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Forward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Nearest 1D, 2D, 3D
TORCH_IMPL_FUNC(upsample_nearest1d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales,
 const Tensor& output) {
  UpSampleNdOut<1, INTERPOLATE_MODE::NEAREST>(
      input,
      output_size,
      false,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(upsample_nearest2d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::NEAREST>(
      input,
      output_size,
      false,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(upsample_nearest3d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<3, INTERPOLATE_MODE::NEAREST>(
      input,
      output_size,
      false,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

// Linear && BiLinear && TriLinear
TORCH_IMPL_FUNC(upsample_linear1d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales,
 const Tensor& output) {
  UpSampleNdOut<1, INTERPOLATE_MODE::LINEAR>(
      input,
      output_size,
      align_corners,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(upsample_bilinear2d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::LINEAR>(
      input,
      output_size,
      align_corners,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::LINEAR>(
      input,
      output_size,
      align_corners,
      true,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(upsample_trilinear3d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<3, INTERPOLATE_MODE::LINEAR>(
      input,
      output_size,
      align_corners,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

// Bicubic2d
TORCH_IMPL_FUNC(upsample_bicubic2d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::BICUBIC>(
      input,
      output_size,
      align_corners,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::BICUBIC>(
      input,
      output_size,
      align_corners,
      true,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

// Nearest Exact 1d, 2d, 3d
TORCH_IMPL_FUNC(_upsample_nearest_exact1d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales,
 const Tensor& output) {
  UpSampleNdOut<1, INTERPOLATE_MODE::NEAREST_EXACT>(
      input,
      output_size,
      false,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<2, INTERPOLATE_MODE::NEAREST_EXACT>(
      input,
      output_size,
      false,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_out_musa)
(const Tensor& input,
 IntArrayRef output_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& output) {
  UpSampleNdOut<3, INTERPOLATE_MODE::NEAREST_EXACT>(
      input,
      output_size,
      false,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(output));
}

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Backward ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Nearest 1D, 2D, 3D
TORCH_IMPL_FUNC(upsample_nearest1d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<1, INTERPOLATE_MODE::NEAREST>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(upsample_nearest2d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::NEAREST>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(upsample_nearest3d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<3, INTERPOLATE_MODE::NEAREST>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

// Linear, Bilinear, Trilinear backward
TORCH_IMPL_FUNC(upsample_linear1d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<1, INTERPOLATE_MODE::LINEAR>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(upsample_bilinear2d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::LINEAR>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(_upsample_bilinear2d_aa_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::LINEAR>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      true,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(upsample_trilinear3d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<3, INTERPOLATE_MODE::LINEAR>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

// Bicubic2dBwd
TORCH_IMPL_FUNC(upsample_bicubic2d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::BICUBIC>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 bool align_corners,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::BICUBIC>(
      grad_output,
      output_size,
      input_size,
      align_corners,
      true,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

// Nearest Exact
TORCH_IMPL_FUNC(_upsample_nearest_exact1d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<1, INTERPOLATE_MODE::NEAREST_EXACT>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      c10::nullopt,
      c10::nullopt,
      scales,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(_upsample_nearest_exact2d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<2, INTERPOLATE_MODE::NEAREST_EXACT>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      c10::nullopt,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

TORCH_IMPL_FUNC(_upsample_nearest_exact3d_backward_out_musa)
(const Tensor& grad_output,
 IntArrayRef output_size,
 IntArrayRef input_size,
 std::optional<double> scales_d,
 std::optional<double> scales_h,
 std::optional<double> scales_w,
 const Tensor& grad_input) {
  UpSampleNdBwdOut<3, INTERPOLATE_MODE::NEAREST_EXACT>(
      grad_output,
      output_size,
      input_size,
      false,
      false,
      scales_d,
      scales_h,
      scales_w,
      const_cast<Tensor&>(grad_input));
}

} // namespace musa
} // namespace at
