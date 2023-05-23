#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/UpSample.h>
#include <torch/library.h>

#include <mudnn_image.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

Tensor& UpSampleNearest2dOut(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    Tensor& result) {
  MUSA_TENSOR_TYPE_CHECK(self);

  int output_height = output_size[0];
  int output_width = output_size[1];

  int contiguous_inputheight = self.size(2);
  int contiguous_inputwidth = self.size(3);

  const float height_scale = at::native::compute_scales_value<float>(
      scales_h, contiguous_inputheight, output_height);
  const float width_scale = at::native::compute_scales_value<float>(
      scales_w, contiguous_inputwidth, output_width);

  if (self.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(false, "Now not supported NHWC");
  }

  // if shapes are the same, just copy and return.
  if (self.sizes() == result.sizes()) {
    result.copy_(self);
  } else if (self.numel() > 0) { // else result should be empty to return
    Tensor contiguous_input = Contiguous(self);

    muHandle& h = GetMudnnHandle();
    auto in = CreateMUTensor(contiguous_input);
    auto out = CreateMUTensor(result);

    ConfigFormat(contiguous_input, in, true);
    ConfigFormat(result, out, false);

    ::musa::dnn::Interpolate op;
    CHECK_MUDNN_STATUS(
        op.SetMode(::musa::dnn::Interpolate::Mode::NEAREST), "SetMode");
    CHECK_MUDNN_STATUS(
        op.SetScaleInfo({height_scale, width_scale}), "SetScaleInfo");

    CHECK_MUDNN_STATUS(op.Run(h, out, in), "Run");
  }
  return result;
}

Tensor UpSampleNearest2d(
    const Tensor& self,
    IntArrayRef output_size,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  c10::musa::MUSAGuard device_guard(self.device());
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

  int output_height = output_size[0];
  int output_width = output_size[1];

  int contiguous_inputheight = contiguous_inputsize[2];
  int contiguous_inputwidth = contiguous_inputsize[3];

  float height_scale = at::native::compute_scales_value<float>(
      scales_h, contiguous_inputheight, output_height);
  float width_scale = at::native::compute_scales_value<float>(
      scales_w, contiguous_inputwidth, output_width);

  if (grad_output.suggest_memory_format() == at::MemoryFormat::ChannelsLast) {
    TORCH_CHECK(false, "Now not supported NHWC");
  }

  // mtDNN use the reverse scale in bwd.
  const float h_scale = 1. / height_scale;
  const float w_scale = 1. / width_scale;

  Tensor contiguous_input = Contiguous(grad_output);

  muHandle& h = GetMudnnHandle();
  auto in = CreateMUTensor(contiguous_input);
  auto out = CreateMUTensor(grad_input);

  ConfigFormat(contiguous_input, in, true);
  ConfigFormat(grad_input, out, false);

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
  c10::musa::MUSAGuard device_guard(grad_output.device());
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("upsample_nearest2d", &UpSampleNearest2d);
  m.impl("upsample_nearest2d.out", &UpSampleNearest2dOut);
  m.impl("upsample_nearest2d_backward", &UpSampleNearest2dBwd);
  m.impl("upsample_nearest2d_backward.grad_input", &UpSampleNearest2dBwdOut);
}

} // namespace musa
} // namespace at
