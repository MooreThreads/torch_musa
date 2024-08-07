#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/native/musa/KernelUtils.muh>
#include "torch_musa/csrc/amp/autocast_mode.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "musa_helpers.h"

namespace vision {
namespace ops {

namespace {

template <typename T>
__global__ void ps_roi_pool_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    int channels_out,
    T* output,
    int* channel_mapping) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c_out, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c_out = (index / pooled_width / pooled_height) % channels_out;
    int n = index / pooled_width / pooled_height / channels_out;

    // (n, c_in, ph, pw) is the associated element in the input
    int c_in = (c_out * pooled_height + ph) * pooled_width + pw;

    // [start, end) interval for spatial sampling
    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w, 1);
    int roi_height = max(roi_end_h - roi_start_h, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height - 1);
    hend = min(max(hend + roi_start_h, 0), height - 1);
    wstart = min(max(wstart + roi_start_w, 0), width - 1);
    wend = min(max(wend + roi_start_w, 0), width - 1);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    const T* offset_input =
        input + (roi_batch_ind * channels + c_in) * height * width;
    T out_sum = 0;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        out_sum += offset_input[input_index];
      }
    }

    T bin_area = (hend - hstart) * (wend - wstart);
    output[index] = is_empty ? static_cast<T>(0) : out_sum / bin_area;
    channel_mapping[index] = c_in;
  }
}

template <typename T>
__global__ void ps_roi_pool_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* channel_mapping,
    int num_rois,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    int channels_out,
    T* grad_input,
    const T* rois,
    const int memory_span) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, *, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int n = index / pooled_width / pooled_height / channels_out;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = roundf(offset_rois[1] * spatial_scale);
    int roi_start_h = roundf(offset_rois[2] * spatial_scale);
    int roi_end_w = roundf(offset_rois[3] * spatial_scale);
    int roi_end_h = roundf(offset_rois[4] * spatial_scale);

    // Force too small ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w, 1);
    int roi_height = max(roi_end_h - roi_start_h, 1);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    int hstart = static_cast<int>(floor(static_cast<T>(ph) * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<T>(pw) * bin_size_w));
    int hend = static_cast<int>(ceil(static_cast<T>(ph + 1) * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<T>(pw + 1) * bin_size_w));

    // Add roi offsets and clip to input boundaries
    hstart = min(max(hstart + roi_start_h, 0), height);
    hend = min(max(hend + roi_start_h, 0), height);
    wstart = min(max(wstart + roi_start_w, 0), width);
    wend = min(max(wend + roi_start_w, 0), width);
    bool is_empty = (hend <= hstart) || (wend <= wstart);

    int c_in = channel_mapping[index];
    T bin_area = (hend - hstart) * (wend - wstart);
    T diff_val = is_empty ? static_cast<T>(0) : grad_output[index] / bin_area;

    const int offset = (roi_batch_ind * channels + c_in) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int grad_input_index = h * width + w;
        at::native::fastAtomicAdd(
            grad_input, offset + grad_input_index, memory_span, diff_val, true);
      }
    }
  }
}

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  // Check if input tensors are MUSA tensors
  TORCH_CHECK(input.is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a MUSA tensor");
  TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "ps_roi_pool_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::musa::MUSAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  TORCH_CHECK(
      channels % (pooled_height * pooled_width) == 0,
      "input channels must be a multiple of pooling height * pooling width");
  int channels_out = channels / (pooled_height * pooled_width);

  auto output = at::zeros(
      {num_rois, channels_out, pooled_height, pooled_width}, input.options());
  auto channel_mapping =
      at::zeros(output.sizes(), input.options().dtype(at::kInt));

  auto output_size = output.numel();
  if (output_size == 0) {
    AT_MUSA_CHECK(musaGetLastError());
    return std::make_tuple(output, channel_mapping);
  }

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "ps_roi_pool_forward_kernel", [&] {
        ps_roi_pool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            channels_out,
            output.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>());
      });
  AT_MUSA_CHECK(musaGetLastError());
  return std::make_tuple(output, channel_mapping);
}

at::Tensor ps_roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& channel_mapping,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width) {
  // Check if input tensors are MUSA tensors
  TORCH_CHECK(grad.is_privateuseone(), "grad must be a MUSA tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a MUSA tensor");
  TORCH_CHECK(
      channel_mapping.is_privateuseone(),
      "channel_mapping must be a MUSA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      channel_mapping_t{channel_mapping, "channel_mapping", 3};

  at::CheckedFrom c = "ps_roi_pool_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, channel_mapping_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::musa::MUSAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);
  auto grad_input =
      at::zeros({batch_size, channels, height, width}, grad.options());

  musaStream_t stream = at::musa::getCurrentMUSAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(grad.numel()), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  // handle possibly empty gradients
  if (grad.numel() == 0) {
    AT_MUSA_CHECK(musaGetLastError());
    return grad_input;
  }

  int channels_out = channels / (pooled_height * pooled_width);

  at::globalContext().alertNotDeterministic("ps_roi_pool_backward_kernel");

  auto grad_ = grad.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "ps_roi_pool_backward_kernel", [&] {
        ps_roi_pool_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad_.data_ptr<scalar_t>(),
            channel_mapping.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            channels_out,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            grad_input.numel());
      });
  AT_MUSA_CHECK(musaGetLastError());
  return grad_input;
}

} // namespace

namespace {

std::tuple<at::Tensor, at::Tensor> ps_roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  C10_LOG_API_USAGE_ONCE(
      "torch_musa.csrc.aten.ops.torchvision.ps_roi_pool_kernel.ps_roi_pool");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::ps_roi_pool", "")
                       .typed<decltype(ps_roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

std::tuple<at::Tensor, at::Tensor> ps_roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  auto result = ps_roi_pool(
      at::musa::autocast::cached_cast(at::kFloat, input),
      at::musa::autocast::cached_cast(at::kFloat, rois),
      spatial_scale,
      pooled_height,
      pooled_width);

  return std::make_tuple(
      std::get<0>(result).to(input.scalar_type()),
      std::get<1>(result).to(input.scalar_type()));
}
} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_pool"),
      TORCH_FN(ps_roi_pool_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_ps_roi_pool_backward"),
      TORCH_FN(ps_roi_pool_backward_kernel));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::ps_roi_pool"),
      TORCH_FN(ps_roi_pool_autocast));
}

} // namespace ops
} // namespace vision
