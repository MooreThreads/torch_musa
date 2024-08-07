#include <ATen/ATen.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <float.h>
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
__global__ void roi_pool_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    const T* rois,
    T* output,
    int* argmax_data) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];
    int roi_start_w = round(offset_rois[1] * spatial_scale);
    int roi_start_h = round(offset_rois[2] * spatial_scale);
    int roi_end_w = round(offset_rois[3] * spatial_scale);
    int roi_end_h = round(offset_rois[4] * spatial_scale);

    // Force malformed ROIs to be 1x1
    int roi_width = max(roi_end_w - roi_start_w + 1, 1);
    int roi_height = max(roi_end_h - roi_start_h + 1, 1);
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

    // Define an empty pooling region to be zero
    T maxval = is_empty ? 0 : -FLT_MAX;
    // If nothing is pooled, argmax = -1 causes nothing to be backprop'd
    int maxidx = -1;
    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;
    for (int h = hstart; h < hend; ++h) {
      for (int w = wstart; w < wend; ++w) {
        int input_index = h * width + w;
        if (offset_input[input_index] > maxval) {
          maxval = offset_input[input_index];
          maxidx = input_index;
        }
      }
    }
    output[index] = maxval;
    argmax_data[index] = maxidx;
  }
}

template <typename T>
__global__ void roi_pool_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const int* argmax_data,
    int num_rois,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    const int output_offset = n * n_stride + c * c_stride;
    const int* argmax_data_offset =
        argmax_data + (n * channels + c) * pooled_height * pooled_width;
    const int argmax = argmax_data_offset[ph * pooled_width + pw];
    const int offset = (roi_batch_ind * channels + c) * height * width;

    if (argmax != -1) {
      at::native::fastAtomicAdd(
          grad_input,
          offset + argmax,
          memory_span,
          static_cast<T>(
              grad_output[output_offset + ph * h_stride + pw * w_stride]),
          true);
    }
  }
}

std::tuple<at::Tensor, at::Tensor> roi_pool_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  TORCH_CHECK(input.is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a MUSA tensor");
  TORCH_CHECK(
      rois.size(1) == 5, "Tensor rois should have shape as Tensor[K, 5]");

  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_pool_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::musa::MUSAGuard device_guard(input.device());

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  at::Tensor argmax = at::zeros(
      {num_rois, channels, pooled_height, pooled_width},
      input.options().dtype(at::kInt));

  auto output_size = num_rois * pooled_height * pooled_width * channels;
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);

  if (output.numel() == 0) {
    AT_MUSA_CHECK(musaGetLastError());
    return std::make_tuple(output, argmax);
  }

  auto input_ = input.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      input.scalar_type(), "roi_pool_forward_kernel", [&] {
        roi_pool_forward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            output_size,
            input_.data_ptr<scalar_t>(),
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            rois_.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            argmax.data_ptr<int>());
      });
  AT_MUSA_CHECK(musaGetLastError());
  return std::make_tuple(output, argmax);
}

at::Tensor roi_pool_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    const at::Tensor& argmax,
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
  TORCH_CHECK(argmax.is_privateuseone(), "argmax must be a MUSA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2},
      argmax_t{argmax, "argmax", 3};

  at::CheckedFrom c = "roi_pool_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t, argmax_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::musa::MUSAGuard device_guard(grad.device());

  auto num_rois = rois.size(0);

  at::Tensor grad_input =
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

  int n_stride = grad.stride(0);
  int c_stride = grad.stride(1);
  int h_stride = grad.stride(2);
  int w_stride = grad.stride(3);

  at::globalContext().alertNotDeterministic("roi_pool_backward_kernel");

  auto argmax_ = argmax.contiguous(), rois_ = rois.contiguous();
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad.scalar_type(), "roi_pool_backward_kernel", [&] {
        roi_pool_backward_kernel_impl<scalar_t><<<grid, block, 0, stream>>>(
            grad.numel(),
            grad.data_ptr<scalar_t>(),
            argmax_.data_ptr<int>(),
            num_rois,
            spatial_scale,
            channels,
            height,
            width,
            pooled_height,
            pooled_width,
            grad_input.data_ptr<scalar_t>(),
            rois_.data_ptr<scalar_t>(),
            n_stride,
            c_stride,
            h_stride,
            w_stride,
            grad_input.numel());
      });
  AT_MUSA_CHECK(musaGetLastError());
  return grad_input;
}

} // namespace

namespace {

std::tuple<at::Tensor, at::Tensor> roi_pool(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  C10_LOG_API_USAGE_ONCE(
      "torch_musa.csrc.aten.ops.torchvision.roi_pool_kernel.roi_pool");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_pool", "")
                       .typed<decltype(roi_pool)>();
  return op.call(input, rois, spatial_scale, pooled_height, pooled_width);
}

std::tuple<at::Tensor, at::Tensor> roi_pool_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  auto result = roi_pool(
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
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN(roi_pool_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_pool_backward"),
      TORCH_FN(roi_pool_backward_kernel));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_pool"),
      TORCH_FN(roi_pool_autocast));
}

} // namespace ops
} // namespace vision
