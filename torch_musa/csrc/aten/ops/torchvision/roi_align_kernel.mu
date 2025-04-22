#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>
#include <ATen/native/musa/KernelUtils.muh>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAMarcos.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include "musa_helpers.h"

namespace vision {
namespace ops {

namespace {
template <typename T>
__device__ T bilinear_interpolate(
    const T* input,
    int height,
    int width,
    T y,
    T x,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    return 0;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  int y_low = (int)y;
  int x_low = (int)x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // do bilinear interpolation
  T v1 = input[y_low * width + x_low];
  T v2 = input[y_low * width + x_high];
  T v3 = input[y_high * width + x_low];
  T v4 = input[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}

template <typename T, bool Aligned, bool SampleRatio>
__global__ void roi_align_forward_kernel_impl(
    int nthreads,
    const T* input,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T inverse_pooled_height,
    T inverse_pooled_width,
    int sampling_ratio,
    const T* rois,
    T* output,
    at::musa::FastDivmod fdm_c,
    at::musa::FastDivmod fdm_ph,
    at::musa::FastDivmod fdm_pw) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    uint32_t n, c, ph, pw;
    uint32_t temp = index;
    fdm_pw(ph, pw, temp);
    fdm_ph(c, ph, ph);
    fdm_c(n, c, c);

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = (T)0.0;
    if constexpr (Aligned) {
      offset = (T)0.5;
    }
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if constexpr (!Aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = roi_height * inverse_pooled_height;
    T bin_size_w = roi_width * inverse_pooled_width;

    const T* offset_input =
        input + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = ceil(bin_size_h);
    if constexpr (SampleRatio) {
      roi_bin_grid_h = sampling_ratio;
    }
    int roi_bin_grid_w = ceil(bin_size_w);
    if constexpr (SampleRatio) {
      roi_bin_grid_w = sampling_ratio;
    }

    // We do average (integral) pooling inside a bin
    // When the grid is empty, output zeros.
    const T count = max(roi_bin_grid_h * roi_bin_grid_w, 1); // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T val = bilinear_interpolate(offset_input, height, width, y, x, index);
        output_val += val;
      }
    }
    output_val /= count;

    output[index] = output_val;
  }
}

template <typename T>
__device__ void bilinear_interpolate_gradient(
    int height,
    int width,
    T y,
    T x,
    T& w1,
    T& w2,
    T& w3,
    T& w4,
    int& x_low,
    int& x_high,
    int& y_low,
    int& y_high,
    int index /* index for debug only*/) {
  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    // empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0)
    y = 0;
  if (x <= 0)
    x = 0;

  y_low = (int)y;
  x_low = (int)x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T)y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T)x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = input[y_low * width + x_low];
  // T v2 = input[y_low * width + x_high];
  // T v3 = input[y_high * width + x_low];
  // T v4 = input[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
}

template <typename T, bool Aligned, bool SampleRatio>
__global__ void roi_align_backward_kernel_impl(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T inverse_pooled_height,
    T inverse_pooled_width,
    int sampling_ratio,
    T* grad_input,
    const T* rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span,
    at::musa::FastDivmod fdm_c,
    at::musa::FastDivmod fdm_ph,
    at::musa::FastDivmod fdm_pw) {
  MUSA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    uint32_t n, c, ph, pw;
    uint32_t temp = index;
    fdm_pw(ph, pw, temp);
    fdm_ph(c, ph, ph);
    fdm_c(n, c, c);

    const T* offset_rois = rois + n * 5;
    int roi_batch_ind = offset_rois[0];

    // Do not using rounding; this implementation detail is critical
    T offset = (T)0.0;
    if constexpr (Aligned) {
      offset = (T)0.5;
    }
    T roi_start_w = offset_rois[1] * spatial_scale - offset;
    T roi_start_h = offset_rois[2] * spatial_scale - offset;
    T roi_end_w = offset_rois[3] * spatial_scale - offset;
    T roi_end_h = offset_rois[4] * spatial_scale - offset;

    T roi_width = roi_end_w - roi_start_w;
    T roi_height = roi_end_h - roi_start_h;
    if constexpr (!Aligned) {
      // Force malformed ROIs to be 1x1
      roi_width = max(roi_width, (T)1.);
      roi_height = max(roi_height, (T)1.);
    }

    T bin_size_h = roi_height * inverse_pooled_height;
    T bin_size_w = roi_width * inverse_pooled_width;

    // We need to index the gradient using the tensor strides to access the
    // correct values.
    const int output_offset = n * n_stride + c * c_stride;
    const T* offset_grad_output = grad_output + output_offset;
    const T grad_output_this_bin =
        offset_grad_output[ph * h_stride + pw * w_stride];

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = ceil(bin_size_h);
    if constexpr (SampleRatio) {
      roi_bin_grid_h = sampling_ratio;
    }
    int roi_bin_grid_w = ceil(bin_size_w);
    if constexpr (SampleRatio) {
      roi_bin_grid_w = sampling_ratio;
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    const int input_offset = (roi_batch_ind * channels + c) * height * width;

    for (int iy = 0; iy < roi_bin_grid_h; iy++) // e.g., iy = 0, 1
    {
      const T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
          static_cast<T>(iy + .5f) * bin_size_h /
              static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
      for (int ix = 0; ix < roi_bin_grid_w; ix++) {
        const T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
            static_cast<T>(ix + .5f) * bin_size_w /
                static_cast<T>(roi_bin_grid_w);

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(
            height,
            width,
            y,
            x,
            w1,
            w2,
            w3,
            w4,
            x_low,
            x_high,
            y_low,
            y_high,
            index);

        T g1 = grad_output_this_bin * w1 / count;
        T g2 = grad_output_this_bin * w2 / count;
        T g3 = grad_output_this_bin * w3 / count;
        T g4 = grad_output_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
          at::native::fastAtomicAdd(
              grad_input,
              input_offset + y_low * width + x_low,
              memory_span,
              static_cast<T>(g1),
              true);
          at::native::fastAtomicAdd(
              grad_input,
              input_offset + y_low * width + x_high,
              memory_span,
              static_cast<T>(g2),
              true);
          at::native::fastAtomicAdd(
              grad_input,
              input_offset + y_high * width + x_low,
              memory_span,
              static_cast<T>(g3),
              true);
          at::native::fastAtomicAdd(
              grad_input,
              input_offset + y_high * width + x_high,
              memory_span,
              static_cast<T>(g4),
              true);
        } // if
      } // ix
    } // iy
  } // MUSA_1D_KERNEL_LOOP
}

template <typename T>
__global__ void rois_reorder_kernel(
    T* reorderd_rois,
    const T* rois,
    int num_rois,
    int batch_size) {
  __shared__ int smem[4096];

  int tid = threadIdx.x;
  for (int i = tid; i < 4096; i += blockDim.x) {
    smem[i] = 0;
  }
  __SYNCTHREADS;

  for (int i = tid; i < num_rois; i += blockDim.x) {
    int roi_batch_ind = *(rois + i * 5);
    at::musa::gpuAtomicAdd(&smem[roi_batch_ind], 1);
  }
  __SYNCTHREADS;

  // num of each batch
  for (int i = tid; i < batch_size; i += blockDim.x) {
    reorderd_rois[i] = smem[i];
  }
  __SYNCTHREADS;

#pragma unroll
  for (int i = 0; i < 4; i++) {
    if (i > 0) {
      smem[i * 1024] += smem[i * 1024 - 1];
    }
    __SYNCTHREADS;
#pragma unroll
    for (int j = 1; j < 1024; j <<= 1) {
      if (tid >= j) {
        smem[i * 1024 + tid] += smem[i * 1024 + tid - j];
      }
      __SYNCTHREADS;
    }
  }

  // start_from where
  reorderd_rois[0 + batch_size] = 0;
  for (int i = tid + 1; i < batch_size; i += blockDim.x) {
    reorderd_rois[i + batch_size] = smem[i - 1];
  }
  __SYNCTHREADS;

  // reorder
  T* offset_rois_out = reorderd_rois + batch_size * 2;
  for (int i = tid; i < num_rois; i += blockDim.x) {
    int roi_batch_ind = *(rois + i * 5);
    int new_pos = at::musa::gpuAtomicAdd(&smem[roi_batch_ind], -1) - 1;
    *(offset_rois_out + new_pos * 5) = i;
    *(offset_rois_out + new_pos * 5 + 1) = *(rois + i * 5 + 1);
    *(offset_rois_out + new_pos * 5 + 2) = *(rois + i * 5 + 2);
    *(offset_rois_out + new_pos * 5 + 3) = *(rois + i * 5 + 3);
    *(offset_rois_out + new_pos * 5 + 4) = *(rois + i * 5 + 4);
  }
}

template <typename T, bool Aligned, bool SampleRatio>
__global__ void __attribute__((mtgpu_workgroup_atomic))
roi_align_backward_kernel_workgroup_atomic_impl(
    int nthreads,
    const T* grad_output,
    const T spatial_scale,
    int grad_in_nc,
    int num_rois,
    int batch_size,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    T inverse_pooled_height,
    T inverse_pooled_width,
    int sampling_ratio,
    T* grad_input,
    const T* reorderd_rois,
    int n_stride,
    int c_stride,
    int h_stride,
    int w_stride,
    const int memory_span,
    at::musa::FastDivmod fdm_c,
    at::musa::FastDivmod fdm_ph,
    at::musa::FastDivmod fdm_pw) {
  for (uint32_t i = blockIdx.x; i < grad_in_nc; i += gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    // int c = i % channels;
    // int n = i / channels;
    uint32_t n, c;
    fdm_c(n, c, i);

    int nr_batches = *(reorderd_rois + n);
    int n_start = *(reorderd_rois + n + batch_size);
    const T* rois = reorderd_rois + batch_size * 2;
    for (int r = n_start; r < n_start + nr_batches; r++) {
      const T* offset_rois = rois + r * 5;
      int grad_out_batch_ind = offset_rois[0];

      const int input_offset = (n * channels + c) * height * width;

      // Do not using rounding; this implementation detail is critical
      T offset = (T)0.0;
      if constexpr (Aligned) {
        offset = (T)0.5;
      }
      T roi_start_w = offset_rois[1] * spatial_scale - offset;
      T roi_start_h = offset_rois[2] * spatial_scale - offset;
      T roi_end_w = offset_rois[3] * spatial_scale - offset;
      T roi_end_h = offset_rois[4] * spatial_scale - offset;

      T roi_width = roi_end_w - roi_start_w;
      T roi_height = roi_end_h - roi_start_h;
      if constexpr (!Aligned) {
        // Force malformed ROIs to be 1x1
        roi_width = max(roi_width, (T)1.);
        roi_height = max(roi_height, (T)1.);
      }

      T bin_size_h = roi_height * inverse_pooled_height;
      T bin_size_w = roi_width * inverse_pooled_width;

      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = ceil(bin_size_h);
      if constexpr (SampleRatio) {
        roi_bin_grid_h = sampling_ratio;
      }
      int roi_bin_grid_w = ceil(bin_size_w);
      if constexpr (SampleRatio) {
        roi_bin_grid_w = sampling_ratio;
      }

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

      // We need to index the gradient using the tensor strides to access the
      // correct values.
      const int output_offset = grad_out_batch_ind * n_stride + c * c_stride;
      const T* offset_grad_output = grad_output + output_offset;
      for (uint32_t phw = threadIdx.x; phw < pooled_height * pooled_width;
           phw += blockDim.x) {
        uint32_t ph, pw;
        fdm_pw(ph, pw, phw);

        const T grad_output_this_bin =
            offset_grad_output[ph * h_stride + pw * w_stride];

        for (int iy = 0; iy < roi_bin_grid_h; iy++) { // e.g., iy = 0, 1
          const T y = roi_start_h + static_cast<T>(ph) * bin_size_h +
              static_cast<T>(iy + .5f) * bin_size_h /
                  static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
          for (int ix = 0; ix < roi_bin_grid_w; ix++) {
            const T x = roi_start_w + static_cast<T>(pw) * bin_size_w +
                static_cast<T>(ix + .5f) * bin_size_w /
                    static_cast<T>(roi_bin_grid_w);

            T w1, w2, w3, w4;
            int x_low, x_high, y_low, y_high;
            int index = i * pooled_height * pooled_width + phw;
            bilinear_interpolate_gradient(
                height,
                width,
                y,
                x,
                w1,
                w2,
                w3,
                w4,
                x_low,
                x_high,
                y_low,
                y_high,
                index);

            T g1 = grad_output_this_bin * w1 / count;
            T g2 = grad_output_this_bin * w2 / count;
            T g3 = grad_output_this_bin * w3 / count;
            T g4 = grad_output_this_bin * w4 / count;

            if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0) {
              at::native::fastAtomicAdd(
                  grad_input,
                  input_offset + y_low * width + x_low,
                  memory_span,
                  static_cast<T>(g1),
                  true);
              at::native::fastAtomicAdd(
                  grad_input,
                  input_offset + y_low * width + x_high,
                  memory_span,
                  static_cast<T>(g2),
                  true);
              at::native::fastAtomicAdd(
                  grad_input,
                  input_offset + y_high * width + x_low,
                  memory_span,
                  static_cast<T>(g3),
                  true);
              at::native::fastAtomicAdd(
                  grad_input,
                  input_offset + y_high * width + x_high,
                  memory_span,
                  static_cast<T>(g4),
                  true);
            } // if
          } // ix
        } // iy
      } // thread
    }
  } // block
}

at::Tensor roi_align_forward_kernel(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(input.is_privateuseone(), "input must be a MUSA tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a MUSA tensor");
  TORCH_CHECK(rois.size(1) == 5, "rois must have shape as Tensor[K, 5]");
  at::TensorArg input_t{input, "input", 1}, rois_t{rois, "rois", 2};
  at::CheckedFrom c = "roi_align_forward_kernel";
  at::checkAllSameGPU(c, {input_t, rois_t});
  at::checkAllSameType(c, {input_t, rois_t});

  at::musa::MUSAGuard device_guard(input.device());
  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);
  at::Tensor output = at::zeros(
      {num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
  musaStream_t stream = at::musa::getCurrentMUSAStream();
  dim3 grid(std::min(
      ceil_div(static_cast<int64_t>(output_size), static_cast<int64_t>(512)),
      static_cast<int64_t>(4096)));
  dim3 block(512);
  if (output.numel() == 0) {
    AT_MUSA_CHECK(musaGetLastError());
    return output;
  }

  at::musa::FastDivmod fdm_c(channels);
  at::musa::FastDivmod fdm_ph(pooled_height);
  at::musa::FastDivmod fdm_pw(pooled_width);

  double inverse_pooled_height = (double)1 / (double)pooled_height;
  double inverse_pooled_width = (double)1 / (double)pooled_width;

  auto input_ = input.contiguous(), rois_ = rois.contiguous();

#define LAUNCH_ROI_ALIGN_FWD_KERNEL(ALIGNED, SAMPLE_RATIO)             \
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                 \
      input.scalar_type(), "roi_align_forward_kernel", [&] {           \
        roi_align_forward_kernel_impl<scalar_t, ALIGNED, SAMPLE_RATIO> \
            <<<grid, block, 0, stream>>>(                              \
                output_size,                                           \
                input_.data_ptr<scalar_t>(),                           \
                spatial_scale,                                         \
                channels,                                              \
                height,                                                \
                width,                                                 \
                pooled_height,                                         \
                pooled_width,                                          \
                inverse_pooled_height,                                 \
                inverse_pooled_width,                                  \
                sampling_ratio,                                        \
                rois_.data_ptr<scalar_t>(),                            \
                output.data_ptr<scalar_t>(),                           \
                fdm_c,                                                 \
                fdm_ph,                                                \
                fdm_pw);                                               \
      });

  if (aligned) {
    if (sampling_ratio > 0) {
      LAUNCH_ROI_ALIGN_FWD_KERNEL(true, true)
    } else {
      LAUNCH_ROI_ALIGN_FWD_KERNEL(true, false)
    }
  } else {
    if (sampling_ratio > 0) {
      LAUNCH_ROI_ALIGN_FWD_KERNEL(false, true)
    } else {
      LAUNCH_ROI_ALIGN_FWD_KERNEL(false, false)
    }
  }

  // Empty rois will raise invalid configuration argument error but output is
  // accurate
  AT_MUSA_CHECK(musaGetLastError());
  return output;

#undef LAUNCH_ROI_ALIGN_FWD_KERNEL
}

at::Tensor roi_align_backward_kernel(
    const at::Tensor& grad,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t batch_size,
    int64_t channels,
    int64_t height,
    int64_t width,
    int64_t sampling_ratio,
    bool aligned) {
  TORCH_CHECK(grad.is_privateuseone(), "grad must be a MUSA tensor");
  TORCH_CHECK(rois.is_privateuseone(), "rois must be a MUSA tensor");

  at::TensorArg grad_t{grad, "grad", 1}, rois_t{rois, "rois", 2};

  at::CheckedFrom c = "roi_align_backward_kernel";
  at::checkAllSameGPU(c, {grad_t, rois_t});
  at::checkAllSameType(c, {grad_t, rois_t});

  at::musa::MUSAGuard device_guard(grad.device());

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

  at::musa::FastDivmod fdm_c(channels);
  at::musa::FastDivmod fdm_ph(pooled_height);
  at::musa::FastDivmod fdm_pw(pooled_width);

  double inverse_pooled_height = (double)1 / (double)pooled_height;
  double inverse_pooled_width = (double)1 / (double)pooled_width;

  auto num_rois = rois.size(0);
  auto grad_in_nc = batch_size * channels;

  // device info
  musaDeviceProp device_prop;
  at::musa::muHandle& h = at::GetMudnnHandle();
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  int device_major_version = device_prop.major;

  at::globalContext().alertNotDeterministic("roi_align_backward_kernel");

  auto rois_ = rois.contiguous();
  at::Tensor reordered_rois =
      at::empty({batch_size * 2 + rois_.numel()}, rois_.options());

  bool run_with_global_atomic = batch_size > 4096 || device_major_version >= 3;

#define LAUNCH_ROI_ALIGN_BWD_KERNEL(ALIGNED, SAMPLE_RATIO)                                                          \
  if (run_with_global_atomic) {                                                                                     \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                            \
        grad.scalar_type(), "roi_align_backward_kernel", [&] {                                                      \
          roi_align_backward_kernel_impl<scalar_t, ALIGNED, SAMPLE_RATIO>                                           \
              <<<grid, block, 0, stream>>>(                                                                         \
                  grad.numel(),                                                                                     \
                  grad.data_ptr<scalar_t>(),                                                                        \
                  spatial_scale,                                                                                    \
                  channels,                                                                                         \
                  height,                                                                                           \
                  width,                                                                                            \
                  pooled_height,                                                                                    \
                  pooled_width,                                                                                     \
                  inverse_pooled_height,                                                                            \
                  inverse_pooled_width,                                                                             \
                  sampling_ratio,                                                                                   \
                  grad_input.data_ptr<scalar_t>(),                                                                  \
                  rois_.data_ptr<scalar_t>(),                                                                       \
                  n_stride,                                                                                         \
                  c_stride,                                                                                         \
                  h_stride,                                                                                         \
                  w_stride,                                                                                         \
                  grad_input.numel(),                                                                               \
                  fdm_c,                                                                                            \
                  fdm_ph,                                                                                           \
                  fdm_pw);                                                                                          \
        });                                                                                                         \
  } else {                                                                                                          \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                       \                                   
        rois_.scalar_type(),                                                   \                                          
        "rois_reorder_kernel",                                                 \                 
        [&] {                                                                  \                                              
          rois_reorder_kernel<                                                 \      
              scalar_t><<<1, 1024, 0, stream>>>(                               \         
              reordered_rois.data_ptr<scalar_t>(),                             \                              
              rois_.data_ptr<scalar_t>(),                                      \                                  
              num_rois,                                                        \                                              
              batch_size);                                                     \                                                 
        }); \
    block.x = 512;                                                                                                  \
    grid.x = std::min(grad_in_nc, (int64_t)4096);                                                                   \
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(                                                                            \
        grad.scalar_type(),                                                                                         \
        "roi_align_backward_kernel_workgroup_atomic_impl",                                                          \
        [&] {                                                                                                       \
          roi_align_backward_kernel_workgroup_atomic_impl<                                                          \
              scalar_t,                                                                                             \
              ALIGNED,                                                                                              \
              SAMPLE_RATIO><<<grid, block, 0, stream>>>(                                                            \
              grad.numel(),                                                                                         \
              grad.data_ptr<scalar_t>(),                                                                            \
              spatial_scale,                                                                                        \
              grad_in_nc,                                                                                           \
              num_rois,                                                                                             \
              batch_size,                                                                                           \
              channels,                                                                                             \
              height,                                                                                               \
              width,                                                                                                \
              pooled_height,                                                                                        \
              pooled_width,                                                                                         \
              inverse_pooled_height,                                                                                \
              inverse_pooled_width,                                                                                 \
              sampling_ratio,                                                                                       \
              grad_input.data_ptr<scalar_t>(),                                                                      \
              reordered_rois.data_ptr<scalar_t>(),                                                                  \
              n_stride,                                                                                             \
              c_stride,                                                                                             \
              h_stride,                                                                                             \
              w_stride,                                                                                             \
              grad_input.numel(),                                                                                   \
              fdm_c,                                                                                                \
              fdm_ph,                                                                                               \
              fdm_pw);                                                                                              \
        });                                                                                                         \
  }

  if (aligned) {
    if (sampling_ratio > 0) {
      LAUNCH_ROI_ALIGN_BWD_KERNEL(true, true)
    } else {
      LAUNCH_ROI_ALIGN_BWD_KERNEL(true, false)
    }
  } else {
    if (sampling_ratio > 0) {
      LAUNCH_ROI_ALIGN_BWD_KERNEL(false, true)
    } else {
      LAUNCH_ROI_ALIGN_BWD_KERNEL(false, false)
    }
  }

  AT_MUSA_CHECK(musaGetLastError());
  return grad_input;

#undef LAUNCH_ROI_ALIGN_BWD_KERNEL
}

} // namespace

namespace {

at::Tensor roi_align(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  C10_LOG_API_USAGE_ONCE(
      "torch_musa.csrc.aten.ops.torchvision.roi_align_kernel.roi_align");
  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("torchvision::roi_align", "")
                       .typed<decltype(roi_align)>();
  return op.call(
      input,
      rois,
      spatial_scale,
      pooled_height,
      pooled_width,
      sampling_ratio,
      aligned);
}

at::Tensor roi_align_autocast(
    const at::Tensor& input,
    const at::Tensor& rois,
    double spatial_scale,
    int64_t pooled_height,
    int64_t pooled_width,
    int64_t sampling_ratio,
    bool aligned) {
  c10::impl::ExcludeDispatchKeyGuard no_autocast(
      c10::DispatchKey::AutocastPrivateUse1);
  return roi_align(
             at::autocast::cached_cast(at::kFloat, input),
             at::autocast::cached_cast(at::kFloat, rois),
             spatial_scale,
             pooled_height,
             pooled_width,
             sampling_ratio,
             aligned)
      .to(input.scalar_type());
}

} // namespace

TORCH_LIBRARY_IMPL(torchvision, PrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_forward_kernel));
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::_roi_align_backward"),
      TORCH_FN(roi_align_backward_kernel));
}

TORCH_LIBRARY_IMPL(torchvision, AutocastPrivateUse1, m) {
  m.impl(
      TORCH_SELECTIVE_NAME("torchvision::roi_align"),
      TORCH_FN(roi_align_autocast));
}

} // namespace ops
} // namespace vision
