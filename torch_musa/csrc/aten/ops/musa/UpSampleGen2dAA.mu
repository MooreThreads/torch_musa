// upsample with antialias
// Copied from UpSampleBilinear2d.cu from PyTorch cuda with only minor
// modifications (the calculations in shm will use acc dtype for higher
// precision)

// clang-format off
// Adapted from interp.cpp from Caffe util by Pauline Luc
// Originally developed by George Papandreou
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/AccumulateType.h>
#include <ATen/ceil_div.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/native/musa/UpSample.muh>
#include <ATen/native/musa/KernelUtils.muh>
#include <ATen/musa/detail/KernelUtils.h>
#include <ATen/native/musa/LaunchUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_upsample_bicubic2d_aa_backward_native.h>
#include <ATen/ops/_upsample_bicubic2d_aa_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAMarcos.muh"


namespace at::musa {

namespace upsample_antialias{

template <typename accscalar_t, typename interp_filter_t>
__device__ __forceinline__ static void _compute_weights(
    accscalar_t* wt_ptr,
    const accscalar_t scale,
    int interp_size,
    const interp_filter_t& interp_filter,
    accscalar_t xmin_m_center,
    int xsize) {

  accscalar_t invscale = (scale >= 1.0) ? 1.0 / scale : 1.0;
  accscalar_t total_w = 0.0;
  int j = 0;
  for (j = 0; j < xsize; j++) {
    accscalar_t w = interp_filter((j + xmin_m_center + static_cast<accscalar_t>(0.5)) * invscale);
    wt_ptr[j] = static_cast<accscalar_t>(w);
    total_w += w;
  }
  for (j = 0; j < xsize; j++) {
    if (total_w != 0.0) {
      wt_ptr[j] /= total_w;
    }
  }
  for (; j < interp_size; j++) {
    wt_ptr[j] = static_cast<accscalar_t>(0.0);
  }
}

template <typename scalar_t, typename accscalar_t>
__device__ __forceinline__ static accscalar_t interpolate_aa_single_dim(
    const scalar_t* src,
    const accscalar_t* weights,
    int size) {
  accscalar_t t = static_cast<accscalar_t>(*src);
  accscalar_t wts = static_cast<accscalar_t>(weights[0]);
  accscalar_t output = t * wts;

  int j = 1;
  for (; j < size; j++) {
    wts = static_cast<accscalar_t>(weights[j]);
    t = static_cast<accscalar_t>(*(src + j));
    output += t * wts;
  }
  return output;
}

template <typename accscalar_t>
__device__ __forceinline__ static accscalar_t interpolate_aa_single_dim(
    const accscalar_t* src,
    const accscalar_t* weights,
    int size) {
  accscalar_t t = static_cast<accscalar_t>(*src);
  accscalar_t wts = static_cast<accscalar_t>(weights[0]);
  accscalar_t output = t * wts;

  int j = 1;
  for (; j < size; j++) {
    wts = static_cast<accscalar_t>(weights[j]);
    t = static_cast<accscalar_t>(*(src + j));
    output += t * wts;
  }
  return output;
}
}  // namespace upsample_antialias

namespace {
// using namespace at::native::upsample_antialias;  // we are under at::musa namespace

// Code for upsampling with antialias
template <typename scalar_t, typename accscalar_t, typename InterpFilter>
__global__ void upsample_gen2d_aa_out_frame(
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    const PackedTensorAccessor64<scalar_t, 4> idata,
    PackedTensorAccessor64<scalar_t, 4> odata,
    const InterpFilter & interp_filter) {

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  const int output_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int output_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (output_x >= output_width || output_y >= output_height) {
    return;
  }

  const accscalar_t half = 0.5;
  const accscalar_t support_h = static_cast<accscalar_t>(
      (height_scale >= 1.0) ? (interp_filter.size * half) * height_scale : interp_filter.size * half);
  const accscalar_t support_w = static_cast<accscalar_t>(
      (width_scale >= 1.0) ? (interp_filter.size * half) * width_scale : interp_filter.size * half);

  const int interp_height = (int)ceilf(support_h) * 2 + 1;
  const int interp_width = (int)ceilf(support_w) * 2 + 1;

  // Setup weights and a buffer using shared memory
  // we are using accscalar_t type, which differs from CUDA's implementation
  extern __shared__ char smem[];
  accscalar_t* wx = reinterpret_cast<accscalar_t*>(smem) + interp_width * threadIdx.x;
  accscalar_t* wy = reinterpret_cast<accscalar_t*>(smem) + interp_width * blockDim.x + interp_height * threadIdx.y;
  const int offset = interp_width * blockDim.x + interp_height * blockDim.y;
  accscalar_t *buffer2 = reinterpret_cast<accscalar_t*>(smem) + offset + \
      interp_height * (threadIdx.x + threadIdx.y * blockDim.x);

  // // Compute weights and kernel spans
  int xmin, xsize, ymin, ysize;
  accscalar_t xcenter, ycenter;
  native::upsample_antialias::_compute_weights_span(
      output_x, input_width, width_scale, support_w, xmin, xsize, xcenter);
  native::upsample_antialias::_compute_weights_span(
      output_y, input_height, height_scale, support_h, ymin, ysize, ycenter);

  if (threadIdx.y == 0)
  {
    // All threadIdx.y have the same wx weights
    upsample_antialias::_compute_weights<accscalar_t>(
        wx,
        width_scale,
        interp_width,
        interp_filter,
        xmin - xcenter,
        xsize);
  }

  if (threadIdx.x == 0)
  {
    // All threadIdx.x have the same wy weights
    upsample_antialias::_compute_weights<accscalar_t>(
        wy,
        height_scale,
        interp_height,
        interp_filter,
        ymin - ycenter,
        ysize);
  }

  __SYNCTHREADS;

  const scalar_t * buffer1;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      // interpolate on y-axis for ymin to ymin + ysize
      for (int y = 0; y < ysize; y++) {
        buffer1 = &(idata[n][c][ymin + y][xmin]);
        buffer2[y] = static_cast<scalar_t>(
            upsample_antialias::interpolate_aa_single_dim<scalar_t, accscalar_t>(
                buffer1, wx, xsize));
      }
      // accumulating on shm
      odata[n][c][output_y][output_x] = static_cast<scalar_t>(
          upsample_antialias::interpolate_aa_single_dim<accscalar_t>(
              buffer2, wy, ysize));
    }
  }
}

// Code for upsampling with antialias
template <typename scalar_t, typename accscalar_t, typename InterpFilter>
__global__ void upsample_gen2d_aa_backward_out_frame(
    const accscalar_t height_scale,
    const accscalar_t width_scale,
    PackedTensorAccessor64<scalar_t, 4> idata,
    const PackedTensorAccessor64<scalar_t, 4> odata,
    const InterpFilter & interp_filter) {

  const int batchsize = idata.size(0);
  const int channels = idata.size(1);
  const int input_height = idata.size(2);
  const int input_width = idata.size(3);
  const int output_height = odata.size(2);
  const int output_width = odata.size(3);

  const int output_x = threadIdx.x + blockIdx.x * blockDim.x;
  const int output_y = threadIdx.y + blockIdx.y * blockDim.y;

  if (output_x >= output_width || output_y >= output_height) {
    return;
  }

  // special case: output just copy
  if (input_height == output_height && input_width == output_width) {
    for (int n = 0; n < batchsize; n++) {
      for (int c = 0; c < channels; c++) {
        const scalar_t val = odata[n][c][output_y][output_x];
        idata[n][c][output_y][output_x] = val;
      }
    }
    return;
  }

  const accscalar_t support_h = static_cast<accscalar_t>(
      (height_scale >= 1.0) ? (interp_filter.size * 0.5) * height_scale
                            : interp_filter.size * 0.5);
  const accscalar_t support_w = static_cast<accscalar_t>(
      (width_scale >= 1.0) ? (interp_filter.size * 0.5) * width_scale
                           : interp_filter.size * 0.5);

  const int interp_height = (int)ceilf(support_h) * 2 + 1;
  const int interp_width = (int)ceilf(support_w) * 2 + 1;

  // Setup weights using shared memory
  extern __shared__ char smem[];
  accscalar_t* wx = reinterpret_cast<accscalar_t*>(smem) + interp_width * threadIdx.x;
  accscalar_t* wy = reinterpret_cast<accscalar_t*>(smem) + interp_width * blockDim.x + interp_height * threadIdx.y;

  // Compute weights and kernel spans
  int xmin, xsize, ymin, ysize;
  accscalar_t xcenter, ycenter;
  native::upsample_antialias::_compute_weights_span(
      output_x, input_width, width_scale, support_w, xmin, xsize, xcenter);
  native::upsample_antialias::_compute_weights_span(
      output_y, input_height, height_scale, support_h, ymin, ysize, ycenter);

  if (threadIdx.y == 0)
  {
    // All threadIdx.y have the same wx weights
    upsample_antialias::_compute_weights<accscalar_t>(
        wx,
        width_scale,
        interp_width,
        interp_filter,
        xmin - xcenter,
        xsize);
  }

  if (threadIdx.x == 0)
  {
    // All threadIdx.x have the same wy weights
    upsample_antialias::_compute_weights<accscalar_t>(
        wy,
        height_scale,
        interp_height,
        interp_filter,
        ymin - ycenter,
        ysize);
  }

  __SYNCTHREADS;

  for (int n = 0; n < batchsize; n++) {
    for (int c = 0; c < channels; c++) {
      scalar_t out_value = odata[n][c][output_y][output_x];
      for (int y = 0; y < ysize; y++) {
        for (int x = 0; x < xsize; x++) {
          native::upsample_increment_value_bounded<scalar_t, accscalar_t>(
              idata,
              n,
              c,
              input_height,
              input_width,
              ymin + y,
              xmin + x,
              wx[x] * wy[y] * out_value);
        }
      }
    }
  }
}

// In the code below interp_filter_t distinguishes between bilinear and bicubic interpolations
// InterpFilter as BilinearFilterFunctor <--> bilinear
// InterpFilter as BicubicFilterFunctor <--> bicubic
template<typename InterpFilter>
static void upsample_gen2d_aa_out_musa_template(
    const Tensor& output,
    const Tensor& input_,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {
  TensorArg input_arg{input_, "input_", 1}, output_arg{output, "output", 2};
  checkAllSameGPU("upsample_gen2d_aa_out_musa", {input_arg, output_arg});

  // TODO: remove this when the musa kernel is updated to support the channels_last memory format.
  // This is a temporary hack to prevent a silence correctness issue when calling this kernel
  // with tensors in channels_last format.
  auto output_c = output.is_contiguous() ? output : at::empty(output.sizes(), output.options());
  auto input = input_.contiguous();

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input.size(2);
  int input_width = input.size(3);

  musaStream_t stream = at::musa::getCurrentMUSAStream();
  size_t sharedMemPerBlock = at::musa::getCurrentDeviceProperties()->sharedMemPerBlock;
  int* maxThreadsDim = at::musa::getCurrentDeviceProperties()->maxThreadsDim;
  int maxThreadsPerBlock = std::min(at::musa::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
  int* maxGridSize = at::musa::getCurrentDeviceProperties()->maxGridSize;
  int block_x = std::min<int>(maxThreadsDim[0], at::musa::warp_size());
  int grid_x = std::min<int>(maxGridSize[0], ceil_div(output_width, block_x));

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      input.scalar_type(), "upsample_bilinear2d_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = input.packed_accessor64<scalar_t, 4>();
        auto odata = output_c.packed_accessor64<scalar_t, 4>();

        const accscalar_t height_scale = native::area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = native::area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        // We are using shared memory to store weights wx, wy and a buffer of size wy unique per thread
        // Let's compute block_y size depending on given height_scale and width_scale
        // We have the following relationship:
        // shmem_size / sizeofdtype =
        //  interp_width * block_x +   <-- wx allocation
        //  interp_height * block_y * (block_x + 1)   <-- wy and buffer allocations

        auto interp_filter = InterpFilter();
        const int interp_height = 1 + 2 * (int)ceilf(
            (height_scale >= 1.0) ? interp_filter.size * 0.5 * height_scale : interp_filter.size * 0.5);
        const int interp_width = 1 + 2 * (int)ceilf(
            (width_scale >= 1.0) ? interp_filter.size * 0.5 * width_scale : interp_filter.size * 0.5);

        int numer = sharedMemPerBlock * 1.0 / sizeof(accscalar_t) - interp_width * block_x;
        int denom = interp_height * (block_x + 1);
        int block_y = native::lastPow2((unsigned int) (numer / denom));
        block_y = std::min<int>(maxThreadsPerBlock / block_x, block_y);
        const dim3 block(block_x, block_y);

        int grid_y = std::min<int>(maxGridSize[1], ceil_div(output_height, block_y));
        const dim3 grid(grid_x, grid_y);

        // Compute actual size of required shared memory and verify if we can allocate it
        // - wx and wy size:
        size_t weights_per_block = interp_width * block_x + interp_height * block_y;
        // - buffer size:
        weights_per_block += interp_height * block_y * block_x;
        size_t shmem_size = weights_per_block * sizeof(accscalar_t);
        TORCH_CHECK(
            shmem_size <= sharedMemPerBlock,
            "Provided interpolation parameters can not be handled with current algorithm implementation. ",
            "Please reduce the scale factor. Too much shared memory required: ",
            shmem_size, " vs ", sharedMemPerBlock);

        upsample_gen2d_aa_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               shmem_size,
               stream>>>(height_scale, width_scale, idata, odata, interp_filter);
        C10_MUSA_KERNEL_LAUNCH_CHECK();
      });

  if (!output.is_contiguous()) {
      output.copy_(output_c);
  }
}

// In the code below interp_filter_t distinguishes between bilinear and bicubic interpolations
// InterpFilter as BilinearFilterFunctor <--> bilinear
// InterpFilter as BicubicFilterFunctor <--> bicubic
template<typename InterpFilter>
static void upsample_gen2d_aa_backward_out_musa_template(
    const Tensor& grad_input,
    const Tensor& grad_output_,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w) {

  // Inspired from UpSampleBicubic2d.mu::upsample_bicubic2d_backward_out_musa_template
  TensorArg grad_input_arg{grad_input, "grad_input", 1},
      grad_output_arg{grad_output_, "grad_output_", 2};
  checkAllSameGPU(
      "upsample_gen2d_backward_out_musa", {grad_output_arg, grad_input_arg});

  int output_height = output_size[0];
  int output_width = output_size[1];

  int input_height = input_size[2];
  int input_width = input_size[3];

  Tensor grad_output = grad_output_.contiguous();

  grad_input.zero_();

  const int num_threads = std::min(at::musa::getCurrentDeviceProperties()->maxThreadsPerBlock, 256);
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  int* maxThreadsDim = at::musa::getCurrentDeviceProperties()->maxThreadsDim;
  int block_x = std::min<int>(maxThreadsDim[0], at::musa::warp_size());
  int block_y = std::min<int>(maxThreadsDim[1], num_threads / block_x);
  const dim3 block(block_x, block_y);

  int* maxGridSize = at::musa::getCurrentDeviceProperties()->maxGridSize;
  int grid_x = std::min<int>(maxGridSize[0], ceil_div(output_width, block_x));
  int grid_y = std::min<int>(maxGridSize[1], ceil_div(output_height, block_y));
  const dim3 grid(grid_x, grid_y);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      grad_output.scalar_type(), "upsample_gen2d_backward_out_frame", [&] {
        using accscalar_t = at::acc_type<scalar_t, true>;

        auto idata = grad_input.packed_accessor64<scalar_t, 4>();
        auto odata = grad_output.packed_accessor64<scalar_t, 4>();

        const accscalar_t height_scale = native::area_pixel_compute_scale<accscalar_t>(
            input_height, output_height, align_corners, scales_h);
        const accscalar_t width_scale = native::area_pixel_compute_scale<accscalar_t>(
            input_width, output_width, align_corners, scales_w);

        auto interp_filter = InterpFilter();
        const int interp_height = 1 + 2 * (int)ceilf(
            (height_scale >= 1.0) ? interp_filter.size * 0.5 * height_scale : interp_filter.size * 0.5);
        const int interp_width = 1 + 2 * (int)ceilf(
            (width_scale >= 1.0) ? interp_filter.size * 0.5 * width_scale : interp_filter.size * 0.5);

        size_t weights_per_block = interp_width * block_x + interp_height * block_y;
        size_t shmem_size = weights_per_block * sizeof(accscalar_t);
        size_t sharedMemPerBlock = at::musa::getCurrentDeviceProperties()->sharedMemPerBlock;
        TORCH_CHECK(
            shmem_size <= sharedMemPerBlock,
            "Provided interpolation parameters can not be handled with current algorithm implementation. ",
            "Please reduce the scale factor. Too much shared memory required: ",
            shmem_size, " vs ", sharedMemPerBlock);

        upsample_gen2d_aa_backward_out_frame<scalar_t, accscalar_t>
            <<<grid,
               block,
               shmem_size,
               stream>>>(height_scale, width_scale, idata, odata, interp_filter);
        C10_MUSA_KERNEL_LAUNCH_CHECK();
      });
}

} // namespace


TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_out_musa) (
    const Tensor& input,
    IntArrayRef output_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& output) {
  upsample_gen2d_aa_out_musa_template<native::upsample_antialias::BicubicFilterFunctor>(
      output, input, output_size, align_corners, scales_h, scales_w);
}

TORCH_IMPL_FUNC(_upsample_bicubic2d_aa_backward_out_musa) (
    const Tensor& grad_output,
    IntArrayRef output_size,
    IntArrayRef input_size,
    bool align_corners,
    c10::optional<double> scales_h,
    c10::optional<double> scales_w,
    const Tensor& grad_input) {
  // See Note [Writing Nondeterministic Operations]
  // Nondeterministic because of atomicAdd usage
  globalContext().alertNotDeterministic("upsample_bicubic2d_aa_backward_out_musa");
  upsample_gen2d_aa_backward_out_musa_template<native::upsample_antialias::BicubicFilterFunctor>(
      grad_input, grad_output, output_size, input_size, align_corners, scales_h, scales_w);
}

} // namespace at::musa
