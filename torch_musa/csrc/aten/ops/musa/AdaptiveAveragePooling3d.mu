#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/Utils.h>
#include <ATen/core/Tensor.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/adaptive_avg_pool3d_backward_native.h>
#include <ATen/ops/adaptive_avg_pool3d_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <ATen/native/AdaptivePooling.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <limits>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {

namespace {

inline bool CanUse32BitIndexing(Tensor t) {
  int64_t max_value = std::numeric_limits<int32_t>::max();
  return t.numel() < max_value;
}

// In forward, pass (i, pool_size, input_size)
// In backward, pass (i, input_size, pool_size)
// keep consistent with torch
__device__ __forceinline__ uint32_t
AdaptPoolStartIndex(uint32_t a, uint32_t b, uint32_t c) {
  return (a / b) * c + ((a % b) * c) / b;
}

__device__ __forceinline__ uint32_t
AdaptPoolEndIndex(uint32_t a, uint32_t b, uint32_t c) {
  return 1 + ((a + 1) * c - 1) / b;
}

struct FastDivModPooling3D {
 public:
  at::musa::FastDivmod channel;
  at::musa::FastDivmod depth;
  at::musa::FastDivmod height;
  at::musa::FastDivmod width;

  explicit __host__ __device__ FastDivModPooling3D(
      const int channels,
      const int output_depth,
      const int output_height,
      const int output_width) {
    channel = at::musa::FastDivmod(channels);
    depth = at::musa::FastDivmod(output_depth);
    height = at::musa::FastDivmod(output_height);
    width = at::musa::FastDivmod(output_width);
  }
};

template <
    bool IS_NCDHW = true,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
__global__ void AdaptiveAvgPool3DForwardKernel(
    const int nelems,
    scalar_t* output,
    const scalar_t* input,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    FastDivModPooling3D fdm_out) {
  index_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t global_stride = blockDim.x * gridDim.x;
  while (work_idx < nelems) {
    uint32_t w_offset, h_offset, d_offset, c_offset, batch_idx;

    if constexpr (IS_NCDHW) {
      // !channels_last
      uint32_t pw_q, ph_q, pd_q;
      fdm_out.width(pw_q, w_offset, (uint32_t)work_idx);
      fdm_out.height(ph_q, h_offset, pw_q);
      fdm_out.depth(pd_q, d_offset, ph_q);
      fdm_out.channel(batch_idx, c_offset, pd_q);
    } else {
      uint32_t pc_q, pw_q, ph_q;
      fdm_out.channel(pc_q, c_offset, (uint32_t)work_idx);
      fdm_out.width(pw_q, w_offset, pc_q);
      fdm_out.height(ph_q, h_offset, pw_q);
      fdm_out.depth(batch_idx, d_offset, ph_q);
    }
    uint32_t pwstart = AdaptPoolStartIndex(w_offset, output_width, input_width);
    uint32_t pwend = AdaptPoolEndIndex(w_offset, output_width, input_width);
    uint32_t phstart =
        AdaptPoolStartIndex(h_offset, output_height, input_height);
    uint32_t phend = AdaptPoolEndIndex(h_offset, output_height, input_height);
    uint32_t pdstart = AdaptPoolStartIndex(d_offset, output_depth, input_depth);
    uint32_t pdend = AdaptPoolEndIndex(d_offset, output_depth, input_depth);

    if constexpr (IS_NCDHW) {
      accscalar_t sum = static_cast<accscalar_t>(0.0);
      index_t offset = ((index_t)batch_idx * channels + c_offset) *
          input_depth * input_height * input_width;
      const scalar_t* cur_input_data = input + offset;

      for (index_t pd = pdstart; pd < pdend; pd++) {
        for (index_t ph = phstart; ph < phend; ph++) {
          for (index_t pw = pwstart; pw < pwend; pw++) {
            index_t idx = (pd * input_height + ph) * input_width + pw;
            scalar_t val = cur_input_data[idx];
            sum += static_cast<accscalar_t>(val);
          }
        }
      }
      // averaging on sum
      accscalar_t out_val =
          sum /
          static_cast<accscalar_t>(
              (pdend - pdstart) * (phend - phstart) * (pwend - pwstart));
      output[work_idx] = static_cast<scalar_t>(out_val);
    } else {
      // nothing different from 'IS_NCDHW' branch except the innermost loop and
      // the calculation of base offset currently
      // TODO(mt-ai): advanced implementation by vectorization LD/ST along
      // channel dimension
      accscalar_t sum = static_cast<accscalar_t>(0.0);
      index_t offset = (index_t)batch_idx * input_width * input_height *
              input_depth * channels +
          c_offset;
      const scalar_t* cur_input_data = input + offset;

      for (index_t pd = pdstart; pd < pdend; pd++) {
        for (index_t ph = phstart; ph < phend; ph++) {
          for (index_t pw = pwstart; pw < pwend; pw++) {
            index_t idx =
                ((pd * input_height + ph) * input_width + pw) * channels;
            scalar_t val = cur_input_data[idx];
            sum += static_cast<accscalar_t>(val);
          }
        }
      }
      accscalar_t out_val =
          sum /
          static_cast<accscalar_t>(
              (pdend - pdstart) * (phend - phstart) * (pwend - pwstart));
      output[work_idx] = static_cast<scalar_t>(out_val);
    }
    work_idx += global_stride;
  }
}

template <
    bool IS_NCDHW = true,
    typename scalar_t,
    typename accscalar_t,
    typename index_t>
__global__ void AdaptiveAvgPool3DBackwardKernel(
    const int nelems,
    scalar_t* grad_input,
    const scalar_t* grad_output,
    const int channels,
    const int input_depth,
    const int input_height,
    const int input_width,
    const int output_depth,
    const int output_height,
    const int output_width,
    FastDivModPooling3D fdm_out) {
  // already ensure contigous inputs
  index_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
  index_t global_stride = blockDim.x * gridDim.x;
  while (work_idx < nelems) {
    uint32_t w_offset, h_offset, d_offset, c_offset, batch_idx;

    if constexpr (IS_NCDHW) {
      // work_idx is global index
      uint32_t pw_q, ph_q, pd_q;
      fdm_out.width(pw_q, w_offset, (uint32_t)work_idx);
      fdm_out.height(ph_q, h_offset, pw_q);
      fdm_out.depth(pd_q, d_offset, ph_q);
      fdm_out.channel(batch_idx, c_offset, pd_q);
    } else {
      uint32_t pc_q, pw_q, ph_q;
      fdm_out.channel(pc_q, c_offset, (uint32_t)work_idx);
      fdm_out.width(pw_q, w_offset, pc_q);
      fdm_out.height(ph_q, h_offset, pw_q);
      fdm_out.depth(batch_idx, d_offset, ph_q);
    }
    uint32_t pwstart = AdaptPoolStartIndex(w_offset, input_width, output_width);
    uint32_t pwend = AdaptPoolEndIndex(w_offset, input_width, output_width);
    uint32_t phstart =
        AdaptPoolStartIndex(h_offset, input_height, output_height);
    uint32_t phend = AdaptPoolEndIndex(h_offset, input_height, output_height);
    uint32_t pdstart = AdaptPoolStartIndex(d_offset, input_depth, output_depth);
    uint32_t pdend = AdaptPoolEndIndex(d_offset, input_depth, output_depth);

    if constexpr (IS_NCDHW) {
      accscalar_t accu_grad = static_cast<accscalar_t>(0.0);
      index_t base_offset = ((index_t)batch_idx * channels + c_offset) *
          output_depth * output_height * output_width;

      for (uint32_t pd = pdstart; pd < pdend; pd++) {
        uint32_t kD = AdaptPoolEndIndex(pd, output_depth, input_depth) -
            AdaptPoolStartIndex(pd, output_depth, input_depth);
        for (uint32_t ph = phstart; ph < phend; ph++) {
          uint32_t kH = AdaptPoolEndIndex(ph, output_height, input_height) -
              AdaptPoolStartIndex(ph, output_height, input_height);
          for (uint32_t pw = pwstart; pw < pwend; pw++) {
            uint32_t kW = AdaptPoolEndIndex(pw, output_width, input_width) -
                AdaptPoolStartIndex(pw, output_width, input_width);
            index_t cur_offset =
                base_offset + (pd * output_height + ph) * output_width + pw;
            const accscalar_t div_factor = kD * kH * kW;
            accu_grad += (accscalar_t)grad_output[cur_offset] / div_factor;
          }
        }
      }
      grad_input[work_idx] = (scalar_t)accu_grad;
    } else {
      // channels_last
      accscalar_t accu_grad = static_cast<accscalar_t>(0.0);
      index_t base_offset = (index_t)batch_idx * channels * output_depth *
              output_height * output_width +
          c_offset;

      for (uint32_t pd = pdstart; pd < pdend; pd++) {
        uint32_t kD = AdaptPoolEndIndex(pd, output_depth, input_depth) -
            AdaptPoolStartIndex(pd, output_depth, input_depth);
        for (uint32_t ph = phstart; ph < phend; ph++) {
          uint32_t kH = AdaptPoolEndIndex(ph, output_height, input_height) -
              AdaptPoolStartIndex(ph, output_height, input_height);
          for (uint32_t pw = pwstart; pw < pwend; pw++) {
            uint32_t kW = AdaptPoolEndIndex(pw, output_width, input_width) -
                AdaptPoolStartIndex(pw, output_width, input_width);
            index_t cur_offset = base_offset +
                ((pd * output_height + ph) * output_width + pw) * channels;
            const accscalar_t div_factor = kD * kH * kW;
            accu_grad += (accscalar_t)grad_output[cur_offset] / div_factor;
          }
        }
      }
      grad_input[work_idx] = (scalar_t)accu_grad;
    }
    work_idx += global_stride;
  }
}

void AdaptiveAvgPool3DOutMUSATemplate(
    Tensor& output,
    const Tensor& input_,
    IntArrayRef& output_size) {
  TensorArg output_arg{output, "output", 1};
  TensorArg input_arg{input_, "input_", 2};

  checkAllSameGPU("adaptive_avg_pool3d_musa", {output_arg, input_arg});

  for (int64_t i = 1; i < input_.ndimension(); i++) {
    TORCH_CHECK(
        input_.size(i) > 0,
        "adaptive_avg_pool3d_musa(): Expected input to have non-zero size for non-batch dimensions, "
        "but input has sizes ",
        input_.sizes(),
        " with dimension ",
        i,
        " being empty");
  }

  TORCH_CHECK(
      (input_.ndimension() == 4 || input_.ndimension() == 5),
      "adaptive_avg_pool3d_musa(): Expected 4D or 5D tensor, but got ",
      input_.sizes());

  // the jit sometimes passes output_size.size() == 1
  TORCH_CHECK(
      output_size.size() == 1 || output_size.size() == 3,
      "adaptive_avg_pool3d: internal error: output_size.size() must be 1 or 3");

  int batch_size, channels, input_depth, input_height, input_width;

  int output_depth = output_size[0];
  int output_height = output_size[1];
  int output_width = output_size[2];

  if (input_.ndimension() == 4) {
    channels = input_.size(0);
    input_depth = input_.size(1);
    input_height = input_.size(2);
    input_width = input_.size(3);

    output.resize_({channels, output_depth, output_height, output_width});
  } else {
    batch_size = input_.size(0);
    channels = input_.size(1);
    input_depth = input_.size(2);
    input_height = input_.size(3);
    input_width = input_.size(4);

    output.resize_(
        {batch_size, channels, output_depth, output_height, output_width});
  }

  if (output.numel() == 0) {
    return;
  }

  const bool can_use_i32_indexing =
      CanUse32BitIndexing(output) && CanUse32BitIndexing(input_);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf, kBFloat16, input_.scalar_type(), "adaptive_avg_pool3d_musa", [&] {
        AT_DISPATCH_INDEX_TYPES(
            can_use_i32_indexing ? ScalarType::Int : ScalarType::Long,
            "adaptive_avg_pool3d_musa_index",
            [&] {
              using accscalar_t = at::acc_type<scalar_t, true>;

              int nelems = batch_size * channels * output_depth *
                  output_height * output_width;
              dim3 block(1024, 1, 1);
#if TORCH_MUSA_ARCH > 210
              dim3 grid((nelems + 1024 - 1) / 1024, 1, 1);
#else
              musaDeviceProp* prop = at::musa::getCurrentDeviceProperties();
              dim3 grid(prop->multiProcessorCount, 1, 1);
#endif
              auto memory_format = output.suggest_memory_format();
              Tensor input_tmp =
                  at::musa::FormatContiguous(input_, memory_format);
              scalar_t* input_data = input_tmp.data_ptr<scalar_t>();
              scalar_t* output_data = output.data_ptr<scalar_t>();

              auto stream = at::musa::getCurrentMUSAStream();
              auto fdm_out = FastDivModPooling3D(
                  channels, output_depth, output_height, output_width);

              if (memory_format == at::MemoryFormat::Contiguous) {
                AdaptiveAvgPool3DForwardKernel<
                    true,
                    scalar_t,
                    accscalar_t,
                    index_t><<<grid, block, 0, stream>>>(
                    nelems,
                    output_data,
                    input_data,
                    channels,
                    input_depth,
                    input_height,
                    input_width,
                    output_depth,
                    output_height,
                    output_width,
                    fdm_out);
              } else {
                AdaptiveAvgPool3DForwardKernel<
                    false,
                    scalar_t,
                    accscalar_t,
                    index_t><<<grid, block, 0, stream>>>(
                    nelems,
                    output_data,
                    input_data,
                    channels,
                    input_depth,
                    input_height,
                    input_width,
                    output_depth,
                    output_height,
                    output_width,
                    fdm_out);
              }
              C10_MUSA_KERNEL_LAUNCH_CHECK();
            });
      });
}

void AdaptiveAvgPool3DBackwardOutTemplate(
    Tensor& grad_input,
    const Tensor& grad_output,
    const Tensor& input) {
  TensorArg grad_input_arg{grad_input, "grad_input", 1};
  TensorArg grad_output_arg{grad_output, "gradOutput_", 2};
  TensorArg input_arg{input, "input", 3};

  at::native::adaptive_pool_empty_output_check(
      grad_output, "AdaptiveAvgPool3DBackward");

  checkAllSameGPU(
      "AdaptiveAvgPool3DBackward",
      {grad_input_arg, grad_output_arg, input_arg});

  grad_input.resize_as_(input);
  if (grad_input.numel() == 0) {
    return;
  }

  grad_input.zero_();

  // [N], C, D, H, W

  int batch_size, channels, input_depth, input_height, input_width;
  int output_depth, output_height, output_width;

  if (input.ndimension() == 4) {
    batch_size = 1;
    channels = input.size(0);
    input_depth = input.size(1);
    input_height = input.size(2);
    input_width = input.size(3);

    output_depth = grad_output.size(1);
    output_height = grad_output.size(2);
    output_width = grad_output.size(3);
  } else {
    batch_size = input.size(0);
    channels = input.size(1);
    input_depth = input.size(2);
    input_height = input.size(3);
    input_width = input.size(4);

    output_depth = grad_output.size(2);
    output_height = grad_output.size(3);
    output_width = grad_output.size(4);
  }

  const bool can_use_i32_indexing =
      CanUse32BitIndexing(grad_input) && CanUse32BitIndexing(grad_output);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      kHalf,
      kBFloat16,
      input.scalar_type(),
      "adaptive_avg_pool3d_backward_musa",
      [&] {
        AT_DISPATCH_INDEX_TYPES(
            can_use_i32_indexing ? ScalarType::Int : ScalarType::Long,
            "adaptive_avg_pool3d_backward_musa_index",
            [&] {
              using accscalar_t = at::acc_type<scalar_t, true>;

              int nelems = batch_size * channels * input_depth * input_height *
                  input_width;
              dim3 block(1024, 1, 1);
#if TORCH_MUSA_ARCH > 210
              dim3 grid((nelems + 1024 - 1) / 1024, 1, 1);
#else
              musaDeviceProp* prop = at::musa::getCurrentDeviceProperties();
              dim3 grid(prop->multiProcessorCount, 1, 1);
#endif
              auto memory_format = grad_input.suggest_memory_format();
              Tensor grad_output_tmp =
                  FormatContiguous(grad_output, memory_format);
              scalar_t* grad_input_data = grad_input.data_ptr<scalar_t>();
              scalar_t* grad_output_data = grad_output_tmp.data_ptr<scalar_t>();

              auto stream = at::musa::getCurrentMUSAStream();
              auto fdm_out = FastDivModPooling3D(
                  channels, input_depth, input_height, input_width);

              if (memory_format == at::MemoryFormat::Contiguous) {
                AdaptiveAvgPool3DBackwardKernel<
                    true,
                    scalar_t,
                    accscalar_t,
                    index_t><<<grid, block, 0, stream>>>(
                    nelems,
                    grad_input_data,
                    grad_output_data,
                    channels,
                    input_depth,
                    input_height,
                    input_width,
                    output_depth,
                    output_height,
                    output_width,
                    fdm_out);
              } else {
                AdaptiveAvgPool3DBackwardKernel<
                    false,
                    scalar_t,
                    accscalar_t,
                    index_t><<<grid, block, 0, stream>>>(
                    nelems,
                    grad_input_data,
                    grad_output_data,
                    channels,
                    input_depth,
                    input_height,
                    input_width,
                    output_depth,
                    output_height,
                    output_width,
                    fdm_out);
              }
              C10_MUSA_KERNEL_LAUNCH_CHECK();
            });
      });
}

} // namespace

Tensor& AdaptiveAvgPool3DOutMUSA(
    const Tensor& input,
    IntArrayRef output_size,
    Tensor& output) {
  AdaptiveAvgPool3DOutMUSATemplate(output, input, output_size);
  return output;
}

Tensor AdaptiveAvgPool3DMUSA(const Tensor& input, IntArrayRef output_size) {
  auto output = at::empty({0}, input.options());
  AdaptiveAvgPool3DOutMUSATemplate(output, input, output_size);
  return output;
}

Tensor& AdaptiveAvgPool3DBackwardOutMUSA(
    const Tensor& grad_output,
    const Tensor& input,
    Tensor& grad_input) {
  AdaptiveAvgPool3DBackwardOutTemplate(grad_input, grad_output, input);
  return grad_input;
}

Tensor AdaptiveAvgPool3DBackwardMUSA(
    const Tensor& grad_output,
    const Tensor& input) {
  auto grad_input = at::zeros_like(input, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  AdaptiveAvgPool3DBackwardOutTemplate(grad_input, grad_output, input);
  return grad_input;
}

} // namespace musa
} // namespace at
