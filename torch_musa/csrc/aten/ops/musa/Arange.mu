#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include <musa_fp16.h>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/RangeFactories.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <algorithm>

namespace at {
namespace native {
namespace {

template <typename T>
__global__ void arange_kernel(T* out_ptr, T start, T step, int nthreads) {
  typedef typename at::musa::Dtype<T>::Vec4 vec4;
  int global_stride = blockDim.x * gridDim.x;
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  int vec_idx = idx * 4;
  vec4* out_vec4_ptr = (vec4*)(out_ptr);
  vec4 vec_o;

  while (vec_idx < nthreads) {
    if (vec_idx + 4 <= nthreads) {
      vec_o.x = start + static_cast<T>(vec_idx + 0) * step;
      vec_o.y = start + static_cast<T>(vec_idx + 1) * step;
      vec_o.z = start + static_cast<T>(vec_idx + 2) * step;
      vec_o.w = start + static_cast<T>(vec_idx + 3) * step;

      out_vec4_ptr[idx] = vec_o;

    } else {
      for (int i = vec_idx; i < nthreads; ++i) {
        out_ptr[i] = start + static_cast<T>(i) * step;
      }
    }
    idx += global_stride;
    vec_idx += global_stride * 4;
  }
}

void CheckParams(double start, double end, double step, Tensor& out) {
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Float ||
          out.scalar_type() == at::ScalarType::Int ||
          out.scalar_type() == at::ScalarType::Long ||
          out.scalar_type() == at::ScalarType::Half ||
          out.scalar_type() == at::ScalarType::BFloat16,
      "unsupported data type ",
      out.scalar_type());
  TORCH_CHECK(step > 0 || step < 0, "step mustn't be zero.");
  TORCH_CHECK(
      (step > 0 && start <= end) || (step < 0 && end <= start),
      "upper and lower bound inconsistent with step");
  double size_d = std::ceil((end - start) / step);
  int64_t size = static_cast<int64_t>(size_d);
  if (size != out.numel()) {
    if (out.numel() > 0) {
      TORCH_WARN(
          "The number of elements in the out tensor of shape ",
          out.sizes(),
          "is ",
          out.numel(),
          " which does not match the computed number of elements ",
          size);
    }
    out.resize_({size});
  }
  TORCH_CHECK(
      size == out.numel(),
      "The number of out tensor elements mismatches with induced number: ",
      out.numel(),
      " vs ",
      size_d);
}

void launch(
    const Tensor& out,
    const Scalar& start,
    const Scalar& step,
    const int nthreads,
    const int nr_block,
    const int thread_per_block) {
  auto stream = c10::musa::getCurrentMUSAStream();
  switch (out.scalar_type()) {
    case at::ScalarType::Float:
      arange_kernel<float><<<nr_block, thread_per_block, 0, stream>>>(
          static_cast<float*>(out.data_ptr()),
          start.toFloat(),
          step.toFloat(),
          nthreads);
      break;
    case at::ScalarType::Half:
      arange_kernel<float16_t><<<nr_block, thread_per_block, 0, stream>>>(
          static_cast<float16_t*>(out.data_ptr()),
          static_cast<float16_t>(start.toFloat()),
          static_cast<float16_t>(step.toFloat()),
          nthreads);
      break;
    case at::ScalarType::BFloat16:
      arange_kernel<bfloat16_t><<<nr_block, thread_per_block, 0, stream>>>(
          static_cast<bfloat16_t*>(out.data_ptr()),
          static_cast<bfloat16_t>(start.toFloat()),
          static_cast<bfloat16_t>(step.toFloat()),
          nthreads);
      break;
    case at::ScalarType::Int:
      arange_kernel<int32_t><<<nr_block, thread_per_block, 0, stream>>>(
          static_cast<int32_t*>(out.data_ptr()),
          start.toInt(),
          step.toInt(),
          nthreads);
      break;
    case at::ScalarType::Long:
      arange_kernel<int64_t><<<nr_block, thread_per_block, 0, stream>>>(
          static_cast<int64_t*>(out.data_ptr()),
          start.toLong(),
          step.toLong(),
          nthreads);
      break;
    default:
      TORCH_CHECK(false, "unsupported data type (", out.scalar_type(), ")");
  }
}
} // namespace

void ArangeRun(
    const Scalar& start,
    const Scalar& end,
    const Scalar& step,
    Tensor& out) {
  if (out.scalar_type() == at::ScalarType::Float ||
      out.scalar_type() == at::ScalarType::Half ||
      out.scalar_type() == at::ScalarType::BFloat16) {
    CheckParams(start.toDouble(), end.toDouble(), step.toDouble(), out);
  } else {
    CheckParams(
        static_cast<double>(start.toLong()),
        static_cast<double>(end.toLong()),
        static_cast<double>(step.toLong()),
        out);
  }
  int out_numel = out.numel();

  // device info
  musaDeviceProp device_prop;
  at::musa::muHandle& h = GetMudnnHandle();
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  int mp_num = device_prop.multiProcessorCount;

  const int block_size = 1024;
  int block_num =
      at::musa::ceil_div(at::musa::ceil_div(out_numel, 4), block_size);
#if MUSA_ARCH > 210
  block_num = block_num;
#else
  block_num = std::min(block_num, mp_num);
#endif

  launch(out, start, step, out_numel, block_size, block_num);
}

REGISTER_MUSA_DISPATCH(arange_start_out_stub, &ArangeRun);

} // namespace native
} // namespace at
