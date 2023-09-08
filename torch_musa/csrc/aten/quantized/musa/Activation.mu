#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <c10/core/QScheme.h>
#include <torch/library.h>

#ifndef AT_ACTIVATION_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <stdio.h>

const int PARALLEL_NUM = 4;

namespace at {
namespace musa {
namespace {

enum class ActMode {
  RELU,
  GELU,
  SIGMOID,
};

__device__ int64_t SigmoidKernel(
    int64_t input,
    const double scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax) {
  float value = static_cast<float>(input - zero_point) * scale;
  printf("input: %d %f, ", input, value);
  value = 1.f / (1.f + expf(-value));
  printf("compute: %f, ", value);
  int64_t qvalue = std::max<int64_t>(
      std::min<int64_t>(
          static_cast<int64_t>(std::nearbyint(value / scale) + zero_point),
          qmax),
      qmin);
  printf("qvalue: %d\n", qvalue);
  return qvalue;
}

__device__ int64_t GeLUKernel(
    int64_t input,
    const double scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax) {
  float value = static_cast<float>(input - zero_point) * scale;
  value = value * 0.5 * (1.f + erff(value / 1.4142135));
  int64_t qvalue = std::max<int64_t>(
      std::min<int64_t>(
          static_cast<int64_t>(std::nearbyint(value / scale) + zero_point),
          qmax),
      qmin);
  return qvalue;
}

__device__ int64_t ReLUKernel(
    int64_t input,
    const double scale,
    const int64_t zero_point,
    const int64_t qmax) {
  float value =
      std::max<float>(static_cast<float>(input - zero_point) * scale, 0.f);
  int64_t qvalue = std::min<int64_t>(
      static_cast<int64_t>(std::nearbyint(value / scale) + zero_point), qmax);
  return qvalue;
}

template <typename DType, ActMode mode>
__global__ void ActQuantizedKernel(
    DType* out,
    const DType* in,
    const double scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t total_num) {
  int64_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * PARALLEL_NUM;
  if (idx < total_num) {
    switch (mode) {
      case ActMode::RELU:
#pragma unroll
        for (int tid = 0; tid < PARALLEL_NUM; ++tid) {
          out[idx + tid] = ReLUKernel(
              static_cast<int64_t>(in[idx + tid]), scale, zero_point, qmax);
        }
        break;
      case ActMode::GELU:
#pragma unroll
        for (int tid = 0; tid < PARALLEL_NUM; ++tid) {
          out[idx + tid] = GeLUKernel(
              static_cast<int64_t>(in[idx + tid]),
              scale,
              zero_point,
              qmin,
              qmax);
        }
        break;
      case ActMode::SIGMOID:
#pragma unroll
        for (int tid = 0; tid < PARALLEL_NUM; ++tid) {
          out[idx + tid] = SigmoidKernel(
              static_cast<int64_t>(in[idx + tid]),
              scale,
              zero_point,
              qmin,
              qmax);
        }
        break;
      default:
        break;
    }
  }
}

template <ActMode mode>
void ActQuantizedImpl(
    at::Tensor& out,
    const at::Tensor& self,
    const double scale,
    const int64_t zero_point) {
  auto stream = c10::musa::getCurrentMUSAStream();
  int64_t numel = self.numel();
  uint32_t block_x = numel > 512 ? 1024 : 512;
  uint32_t grid_x = (numel / PARALLEL_NUM + block_x) / block_x;

  dim3 block_size{block_x, 1, 1};
  dim3 grid_size{grid_x, 1, 1};

  if (self.scalar_type() == ScalarType::QInt8) {
    constexpr int64_t qmin = std::numeric_limits<int8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<int8_t>::max();
    ActQuantizedKernel<int8_t, mode><<<grid_size, block_size, 0, stream>>>(
        static_cast<int8_t*>(out.data_ptr()),
        static_cast<int8_t*>(self.data_ptr()),
        scale,
        zero_point,
        qmin,
        qmax,
        numel);
  } else if (self.scalar_type() == ScalarType::QUInt8) {
    constexpr int64_t qmin = std::numeric_limits<uint8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<uint8_t>::max();
    ActQuantizedKernel<uint8_t, mode><<<grid_size, block_size, 0, stream>>>(
        static_cast<uint8_t*>(out.data_ptr()),
        static_cast<uint8_t*>(self.data_ptr()),
        scale,
        zero_point,
        qmin,
        qmax,
        numel);
  } else {
    TORCH_CHECK(false, "unsupported data type", self.scalar_type());
  }
  musaDeviceSynchronize();
}

template <ActMode mode>
at::Tensor ActQuantized(const at::Tensor& self) {
  const OptionalDeviceGuard device_guard(device_of(self));
  if (self.qscheme() == c10::kPerTensorAffine ||
      self.qscheme() == c10::kPerTensorSymmetric) {
    const double scale = self.q_scale();
    const int64_t zero_point = self.q_zero_point();
    at::Tensor quantized_output = at::_empty_affine_quantized(
        self.sizes(),
        at::device(at::kPrivateUse1).dtype(self.scalar_type()),
        scale,
        zero_point,
        self.suggest_memory_format());
    ActQuantizedImpl<mode>(quantized_output, self, scale, zero_point);
    return quantized_output;
  } else {
    TORCH_CHECK(
        false,
        "we currently only support per-tensor quantized relu, but got ",
        toString(self.qscheme()));
  }
}

// GeLU has difference function signature
at::Tensor ActQuantizedGeLU(
    const at::Tensor& self,
    c10::string_view approximate) {
  const OptionalDeviceGuard device_guard(device_of(self));
  (void)approximate; // avoid unused variable lint warning
  if (self.qscheme() == c10::kPerTensorAffine ||
      self.qscheme() == c10::kPerTensorSymmetric) {
    const double scale = self.q_scale();
    const int64_t zero_point = self.q_zero_point();
    at::Tensor quantized_output = at::_empty_affine_quantized(
        self.sizes(),
        at::device(at::kPrivateUse1).dtype(self.scalar_type()),
        scale,
        zero_point,
        self.suggest_memory_format());
    ActQuantizedImpl<ActMode::GELU>(quantized_output, self, scale, zero_point);
    return quantized_output;
  } else {
    TORCH_CHECK(
        false,
        "we currently only support per-tensor quantized relu, but got ",
        toString(self.qscheme()));
  }
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("relu_", TORCH_FN(at::native::relu_quantized_cuda_));
  m.impl("relu", TORCH_FN(ActQuantized<ActMode::RELU>));
  m.impl("gelu", TORCH_FN(ActQuantizedGeLU));
  m.impl("sigmoid", TORCH_FN(ActQuantized<ActMode::SIGMOID>));
}

} // namespace
} // namespace musa
} // namespace at
