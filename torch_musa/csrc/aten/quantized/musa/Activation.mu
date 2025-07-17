#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <c10/core/QScheme.h>
#include <torch/library.h>

#ifndef AT_ACTIVATION_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#endif

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

const int PARALLEL_NUM = 4;

#define ACT_QUANT_CALL(dtype, act_mode)                       \
  constexpr int64_t qmin = std::numeric_limits<dtype>::min(); \
  constexpr int64_t qmax = std::numeric_limits<dtype>::max(); \
  ActQuantizedKernel<dtype, act_mode>                         \
    <<<grid_size, block_size, 0, stream>>>(                   \
      static_cast<dtype*>(out.data_ptr()),                    \
      static_cast<dtype*>(self.data_ptr()),                   \ 
      scale,                                                  \
      inv_scale,                                              \
      zero_point,                                             \
      qmin,                                                   \
      qmax,                                                   \
      numel);

#define KERNEl_CALL(name, idx, val) \
  out[idx] = name(                  \
      static_cast<int64_t>(val), scale, inv_scale, zero_point, qmin, qmax);

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
    const double inv_scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax) {
  float value = static_cast<float>(input - zero_point) * scale;
  value = 1.f / (1.f + expf(-value));
  int64_t qvalue = std::max<int64_t>(
      std::min<int64_t>(
          static_cast<int64_t>(__float2int_rn(value * inv_scale) + zero_point),
          qmax),
      qmin);
  return qvalue;
}

__device__ int64_t GeLUKernel(
    int64_t input,
    const double scale,
    const double inv_scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax) {
  float value = static_cast<float>(input - zero_point) * scale;
  value = value * 0.5 * (1.f + erff(value / 1.4142135));
  int64_t qvalue = std::max<int64_t>(
      std::min<int64_t>(
          static_cast<int64_t>(__float2int_rn(value * inv_scale) + zero_point),
          qmax),
      qmin);
  return qvalue;
}

__device__ int64_t ReLUKernel(
    int64_t input,
    const double scale,
    const double inv_scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax) {
  float value =
      std::max<float>(static_cast<float>(input - zero_point) * scale, 0.f);
  int64_t qvalue = std::max<int64_t>(
      std::min<int64_t>(
          static_cast<int64_t>(__float2int_rn(value * inv_scale) + zero_point),
          qmax),
      qmin);
  return qvalue;
}

template <typename DType, ActMode mode>
__global__ void ActQuantizedKernel(
    DType* out,
    const DType* in,
    const double scale,
    const double inv_scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t total_num) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = tid * PARALLEL_NUM;
  using VecType = typename std::
      conditional<std::is_same<DType, int8_t>::value, char4, uchar4>::type;
  const VecType in_val = reinterpret_cast<const VecType*>(in)[tid];

  if (idx < total_num) {
    switch (mode) {
      case ActMode::RELU:
        KERNEl_CALL(ReLUKernel, idx, in_val.x);
        KERNEl_CALL(ReLUKernel, idx + 1, in_val.y);
        KERNEl_CALL(ReLUKernel, idx + 2, in_val.z);
        KERNEl_CALL(ReLUKernel, idx + 3, in_val.w);
        break;

      case ActMode::GELU:
        KERNEl_CALL(GeLUKernel, idx, in_val.x);
        KERNEl_CALL(GeLUKernel, idx + 1, in_val.y);
        KERNEl_CALL(GeLUKernel, idx + 2, in_val.z);
        KERNEl_CALL(GeLUKernel, idx + 3, in_val.w);
        break;

      case ActMode::SIGMOID:
        KERNEl_CALL(SigmoidKernel, idx, in_val.x);
        KERNEl_CALL(SigmoidKernel, idx + 1, in_val.y);
        KERNEl_CALL(SigmoidKernel, idx + 2, in_val.z);
        KERNEl_CALL(SigmoidKernel, idx + 3, in_val.w);
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

  double inv_scale = 1.0f / scale;

  dim3 block_size{block_x, 1, 1};
  dim3 grid_size{grid_x, 1, 1};

  if (self.scalar_type() == ScalarType::QInt8) {
    ACT_QUANT_CALL(int8_t, mode)
  } else if (self.scalar_type() == ScalarType::QUInt8) {
    ACT_QUANT_CALL(uint8_t, mode)
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
        self.options(),
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
        self.options(),
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
