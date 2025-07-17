#include <ATen/TensorIterator.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/musa/thread_constants.h>
#include <ATen/native/quantized/AffineQuantizer.h>
#include <ATen/native/musa/Loops.muh>
#include <cmath>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/any.h>
#include <ATen/ops/gt.h>
#include <ATen/ops/lt.h>
#endif

namespace at {
namespace native {
namespace {

template <typename T>
void check_zero_points_musa(
    const std::string& fn_name,
    const Tensor& zero_points) {
  constexpr int64_t qmin = std::numeric_limits<T>::min();
  constexpr int64_t qmax = std::numeric_limits<T>::max();
  auto zp_within_upper = at::any(at::gt(zero_points, qmax)).item().equal(false);
  auto zp_within_lower = at::any(at::lt(zero_points, qmin)).item().equal(false);
  TORCH_CHECK(zp_within_lower, fn_name, "zero_point is below lower bound.");
  TORCH_CHECK(zp_within_upper, fn_name, "zero_point is above upper bound.");
}

template <typename DType>
__global__ void QuantizePerTensorAffineKernel(
    DType* out,
    const float* in,
    const double inv_scale,
    const int64_t zero_point,
    const int64_t qmin,
    const int64_t qmax,
    const int64_t total_num) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = tid * 4;

  if (idx + 3 < total_num) {
    const float4 in_val = reinterpret_cast<const float4*>(in + idx)[0];

    int4 qval;
    qval.x = __float2int_rn(in_val.x * inv_scale) + zero_point;
    qval.y = __float2int_rn(in_val.y * inv_scale) + zero_point;
    qval.z = __float2int_rn(in_val.z * inv_scale) + zero_point;
    qval.w = __float2int_rn(in_val.w * inv_scale) + zero_point;

    qval.x = std::min<int32_t>(std::max<int32_t>(qval.x, qmin), qmax);
    qval.y = std::min<int32_t>(std::max<int32_t>(qval.y, qmin), qmax);
    qval.z = std::min<int32_t>(std::max<int32_t>(qval.z, qmin), qmax);
    qval.w = std::min<int32_t>(std::max<int32_t>(qval.w, qmin), qmax);

    out[idx] = static_cast<DType>(qval.x);
    out[idx + 1] = static_cast<DType>(qval.y);
    out[idx + 2] = static_cast<DType>(qval.z);
    out[idx + 3] = static_cast<DType>(qval.w);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t elem_idx = idx + i;
      if (elem_idx >= total_num)
        return;

      const int qval = __float2int_rn(in[elem_idx] * inv_scale) + zero_point;
      out[elem_idx] = static_cast<DType>(
          std::min<int32_t>(std::max<int32_t>(qval, qmin), qmax));
    }
  }
}

template <typename DType>
__global__ void DequantizePerTensorAffineKernel(
    float* out,
    const DType* in,
    const double scale,
    const int64_t zero_point,
    const int64_t total_num) {
  int64_t idx = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
  if (idx < total_num) {
    float value0 = (static_cast<float>(in[idx]) - zero_point) * scale;
    float value1 = (static_cast<float>(in[idx + 1]) - zero_point) * scale;
    float value2 = (static_cast<float>(in[idx + 2]) - zero_point) * scale;
    float value3 = (static_cast<float>(in[idx + 3]) - zero_point) * scale;
    out[idx] = value0;
    out[idx + 1] = value1;
    out[idx + 2] = value2;
    out[idx + 3] = value3;
  }
}

void quantize_tensor_per_tensor_affine_musa(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  auto stream = c10::musa::getCurrentMUSAStream();
  int64_t numel = qtensor.numel();

  uint32_t block_x = numel > 512 ? 1024 : 512;
  uint32_t grid_x = (numel / 4 + block_x - 1) / block_x;

  dim3 block_size{block_x, 1, 1};
  dim3 grid_size{grid_x, 1, 1};

  double inv_scale = 1.0f / scale;

  if (qtensor.scalar_type() == ScalarType::QInt8) {
    constexpr int64_t qmin = std::numeric_limits<int8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<int8_t>::max();
    QuantizePerTensorAffineKernel<int8_t><<<grid_size, block_size, 0, stream>>>(
        static_cast<int8_t*>(qtensor.data_ptr()),
        (float*)rtensor.data_ptr(),
        inv_scale,
        zero_point,
        qmin,
        qmax,
        numel);
  } else if (qtensor.scalar_type() == ScalarType::QUInt8) {
    constexpr int64_t qmin = std::numeric_limits<uint8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<uint8_t>::max();
    QuantizePerTensorAffineKernel<uint8_t>
        <<<grid_size, block_size, 0, stream>>>(
            static_cast<uint8_t*>(qtensor.data_ptr()),
            (float*)rtensor.data_ptr(),
            inv_scale,
            zero_point,
            qmin,
            qmax,
            numel);
  } else {
    TORCH_CHECK(
        false, "quantize_per_tensor now only supports qint8 and quint8");
  }
  musaDeviceSynchronize();
}

void dequantize_tensor_per_tensor_affine_musa(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  auto stream = c10::musa::getCurrentMUSAStream();
  int64_t numel = rtensor.numel();

  uint32_t block_x = numel > 512 ? 1024 : 512;
  uint32_t grid_x = (numel / 4 + block_x - 1) / block_x;

  dim3 block_size{block_x, 1, 1};
  dim3 grid_size{grid_x, 1, 1};

  if (qtensor.scalar_type() == ScalarType::QInt8) {
    DequantizePerTensorAffineKernel<int8_t>
        <<<grid_size, block_size, 0, stream>>>(
            (float*)rtensor.data_ptr(),
            static_cast<int8_t*>(qtensor.data_ptr()),
            scale,
            zero_point,
            numel);
  } else if (qtensor.scalar_type() == ScalarType::QUInt8) {
    DequantizePerTensorAffineKernel<uint8_t>
        <<<grid_size, block_size, 0, stream>>>(
            (float*)rtensor.data_ptr(),
            static_cast<uint8_t*>(qtensor.data_ptr()),
            scale,
            zero_point,
            numel);
  } else {
    TORCH_CHECK(false, "qint8 and quint8 quantized tensor can be dequantized");
  }
  musaDeviceSynchronize();
}

void quantize_tensor_per_channel_affine_musa(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "quantize_tensor_per_channel_affine_musa";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_musa<underlying_t>(fn_name, zero_points);

    constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
    constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
    // trying to match _quantize_per_channel_ref_nd in test_quantized_tensor.py
    gpu_kernel(
        iter,
        [=] GPU_LAMBDA(
            float raw_val,
            scalar_t quantized_val,
            double scale,
            int64_t zero_point) -> scalar_t {
          int64_t qvalue = static_cast<int64_t>(
              std::nearbyint(raw_val / scale) + zero_point);
          qvalue = std::max<int64_t>(qvalue, qmin);
          qvalue = std::min<int64_t>(qvalue, qmax);
          quantized_val.val_ = qvalue;
          return quantized_val;
        });
  });
}

void dequantize_tensor_per_channel_affine_musa(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name = "dequantize_tensor_per_channel_affine_musa";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_musa<underlying_t>(fn_name, zero_points);

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .add_output(rtensor)
                    .add_input(qtensor)
                    .add_input(shaped_scales)
                    .add_input(shaped_zero_points)
                    .build();

    gpu_kernel(
        iter,
        [=] GPU_LAMBDA(
            scalar_t value, double scale, int64_t zero_point) -> float {
          return static_cast<float>(value.val_ - zero_point) * scale;
        });
  });
}

void quantize_tensor_per_channel_float_qparams_musa(
    const Tensor& rtensor,
    Tensor& qtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "quantize_tensor_per_channel_float_qparams_musa";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  auto iter = TensorIteratorConfig()
                  .check_all_same_dtype(false)
                  .add_output(qtensor)
                  .add_input(rtensor)
                  .add_input(qtensor)
                  .add_input(shaped_scales)
                  .add_input(shaped_zero_points)
                  .build();

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_musa<underlying_t>(fn_name, zero_points);

    constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
    constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();
    // trying to match _quantize_per_channel_ref_nd in
    gpu_kernel(
        iter,
        [=] GPU_LAMBDA(
            float raw_val,
            scalar_t quantized_val,
            float scale,
            float zero_point) -> scalar_t {
          float inv_scale = 1.0f / scale;
          int64_t qvalue = lrintf(raw_val * inv_scale + zero_point);
          qvalue = std::max<int64_t>(qvalue, qmin);
          qvalue = std::min<int64_t>(qvalue, qmax);
          quantized_val.val_ = qvalue;
          return quantized_val;
        });
  });
}

void dequantize_tensor_per_channel_float_qparams_musa(
    const Tensor& qtensor,
    Tensor& rtensor,
    const Tensor& scales,
    const Tensor& zero_points,
    int64_t axis) {
  static constexpr auto fn_name =
      "dequantize_tensor_per_channel_float_qparams_musa";
  std::vector<int64_t> expected_shape(rtensor.dim(), 1);
  expected_shape[axis] = rtensor.size(axis);

  auto shaped_scales = native::_unsafe_view(scales, expected_shape);
  auto shaped_zero_points = native::_unsafe_view(zero_points, expected_shape);

  AT_DISPATCH_QINT_TYPES(qtensor.scalar_type(), fn_name, [&]() {
    check_zero_points_musa<underlying_t>(fn_name, zero_points);

    auto iter = TensorIteratorConfig()
                    .check_all_same_dtype(false)
                    .add_output(rtensor)
                    .add_input(qtensor)
                    .add_input(shaped_scales)
                    .add_input(shaped_zero_points)
                    .build();

    gpu_kernel(
        iter,
        [=] GPU_LAMBDA(scalar_t value, float scale, float zero_point) -> float {
          return (static_cast<float>(value.val_) - zero_point) * scale;
        });
  });
}

} // anonymous namespace

REGISTER_DISPATCH(
    quantize_tensor_per_tensor_affine_stub,
    &quantize_tensor_per_tensor_affine_musa);
REGISTER_DISPATCH(
    dequantize_tensor_per_tensor_affine_stub,
    &dequantize_tensor_per_tensor_affine_musa);
REGISTER_DISPATCH(
    quantize_tensor_per_channel_affine_stub,
    &quantize_tensor_per_channel_affine_musa);
REGISTER_DISPATCH(
    dequantize_tensor_per_channel_affine_stub,
    &dequantize_tensor_per_channel_affine_musa);
REGISTER_DISPATCH(
    quantize_tensor_per_channel_float_qparams_stub,
    &quantize_tensor_per_channel_float_qparams_musa);
REGISTER_DISPATCH(
    dequantize_tensor_per_channel_float_qparams_stub,
    &dequantize_tensor_per_channel_float_qparams_musa);

} // namespace native
} // namespace at
