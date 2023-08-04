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

void quantize_tensor_per_tensor_affine_musa(
    const Tensor& rtensor,
    Tensor& qtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "quantize_tensor_per_tensor_affine_musa", [&]() {
        constexpr int64_t qmin = std::numeric_limits<underlying_t>::min();
        constexpr int64_t qmax = std::numeric_limits<underlying_t>::max();

        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(qtensor)
                        .add_input(rtensor)
                        .add_input(qtensor)
                        .build();
        gpu_kernel(
            iter,
            [=] GPU_LAMBDA(float raw_val, scalar_t quantized_val) -> scalar_t {
              int64_t qvalue = static_cast<int64_t>(
                  std::nearbyint(raw_val / scale) + zero_point);
              qvalue = std::max<int64_t>(qvalue, qmin);
              qvalue = std::min<int64_t>(qvalue, qmax);
              quantized_val.val_ = qvalue;
              return quantized_val;
            });
      });
}

void dequantize_tensor_per_tensor_affine_musa(
    const Tensor& qtensor,
    Tensor& rtensor,
    double scale,
    int64_t zero_point) {
  AT_DISPATCH_QINT_TYPES(
      qtensor.scalar_type(), "dequantize_tensor_per_tensor_affine_musa", [&]() {
        auto iter = TensorIteratorConfig()
                        .check_all_same_dtype(false)
                        .add_output(rtensor)
                        .add_input(qtensor)
                        .build();
        gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t value) -> float {
          return (static_cast<float>(value.val_) - zero_point) * scale;
        });
      });
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
