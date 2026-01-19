#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>

#include "torch_musa/csrc/aten/ops/musa/Loops.muh"

namespace at::musa {

template <typename T>
__device__ inline T true_div(T a, T b) {
  static_assert(std::is_same_v<T, at::opmath_type<T>>);
  static_assert(!std::is_same_v<T, bool>);
  return a / b;
}

template <typename T>
struct true_div_right_scalar {
  T right_rcp_;

  true_div_right_scalar(T right) : right_rcp_(T(1.0) / right) {}

  __device__ T operator()(T left) const {
    static_assert(std::is_same_v<T, at::opmath_type<T>>);
    static_assert(!std::is_same_v<T, bool>);
    return left * right_rcp_;
  }
};

template <typename T>
struct true_div_left_scalar {
  T left_;

  true_div_left_scalar(T left) noexcept : left_(left) {}

  __device__ T operator()(T right) const {
    return true_div<T>(left_, right);
  }
};

template <typename T>
struct true_div_tensor {
  __device__ T operator()(T left, T right) const {
    return true_div<T>(left, right);
  }
};

void TrueDivKernel(TensorIteratorBase& iter) {
  AT_DISPATCH_V2(
      iter.common_dtype(),
      "TrueDivKernel",
      AT_WRAP([&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        if (iter.is_cpu_scalar(1)) {
          const auto f =
              true_div_left_scalar<opmath_t>(iter.scalar_value<opmath_t>(1));
          iter.remove_operand(1);
          gpu_kernel(iter, f);
        } else if (iter.is_cpu_scalar(2)) {
          const auto f =
              true_div_right_scalar<opmath_t>(iter.scalar_value<opmath_t>(2));
          iter.remove_operand(2);
          gpu_kernel(iter, f);
        } else {
          const auto f = true_div_tensor<opmath_t>();
          gpu_kernel(iter, f);
        }
      }),
      AT_EXPAND(AT_COMPLEX_TYPES));
}

} // namespace at::musa

namespace at::native {

REGISTER_DISPATCH(div_true_stub, &at::musa::TrueDivKernel)

} // namespace at::native
