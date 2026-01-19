#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>
#include <ATen/native/BinaryOps.h>

#include "torch_musa/csrc/aten/ops/musa/Loops.muh"

namespace at::musa {

template <typename T>
__device__ inline T mul(T a, T b) {
  static_assert(std::is_same_v<T, at::opmath_type<T>>);
  if constexpr (std::is_same_v<T, bool>) {
    return a && b;
  } else {
    return a * b;
  }
}

template <typename T>
struct mul_scalar {
  T one_;

  mul_scalar(T one) noexcept : one_(one) {}

  __device__ T operator()(T another) const {
    return mul<T>(one_, another);
  }
};

template <typename T>
struct mul_tensor {
  __device__ T operator()(T a, T b) const {
    return mul<T>(a, b);
  }
};

void MulKernel(TensorIteratorBase& iter) {
  AT_DISPATCH_V2(
      iter.common_dtype(),
      "MulKernel",
      AT_WRAP([&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        if (iter.is_cpu_scalar(1)) {
          const auto f = mul_scalar<opmath_t>(iter.scalar_value<opmath_t>(1));
          iter.remove_operand(1);
          gpu_kernel(iter, f);
        } else if (iter.is_cpu_scalar(2)) {
          const auto f = mul_scalar<opmath_t>(iter.scalar_value<opmath_t>(2));
          iter.remove_operand(2);
          gpu_kernel(iter, f);
        } else {
          const auto f = mul_tensor<opmath_t>();
          gpu_kernel(iter, f);
        }
      }),
      AT_EXPAND(AT_COMPLEX_TYPES));
}

} // namespace at::musa

namespace at::native {

REGISTER_DISPATCH(mul_stub, &at::musa::MulKernel)

} // namespace at::native
