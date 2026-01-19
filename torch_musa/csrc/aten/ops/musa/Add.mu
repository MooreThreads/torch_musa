#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch_v2.h>
#include <ATen/OpMathType.h>

#include <c10/core/Scalar.h>

#include "torch_musa/csrc/aten/ops/musa/Loops.muh"

namespace at::musa {

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add(T self, T other, T alpha) {
  return self + alpha * other;
}

template <typename T>
C10_HOST_DEVICE C10_ALWAYS_INLINE T add_no_alpha(T self, T other) {
  return self + other;
}

template <typename scalar_t>
struct add_scalar_no_alpha {
  using opmath_t = at::opmath_type<scalar_t>;

  opmath_t one_;

  add_scalar_no_alpha(opmath_t one) : one_(one) {}

  __device__ scalar_t operator()(scalar_t another) const {
    return (scalar_t)add_no_alpha(one_, (opmath_t)another);
  }
};

template <typename scalar_t, bool left_scalar>
struct add_scalar_with_alpha {
  using opmath_t = at::opmath_type<scalar_t>;

  opmath_t self_;
  opmath_t alpha_;

  add_scalar_with_alpha(opmath_t self, opmath_t alpha)
      : self_(self), alpha_(alpha) {}

  __device__ scalar_t operator()(scalar_t other) const {
    return (scalar_t)add(self_, (opmath_t)other, alpha_);
  }
};

template <typename scalar_t>
struct add_scalar_with_alpha<scalar_t, false> : add_scalar_no_alpha<scalar_t> {
  using typename add_scalar_no_alpha<scalar_t>::opmath_t;

  add_scalar_with_alpha(opmath_t other, opmath_t alpha)
      : add_scalar_no_alpha<scalar_t>(other * alpha) {}
};

template <typename scalar_t>
struct add_tensor_no_alpha {
  using opmath_t = at::opmath_type<scalar_t>;

  __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return (scalar_t)add_no_alpha((opmath_t)self, (opmath_t)other);
  }
};

template <typename scalar_t>
struct add_tensor_with_alpha {
  using opmath_t = at::opmath_type<scalar_t>;

  opmath_t alpha_;

  add_tensor_with_alpha(opmath_t alpha) : alpha_(alpha) {}

  __device__ scalar_t operator()(scalar_t self, scalar_t other) const {
    return (scalar_t)add((opmath_t)self, (opmath_t)other, alpha_);
  }
};

void AddKernel(TensorIteratorBase& iter, const Scalar& alpha) {
  AT_DISPATCH_V2(
      iter.common_dtype(),
      "AddKernel",
      AT_WRAP([&]() {
        using opmath_t = at::opmath_type<scalar_t>;
        if (alpha.equal(1)) {
          if (iter.is_cpu_scalar(1)) {
            const auto f =
                add_scalar_no_alpha<scalar_t>(iter.scalar_value<opmath_t>(1));
            iter.remove_operand(1);
            gpu_kernel(iter, f);
          } else if (iter.is_cpu_scalar(2)) {
            const auto f =
                add_scalar_no_alpha<scalar_t>(iter.scalar_value<opmath_t>(2));
            iter.remove_operand(2);
            gpu_kernel(iter, f);
          } else {
            const auto f = add_tensor_no_alpha<scalar_t>();
            gpu_kernel(iter, f);
          }
        } else {
          const auto alp = (alpha).to<opmath_t>();
          if (iter.is_cpu_scalar(1)) {
            const auto f = add_scalar_with_alpha<scalar_t, true>(
                iter.scalar_value<opmath_t>(1), alp);
            iter.remove_operand(1);
            gpu_kernel(iter, f);
          } else if (iter.is_cpu_scalar(2)) {
            const auto f = add_scalar_with_alpha<scalar_t, false>(
                iter.scalar_value<opmath_t>(2), alp);
            iter.remove_operand(2);
            gpu_kernel(iter, f);
          } else {
            const auto f = add_tensor_with_alpha<scalar_t>(alp);
            gpu_kernel(iter, f);
          }
        }
      }),
      AT_EXPAND(AT_COMPLEX_TYPES));
}

} // namespace at::musa
