#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Math.h>
#include <ATen/native/musa/jit_utils.h>
#include <ATen/native/musa/JitLoops.muh>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/musa/Math.muh>

namespace at::native {

// use force noinline inplace original impl
template <typename T>
MUSA_DEVICE_NOINLINE T ni_shifted_chebyshev_polynomial_u_forward(T x, T n) {
  return shifted_chebyshev_polynomial_u_forward<T, true>(x, n);
}
namespace {
constexpr char shifted_chebyshev_polynomial_u_name[] =
    "shifted_chebyshev_polynomial_u_forward";

void shifted_chebyshev_polynomial_u_kernel_cuda(TensorIteratorBase& iterator) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda", [&]() {
        opmath_jitted_gpu_kernel_with_scalars<
            shifted_chebyshev_polynomial_u_name,
            scalar_t,
            scalar_t>(iterator, shifted_chebyshev_polynomial_u_string);
      });
#else
  AT_DISPATCH_FLOATING_TYPES(
      iterator.common_dtype(), "shifted_chebyshev_polynomial_u_cuda", [&]() {
        gpu_kernel_with_scalars(
            iterator, [] GPU_LAMBDA(scalar_t x, scalar_t n) -> scalar_t {
              return ni_shifted_chebyshev_polynomial_u_forward<scalar_t>(x, n);
            });
      });
#endif
} // shifted_chebyshev_polynomial_u_kernel_cuda
} // namespace

REGISTER_DISPATCH(
    shifted_chebyshev_polynomial_u_stub,
    &shifted_chebyshev_polynomial_u_kernel_cuda)
} // namespace at::native
