#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Math.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/JitLoops.muh>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/musa/Math.muh>
#include <limits>

namespace at::native {

// use force noinline inplace original impl
template <typename T>
MUSA_DEVICE_NOINLINE T ni_digamma(T a) {
  return calc_digamma(a);
}
template <typename T>
MUSA_DEVICE_NOINLINE T ni_trigamma(T a) {
  return calc_trigamma(a);
}
template <typename T>
MUSA_DEVICE_NOINLINE T ni_polygamma(T a, int n) {
  return calc_polygamma<T, true>(a, n);
}
template <typename T>
MUSA_DEVICE_NOINLINE T ni_lgamma(T a) {
  return ::lgamma(a);
}

#if AT_USE_JITERATOR()
constexpr char digamma_name[] = "digamma";
#endif // AT_USE_JITERATOR()
// See note [Jiterator]
void digamma_kernel_cuda(TensorIteratorBase& iter) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "digamma_cuda",
      [&]() {
        jitted_gpu_kernel<
            /*name=*/digamma_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/1>(iter, digamma_string);
      });
#else
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "digamma_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ni_digamma<scalar_t>(a);
        });
      });
#endif // AT_USE_JITERATOR()
}

// See note [Jiterator]
constexpr char trigamma_name[] = "trigamma";
void trigamma_kernel_cuda(TensorIteratorBase& iter) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "trigamma_cuda",
      [&]() {
        jitted_gpu_kernel<
            /*name=*/trigamma_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/1>(iter, trigamma_string);
      });
#else
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "trigamma_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ni_trigamma<scalar_t>(a);
        });
      });
#endif // AT_USE_JITERATOR()
}

constexpr char polygamma_name[] = "polygamma";
void polygamma_kernel_cuda(TensorIteratorBase& iter, int64_t n) {
  if (n == 0) {
    digamma_kernel_cuda(iter);
  } else if (n == 1) {
    trigamma_kernel_cuda(iter);
  } else {
#if AT_USE_JITERATOR()
    // TODO : `unary_jitted_gpu_kernel` for cleaner UX.
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "polygamma_cuda",
        [&]() {
          jitted_gpu_kernel<
              /*name=*/polygamma_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(
              iter,
              polygamma_string,
              /*scalar_pos=*/at::musa::jit::BinaryFuncVariant::NoScalar,
              /*scalar_val=*/0,
              /*extra_args=*/std::make_tuple(n));
        });
#else
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        iter.common_dtype(),
        "polygamma_cuda",
        [&]() {
          gpu_kernel(iter, [=] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ni_polygamma<scalar_t>(a, static_cast<int>(n));
          });
        });
#endif // AT_USE_JITERATOR()
  }
}

constexpr char lgamma_name[] = "lgamma_kernel";
void lgamma_kernel_cuda(TensorIteratorBase& iter) {
#if AT_USE_JITERATOR()
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lgamma_cuda",
      [&]() {
        jitted_gpu_kernel<
            /*name=*/lgamma_name,
            /*return_dtype=*/scalar_t,
            /*common_dtype=*/scalar_t,
            /*arity=*/1>(iter, lgamma_string);
      });
#else
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "lgamma_cuda",
      [&]() {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return ni_lgamma<scalar_t>(a);
        });
      });
#endif
}

REGISTER_DISPATCH(digamma_stub, &digamma_kernel_cuda)
REGISTER_DISPATCH(polygamma_stub, &polygamma_kernel_cuda)
REGISTER_DISPATCH(lgamma_stub, &lgamma_kernel_cuda)

} // namespace at::native
