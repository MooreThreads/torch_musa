#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/musa/Loops.muh>

// NOTE: CUDA on Windows requires that the enclosing function
// of a __device__ lambda not have internal linkage.

namespace at::native {

// use force noinline inplace original impl
template <typename T>
MUSA_DEVICE_NOINLINE T ni_atan2(T a, T b) {
  return ::atan2(a, b);
}
template <typename T>
MUSA_DEVICE_NOINLINE T ni_hypot(T a, T b) {
  return ::hypot(a, b);
}

void atan2_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "atan2_cuda",
      [&]() {
        gpu_kernel_with_scalars(
            iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
              return ni_atan2<scalar_t>(a, b);
            });
      });
}

void hypot_kernel_cuda(TensorIteratorBase& iter) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      iter.common_dtype(),
      "hypot_cuda",
      [&]() {
        opmath_symmetric_gpu_kernel_with_scalars<scalar_t>(
            iter, [] GPU_LAMBDA(scalar_t a, scalar_t b) -> scalar_t {
              return ni_hypot<scalar_t>(a, b);
            });
      });
}

REGISTER_DISPATCH(atan2_stub, &atan2_kernel_cuda)
REGISTER_DISPATCH(hypot_stub, &hypot_kernel_cuda)

} // namespace at::native
