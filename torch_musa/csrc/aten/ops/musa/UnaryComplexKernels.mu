// Copied from aten/src/ATen/native/cuda/UnaryComplexKernels.cu
// explicitly porting angle kernel here is to avoid porting extra files, i.e.
// Copy.cu
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/Copy.h>
#include <ATen/native/musa/JitLoops.muh>
#include <ATen/native/musa/Loops.muh>
#include <limits>

#include "torch_musa/csrc/aten/ops/Copy.h"
namespace at::native {

// We manually overload angle because std::arg does not work with types other
// than c10::complex.
template <typename scalar_t>
__host__ __device__ static inline scalar_t angle_wrapper(scalar_t v) {
  if (at::_isnan(v)) {
    return v;
  }
  return v < 0 ? M_PI : 0;
}

template <typename T>
__host__ __device__ static inline c10::complex<T> angle_wrapper(
    c10::complex<T> v) {
  return c10::complex<T>{std::arg(v), 0};
}

#if AT_USE_JITERATOR()
CONSTEXPR_EXCEPT_WIN_CUDA char angle_name[] = "angle_kernel";
#endif

void angle_kernel_musa(TensorIteratorBase& iter) {
  auto dtype = iter.common_dtype();
  if (at::isComplexType(dtype)) {
#if AT_USE_JITERATOR()
    static const auto angle_string =
        jiterator_stringify(template <typename T> T angle_kernel(T v) {
          return T{std::arg(v)};
        }); // angle string
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "angle_cuda", [&]() {
      jitted_gpu_kernel<
          /*name=*/angle_name,
          /*return_dtype=*/scalar_t,
          /*common_dtype=*/scalar_t,
          /*arity=*/1>(iter, angle_string);
    });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(kComplexHalf, dtype, "angle_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return angle_wrapper(a);
      });
    });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES(dtype, "angle_cuda", [&]() {
      gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
        return angle_wrapper(a);
      });
    });
  }
}

// NB: Ignores the negative bit on tensors
constexpr char conj_name[] = "conj_kernel";
void conj_kernel_musa(TensorIteratorBase& iter) {
  auto conj_chalf = [&] {
    using scalar_t = c10::complex<at::Half>;
#if AT_USE_JITERATOR()
    static const auto conj_string = jiterator_stringify(
        template <typename T> T conj_kernel(T z) { return std::conj(z); });
    jitted_gpu_kernel<conj_name, scalar_t, scalar_t, 1>(iter, conj_string);
#else
    gpu_kernel(
        iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t { return std::conj(a); });
#endif
  };

  AT_DISPATCH_SWITCH(
      iter.common_dtype(),
      "conj_cuda",
      AT_DISPATCH_CASE_ALL_TYPES_AND3(kBool, kBFloat16, kHalf, [&] {
        // Conj is a no-op for non-complex types
        at::musa::direct_copy_kernel_musa(iter);
      }) AT_DISPATCH_CASE_COMPLEX_TYPES([&] {
        gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
          return std::conj(a);
        });
      }) AT_DISPATCH_CASE(kComplexHalf, conj_chalf));
}

REGISTER_MUSA_DISPATCH(angle_stub, &angle_kernel_musa);
REGISTER_DISPATCH(conj_physical_stub, &conj_kernel_musa)

} // namespace at::native
