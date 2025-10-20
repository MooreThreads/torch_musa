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

REGISTER_MUSA_DISPATCH(angle_stub, &angle_kernel_musa);

} // namespace at::native
