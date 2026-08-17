#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/UnaryOps.h>
#include <ATen/native/musa/JitLoops.muh>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/musa/Math.muh>
#include <limits>

namespace at::native {

// use force noinline inplace original impl
template <typename T>
MUSA_DEVICE_NOINLINE T ni_asinh(T a) {
  return ::asinh(a);
}

#if 0 && AT_USE_JITERATOR()
constexpr char asinh_name[] = "asinh_impl";
#endif

void asinh_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // Disabled due to accuracy issues
#if 0 && AT_USE_JITERATOR()
    static const auto asinh_string = jiterator_stringify(
        template <typename T> T asinh_impl(T a) { return std::asinh(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "asinh_name", [&]() {
          jitted_gpu_kernel<
              /*name=*/asinh_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, asinh_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "asinh_name", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ni_asinh<opmath_t>(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "asinh_cuda",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ni_asinh<scalar_t>(a);
          });
        });
  }
}

REGISTER_DISPATCH(asinh_stub, &asinh_kernel_cuda)

} // namespace at::native
