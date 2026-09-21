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

template <typename T>
MUSA_DEVICE_NOINLINE T ni_tan(T a) {
  return ::tan(a);
}

#if 0 && AT_USE_JITERATOR()
constexpr char tan_name[] = "tan_impl";
#endif

void tan_kernel_cuda(TensorIteratorBase& iter) {
  auto common_dtype = iter.common_dtype();
  if (at::isComplexType(common_dtype)) {
    // Disabled due to accuracy issues
#if 0 && AT_USE_JITERATOR()
    static const auto tan_string = jiterator_stringify(
        template <typename T> T tan_impl(T a) { return std::tan(a); });
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tan_musa", [&]() {
          jitted_gpu_kernel<
              /*name=*/tan_name,
              /*return_dtype=*/scalar_t,
              /*common_dtype=*/scalar_t,
              /*arity=*/1>(iter, tan_string);
        });
#else
    AT_DISPATCH_COMPLEX_TYPES_AND(
        kComplexHalf, common_dtype, "tan_musa", [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            using opmath_t = at::opmath_type<scalar_t>;
            return ni_tan(static_cast<opmath_t>(a));
          });
        });
#endif
  } else {
    AT_DISPATCH_FLOATING_TYPES_AND2(
        ScalarType::Half,
        ScalarType::BFloat16,
        common_dtype,
        "tan_musa",
        [&]() {
          gpu_kernel(iter, [] GPU_LAMBDA(scalar_t a) -> scalar_t {
            return ni_tan(a);
          });
        });
  }
}

REGISTER_DISPATCH(tan_stub, &tan_kernel_cuda)

} // namespace at::native
