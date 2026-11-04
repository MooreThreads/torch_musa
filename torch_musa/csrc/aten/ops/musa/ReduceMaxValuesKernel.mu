#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/NumericUtils.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/ReduceAllOps.h>
#include <ATen/native/ReduceOps.h>
#include <ATen/native/SharedReduceOps.h>
#include <ATen/native/TensorCompare.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/musa/NumericLimits.muh>

#include "torch_musa/csrc/aten/ops/musa/Reduce.muh"

namespace at::native {

template <typename acc_t>
struct MaxNanFunctor {
  __device__ __forceinline__ acc_t operator()(acc_t a, acc_t b) const {
    return (at::_isnan(a) || a > b) ? a : b;
  }
};

template <typename scalar_t, typename acc_t = scalar_t>
void max_values_kernel_musa_impl(TensorIterator& iter) {
  at::native_musa_reduce::gpu_reduce_kernel<scalar_t, scalar_t>(
      iter,
      at::native_musa_reduce::func_wrapper<acc_t>(MaxNanFunctor<acc_t>()),
      at::numeric_limits<acc_t>::lower_bound());
}

void max_values_kernel_musa(TensorIterator& iter) {
  AT_DISPATCH_ALL_TYPES_AND3(
      kBFloat16, kHalf, kBool, iter.dtype(), "max_values_musa", [&]() {
        max_values_kernel_musa_impl<scalar_t>(iter);
      });
}

REGISTER_DISPATCH(max_values_stub, &max_values_kernel_musa)

} // namespace at::native
