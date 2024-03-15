#define TORCH_ASSERT_NO_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Fill.h>
#include <ATen/native/TensorIterator.h>
#include <c10/core/Scalar.h>
#include <ATen/native/musa/Loops.muh>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::native {

template <typename scalar_t>
struct FillFunctor {
  FillFunctor(scalar_t v) : value(v) {}
  __device__ __forceinline__ scalar_t operator()() const {
    return value;
  }

 private:
  scalar_t value;
};

void fill_kernel_musa(TensorIterator& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND4(
      kComplexHalf, kBool, kHalf, kBFloat16, iter.dtype(), "fill_musa", [&]() {
        gpu_kernel(iter, FillFunctor<scalar_t>(value.to<scalar_t>()));
      });
}

REGISTER_MUSA_DISPATCH(fill_stub, &fill_kernel_musa);

} // namespace at::native
