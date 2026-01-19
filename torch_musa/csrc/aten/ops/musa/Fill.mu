#define TORCH_ASSERT_NO_OPERATORS

#include <ATen/Dispatch_v2.h>
#include <c10/core/Scalar.h>

#include "torch_musa/csrc/aten/ops/musa/Loops.muh"

namespace at::musa {

template <typename scalar_t>
struct fill_scalar {
  scalar_t value_;

  fill_scalar(scalar_t value) : value_(value) {}

  __device__ __forceinline__ scalar_t operator()() const {
    return value_;
  }
};

void FillKernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_V2(
      iter.dtype(),
      "FillKernel",
      AT_WRAP([&]() {
        auto f = fill_scalar<scalar_t>(value.to<scalar_t>());
        gpu_kernel(iter, f);
      }),
      AT_EXPAND(AT_COMPLEX_TYPES));
}

} // namespace at::musa
