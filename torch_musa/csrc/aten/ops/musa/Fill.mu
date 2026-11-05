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

template <typename scalar_t>
struct masked_fill_scalar {
  scalar_t value_;

  masked_fill_scalar(scalar_t value) : value_(value) {}

  __device__ __forceinline__ scalar_t operator()(scalar_t self, bool mask)
      const {
    return mask ? value_ : self;
  }
};

// In-house masked fill for tensors past the 32-bit element-count limit of
// the MUDNN masked Fill: gpu_kernel splits 64-bit iteration into 32-bit
// sub-ranges (Loops.muh), so this path is safe at any numel.
void MaskedFillKernel(TensorIteratorBase& iter, const Scalar& value) {
  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND3(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      at::ScalarType::Bool,
      iter.dtype(),
      "MaskedFillKernel",
      [&]() {
        auto f = masked_fill_scalar<scalar_t>(value.to<scalar_t>());
        gpu_kernel(iter, f);
      });
}

} // namespace at::musa
