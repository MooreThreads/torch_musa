#ifndef TORCH_MUSA_CSRC_ATEN_OPS_MUSA_MUSA_OPS_H
#define TORCH_MUSA_CSRC_ATEN_OPS_MUSA_MUSA_OPS_H
#include <ATen/core/TensorBody.h>
#include <c10/core/ScalarType.h>

namespace at {
namespace musa {

Tensor Tril(const Tensor& self, int64_t diagonal);

Tensor Baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha);

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_MUSA_MUSA_OPS_H
