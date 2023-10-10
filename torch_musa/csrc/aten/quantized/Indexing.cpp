#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/index_add_native.h>
#include <ATen/ops/index_reduce_native.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/masked_fill_native.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/zeros_like.h>
#endif

#include <torch/library.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

Tensor& QIndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_out_cuda(self, dim, index, out);
}

Tensor QIndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_quantized_cuda(self, dim, index);
}

ADVANCED_REGISTER(
    aten,
    QuantizedPrivateUse1,
    "index_select.out",
    QIndexSelectOut)
ADVANCED_REGISTER(aten, QuantizedPrivateUse1, "index_select", QIndexSelect)

} // namespace musa
} // namespace at
