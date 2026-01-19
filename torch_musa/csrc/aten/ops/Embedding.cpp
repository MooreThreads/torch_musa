#include <ATen/Config.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/native/EmbeddingBag.h>
#else
#include <ATen/native/EmbeddingBag.h>
#include <ATen/ops/_sparse_coo_tensor_unsafe.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_dense_backward.h>
#include <ATen/ops/embedding_dense_backward_native.h>
#include <ATen/ops/embedding_native.h>
#include <ATen/ops/embedding_renorm_native.h>
#include <ATen/ops/embedding_sparse_backward.h>
#include <ATen/ops/embedding_sparse_backward_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/zeros.h>
#endif

#include "torch_musa/csrc/aten/ops/Embedding.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

DEFINE_DISPATCH(embedding_dense_backward_stub);

REGISTER_NO_CPU_DISPATCH(embedding_dense_backward_stub);
} // namespace native

namespace musa {
namespace {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

std::pair<Tensor, Tensor> promoteIndicesAndOffsets(
    const Tensor& indices,
    const Tensor& offsets) {
  const auto commonType =
      promoteTypes(offsets.scalar_type(), indices.scalar_type());
  return {
      indices.scalar_type() == commonType ? indices
                                          : indices.toType(commonType),
      offsets.scalar_type() == commonType ? offsets
                                          : offsets.toType(commonType)};
}

} // namespace

Tensor EmbeddingDenseBwd(
    const at::Tensor& grad_output,
    const at::Tensor& indices,
    int64_t num_weights,
    int64_t padding_idx,
    bool scale_grad_by_freq) {
  c10::musa::MUSAGuard device_guard(grad_output.device());
  return at::native::embedding_dense_backward_stub(
      kMUSA,
      grad_output,
      indices,
      num_weights,
      padding_idx,
      scale_grad_by_freq);
}

} // namespace musa
} // namespace at
