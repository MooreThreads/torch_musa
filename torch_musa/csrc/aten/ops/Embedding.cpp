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

DEFINE_DISPATCH(embedding_bag_stub);
DEFINE_DISPATCH(embedding_dense_backward_stub);

REGISTER_NO_CPU_DISPATCH(embedding_bag_stub);
REGISTER_NO_CPU_DISPATCH(embedding_dense_backward_stub);
} // namespace native

namespace musa {
namespace {

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

std::tuple<Tensor, Tensor, Tensor, Tensor> _EmbeddingBagImpl(
    const Tensor& weight,
    const Tensor& indices_,
    const Tensor& offsets_,
    const int64_t mode,
    bool include_last_offset,
    const Tensor& per_sample_weights,
    int64_t padding_idx,
    bool requires_grad) {
  TORCH_CHECK(
      indices_.dim() == 1 || indices_.dim() == 2,
      "input has to be a 1D or 2D Tensor, but got Tensor of dimension ",
      indices_.dim());
  if (indices_.dim() == 1) {
    TORCH_CHECK(
        offsets_.dim() == 1,
        "offsets has to be a 1D Tensor, but got Tensor of dimension ",
        offsets_.dim());
  }
  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);

  Tensor output = at::empty(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.sizes()[1]},
      weight.options());

  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  at::native::make_bag_size_out(
      bag_size, offsets, indices, mode, include_last_offset, requires_grad);
  Tensor max_indices = at::empty(bag_size.sizes(), offsets.options());
  at::native::make_max_indices_out(
      max_indices,
      weight,
      indices,
      offsets,
      bag_size,
      mode,
      include_last_offset);
  Tensor offset2bag = at::empty({0}, offsets.options());
  at::native::make_offset2bag_out(
      offset2bag,
      output,
      weight,
      indices,
      offsets,
      mode,
      per_sample_weights,
      padding_idx);

  at::native::embedding_bag_stub(
      kMUSA, output, weight, indices, offsets, mode, padding_idx);

  return std::make_tuple(
      std::move(output),
      std::move(offset2bag),
      std::move(bag_size),
      std::move(max_indices));
}

std::tuple<Tensor, Tensor, Tensor, Tensor> EmbeddingBag(
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const bool scale_grad_by_freq,
    const int64_t mode,
    bool sparse,
    const c10::optional<Tensor>& per_sample_weights_opt,
    bool include_last_offset,
    int64_t padding_idx) {
  TORCH_CHECK(
      weight.device().type() == kMUSA,
      "Device of weight tensor of embedding_bag must "
      "be MUSA, but now is ",
      weight.device());
  TORCH_CHECK(
      indices.device().type() == kMUSA,
      "Device of indices tensor of embedding_bag must be "
      "MUSA, but now is ",
      indices.device());
  TORCH_CHECK(
      offsets.device().type() == kMUSA,
      "Device of offset tensor of embedding_bag must be "
      "MUSA, but now is ",
      offsets.device());
  c10::musa::MUSAGuard device_guard(weight.device());
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  UNUSED(scale_grad_by_freq);
  UNUSED(sparse);

  return _EmbeddingBagImpl(
      weight,
      indices,
      offsets,
      mode,
      include_last_offset,
      per_sample_weights,
      padding_idx,
      /*requires_grad*/ true);
}

} // namespace musa
} // namespace at
