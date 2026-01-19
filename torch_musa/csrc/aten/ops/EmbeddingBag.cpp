#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <torch/library.h>

#include <ATen/NativeFunctions.h>
#include <ATen/TensorOperators.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/EmbeddingBag.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#include <ATen/native/EmbeddingBag.h>
#else
#include <ATen/ops/_embedding_bag.h>
#include <ATen/ops/_embedding_bag_backward_native.h>
#include <ATen/ops/_embedding_bag_dense_backward.h>
#include <ATen/ops/_embedding_bag_dense_backward_native.h>
#include <ATen/ops/_embedding_bag_forward_only.h>
#include <ATen/ops/_embedding_bag_forward_only_native.h>
#include <ATen/ops/_embedding_bag_native.h>
#include <ATen/ops/_embedding_bag_per_sample_weights_backward_native.h>
#include <ATen/ops/_embedding_bag_sparse_backward.h>
#include <ATen/ops/_embedding_bag_sparse_backward_native.h>
#include <ATen/ops/embedding_backward_native.h>
#include <ATen/ops/embedding_bag_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/max.h>
#include <ATen/ops/ones_like.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/zero_native.h>
#include <ATen/ops/zeros.h>
#endif

#include "torch_musa/csrc/aten/ops/Embedding.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

DEFINE_DISPATCH(embedding_bag_stub);

REGISTER_NO_CPU_DISPATCH(embedding_bag_stub);

} // namespace native

namespace musa {
namespace {

constexpr int MODE_SUM = 0;
constexpr int MODE_MEAN = 1;
constexpr int MODE_MAX = 2;

static void make_offset2bag(const Tensor& offsets, Tensor& offset2bag) {
  offset2bag.index_add_(
      0,
      offsets,
      at::ones_like(
          offsets,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT)); // offset2bag = [1 0 1 0 1]
  offset2bag[0] -= 1; // offset2bag = [0 0 1 0 1]
  offset2bag = offset2bag.cumsum(
      0, offset2bag.scalar_type()); // offset2bag = [0 0 1 1 2]
}

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

Tensor EmbeddingBagBackward(
    const Tensor& grad,
    const Tensor& indices_,
    const Tensor& offsets_,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    const Tensor& max_indices_,
    int64_t num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    bool sparse,
    const std::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  TORCH_CHECK(
      grad.device().type() == kMUSA,
      "Device of grad tensor of embedding_bag must "
      "be MUSA, but now is ",
      grad.device());
  TORCH_CHECK(
      indices_.device().type() == kMUSA,
      "Device of indices_ tensor of embedding_bag must be "
      "MUSA, but now is ",
      indices_.device());
  TORCH_CHECK(
      offsets_.device().type() == kMUSA,
      "Device of offsets_ tensor of embedding_bag must be "
      "MUSA, but now is ",
      offsets_.device());

  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);
  auto indices_arg = TensorArg(indices, "indices", 1);
  checkScalarTypes("embedding_bag", indices_arg, {kLong, kInt});
  checkContiguous("embedding_bag", indices_arg);
  auto offsets_arg = TensorArg(offsets, "offsets", 1);
  checkScalarTypes("embedding_bag", offsets_arg, {kLong, kInt});
  checkSameType("embedding_bag", indices_arg, offsets_arg);
  checkContiguous("embedding_bag", offsets_arg);

  Tensor offset2bag_;
  if (indices.sym_numel() != 0 && offset2bag.sym_numel() == 0) {
    offset2bag_ = offsets.new_zeros(
        {indices.size(0) + 1}, offsets.options()); // offset2bag = [0 0 0 0 0]

    make_offset2bag(offsets, offset2bag_);
    // For Composite Compliance, if `offset2bag_` is CCT
    // then we can't call `resize_`. Instead we call `narrow`
    // to slice the tensor.
    if (isTensorSubclassLike(offset2bag_)) {
      offset2bag_ = offset2bag_.narrow(0, 0, indices.size(0));
    } else {
      offset2bag_.resize_({indices.size(0)});
    }
  } else {
    auto offset2bag_arg = TensorArg(offset2bag, "offset2bag", 1);
    checkScalarTypes("embedding_bag", offset2bag_arg, {kLong, kInt});
    checkContiguous("embedding_bag", offset2bag_arg);
    offset2bag_ = offset2bag;
  }

  if (sparse) {
    return at::_embedding_bag_sparse_backward_symint(
        grad,
        indices,
        offsets,
        offset2bag_,
        bag_size_,
        std::move(num_weights),
        scale_grad_by_freq,
        mode,
        per_sample_weights,
        padding_idx);
  } else {
    return at::_embedding_bag_dense_backward_symint(
        grad,
        indices,
        offset2bag_,
        bag_size_,
        max_indices_,
        std::move(num_weights),
        scale_grad_by_freq,
        mode,
        per_sample_weights,
        padding_idx);
  }
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

  int64_t numBags = offsets_.size(0);
  if (include_last_offset) {
    TORCH_CHECK(
        numBags >= 1, "include_last_offset: numBags should be at lease 1");
    numBags -= 1;
  }
  int64_t featureSize = weight.size(1);
  Tensor indices, offsets;
  std::tie(indices, offsets) = promoteIndicesAndOffsets(indices_, offsets_);

  Tensor output = at::empty(
      {include_last_offset ? offsets.size(0) - 1 : offsets.size(0),
       weight.sizes()[1]},
      weight.options());

  Tensor bag_size = at::empty(offsets.sizes(), offsets.options());
  at::native::make_bag_size_out(
      bag_size, offsets, indices, mode, include_last_offset, requires_grad);

  Tensor max_indices;
  if (mode == MODE_MAX) {
    max_indices = at::empty({numBags, featureSize}, indices.options());
  } else {
    max_indices = at::empty(bag_size.sizes(), indices.options());
  }

  // no need ?
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

static Tensor apply_bag_size_backward(
    const int64_t mode,
    Tensor& output,
    const Tensor& offset2bag,
    const Tensor& bag_size) {
  if (mode == native::EmbeddingBagMode::MEAN) {
    auto inv_bag_size_ = (1 / bag_size.to(output.options()))
                             .unsqueeze(1)
                             .index_select(0, offset2bag);
    output *= inv_bag_size_;
  }
  return output;
}

Tensor _embedding_bag_sparse_backward_symint(
    const Tensor& grad_,
    const Tensor& indices,
    [[maybe_unused]] const Tensor& offsets,
    const Tensor& offset2bag,
    const Tensor& bag_size_,
    SymInt num_weights,
    bool scale_grad_by_freq,
    int64_t mode,
    const std::optional<Tensor>& per_sample_weights_opt,
    int64_t padding_idx) {
  // See [Note: hacky wrapper removal for optional tensor]
  c10::MaybeOwned<Tensor> per_sample_weights_maybe_owned =
      at::borrow_from_optional_tensor(per_sample_weights_opt);
  const Tensor& per_sample_weights = *per_sample_weights_maybe_owned;

  // indices, offsets and offset2bag are assumed having correct dtypes and
  // contiguous here due to the checks in _embedding_bag_backward above.
  // Also see NOTE [ embedding_bag Native Functions ] in native_functions.yaml
  // for more details.

  Tensor index_grad = grad_.index_select(0, offset2bag);

  index_grad = apply_bag_size_backward(mode, index_grad, offset2bag, bag_size_);

  if (per_sample_weights.defined()) {
    AT_ASSERT(mode == native::EmbeddingBagMode::SUM);
    index_grad.mul_(per_sample_weights.unsqueeze(1));
  }
  return native::embedding_backward_symint(
      index_grad,
      indices,
      std::move(num_weights),
      padding_idx,
      scale_grad_by_freq,
      true);
}

} // namespace musa
} // namespace at
