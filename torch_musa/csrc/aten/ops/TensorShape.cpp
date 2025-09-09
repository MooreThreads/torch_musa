#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/_index_put_impl_ops.h>
#include <ATen/ops/_indices_copy_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/expand_as_native.h>
#include <ATen/ops/expand_copy_native.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/indices_copy_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/unsqueeze_copy_native.h>
#include <ATen/ops/unsqueeze_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/ops/TensorShape.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

DEFINE_DISPATCH(indexselect_stub);
REGISTER_NO_CPU_DISPATCH(indexselect_stub);

static bool all_strides_match(TensorList tensors) {
  TORCH_CHECK(!tensors.empty());
  auto strides = tensors[0].strides();
  for (auto& tensor : tensors.slice(1)) {
    if (!strides.equals(tensor.strides())) {
      return false;
    }
  }
  return true;
}

// Add dimensions of size 1 to an index tensor so that it can be broadcast to
// the result shape and iterated over element-wise like the result tensor and
// the restrided src.
static Tensor reshape_indexer(
    const Tensor& index,
    int64_t dims_before,
    int64_t dims_after) {
  auto orig_shape = index.sizes();
  auto shape = DimVector();
  shape.append(dims_before, 1);
  shape.append(orig_shape.begin(), orig_shape.end());
  shape.append(dims_after, 1);
  return index.reshape(shape);
}

static Tensor restride_src(
    const Tensor& src,
    int64_t dims_before,
    int64_t dims_indexed,
    IntArrayRef replacement_shape) {
  auto shape = DimVector(src.sizes());
  auto strides = DimVector(src.strides());
  int64_t end = dims_before + dims_indexed;
  shape.erase(shape.begin() + dims_before, shape.begin() + end);
  strides.erase(strides.begin() + dims_before, strides.begin() + end);
  shape.insert(
      shape.begin() + dims_before,
      replacement_shape.begin(),
      replacement_shape.end());
  strides.insert(strides.begin() + dims_before, replacement_shape.size(), 0);
  return src.as_strided(shape, strides);
}

AdvancedIndex::AdvancedIndex(const Tensor& src, TensorList indices_list) {
  int64_t element_size_bytes = src.element_size();
  int64_t dims_before = 0, dims_after = 0, dims_indexed = 0;
  IntArrayRef replacement_shape;
  for (const auto dim : c10::irange(indices_list.size())) {
    if (!indices_list[dim].defined()) {
      if (dims_indexed == 0) {
        dims_before++;
      } else {
        dims_after++;
      }
    } else {
      dims_indexed++;
      replacement_shape = indices_list[dim].sizes();
      indexed_sizes.push_back(src.size(dim));
      indexed_strides.push_back(src.stride(dim) * element_size_bytes);
    }
  }

  // Check if the indexed subspace contains a dim of size 0, but the replacement
  // shape does not. This implies that an index is out of bounds, because there
  // is no number that's a valid index for an empty tensor. Normally, out of
  // bounds is handled in the indexing kernel, but this case fails earlier in
  // restride_src with an unhelpful error message.
  if (std::find(indexed_sizes.begin(), indexed_sizes.end(), 0) !=
          indexed_sizes.end() &&
      std::find(replacement_shape.begin(), replacement_shape.end(), 0) ==
          replacement_shape.end()) {
    TORCH_CHECK_INDEX(
        false, "index is out of bounds for dimension with size 0");
  }

  this->dims_before = dims_before;
  this->dims_after = dims_after;
  this->src = restride_src(src, dims_before, dims_indexed, replacement_shape);

  for (auto& index : indices_list) {
    if (index.defined()) {
      indices.push_back(reshape_indexer(index, dims_before, dims_after));
    }
  }

  // For MUSA tensors, force all index tensors to have the same striding to
  // simplify the MUSA kernel.
  if (indices.size() >= 2 && this->src.device().type() == kMUSA) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

static TensorIterator make_index_put_iterator(
    const AdvancedIndex& info,
    const Tensor& value) {
  TORCH_CHECK(
      is_expandable_to(value.sizes(), info.src.sizes()),
      "shape mismatch: value tensor of shape ",
      value.sizes(),
      " cannot be broadcast to indexing result of shape ",
      info.src.sizes());
  TORCH_CHECK(
      value.scalar_type() == info.src.scalar_type(),
      "Index put requires the source and destination dtypes match, "
      "got ",
      info.src.scalar_type(),
      " for the destination "
      "and ",
      value.scalar_type(),
      " for the source.");
  TensorIteratorConfig config;
  // info.src is restrided by restride_src with 0 strided dimensions
  config.set_check_mem_overlap(false);
  config.resize_outputs(false);
  config.check_all_same_dtype(false);
  config.add_output(info.src);
  config.add_input(value);
  for (auto& index : info.indices) {
    config.add_input(index);
  }
  return config.build();
}

} // namespace native

namespace musa {

Tensor& IndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  uint64_t numIndices = index.numel();
  dim = at::maybe_wrap_dim(dim, self);
  std::vector<int64_t> newSize = self.sizes().vec();
  if (self.dim() > 0) {
    newSize[dim] = numIndices;
  }
  if (self.is_quantized()) {
    out = at::empty_quantized(newSize, out);
  } else {
    at::native::resize_output(out, newSize);
  }
  if (out.numel() == 0) {
    return out;
  }

  TORCH_CHECK(
      self.scalar_type() != at::ScalarType::QInt32,
      "Unsupported IndexSelect input dtype: ",
      self.scalar_type());
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Int ||
          index.scalar_type() == at::ScalarType::Long,
      "Unsupported IndexSelect index dtype: ",
      index.scalar_type());
  TORCH_CHECK(index.device().type() == kMUSA, "Index should be on GPU device");
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor contiguous_self = self.contiguous();
  Tensor contiguous_other = index.contiguous();

  at::native::indexselect_stub(
      kMUSA, dim, out, contiguous_other, contiguous_self);
  return out;
}

Tensor IndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  auto out_shape = std::vector<int64_t>(self.sizes().vec());
  int64_t index_len = index.numel();
  dim = (dim + self.dim()) % self.dim();
  out_shape[dim] = index_len;
  Tensor out;
  if (self.is_quantized()) {
    out = at::empty_quantized(
        out_shape, self, self.options().dtype(self.scalar_type()));
  } else {
    out = at::empty(
        out_shape,
        self.options()
            .dtype(self.scalar_type())
            .memory_format(at::MemoryFormat::Contiguous));
  }
  out = IndexSelectOut(self, dim, index, out);
  return out;
}

Tensor& IndexPut(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  const OptionalDeviceGuard device_guard(device_of(self));
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch =
        at::native::canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_ = value;
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    value_ = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  // for device check and broadcast, we use tensor_iterator to warp input
  // tensors here.
  auto info = at::native::make_info(self, indices);
  auto iter = at::native::make_index_put_iterator(info, value_);
  at::native::index_put_stub(
      iter.device_type(),
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      accumulate);
  return self;
}

} // namespace musa
} // namespace at
