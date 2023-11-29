#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/List.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/IndexKernel.h>
#include <ATen/native/IndexingUtils.h>
#include <ATen/native/TensorAdvancedIndexing.h>
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_native.h>
#include <c10/util/irange.h>
#include <torch/library.h>

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
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace native {

DEFINE_DISPATCH(indexput_stub);
REGISTER_NO_CPU_DISPATCH(indexput_stub);
DEFINE_DISPATCH(indexes_stub);
REGISTER_NO_CPU_DISPATCH(indexes_stub);
DEFINE_DISPATCH(indexselect_stub);
REGISTER_NO_CPU_DISPATCH(indexselect_stub);

// borrowed from Aten/native/TensorAdvancedIndexing.cpp
// directly include cpp file will cause undefined symbols error.
static std::string shapes_as_str(TensorList tensors) {
  std::ostringstream os;
  bool first = true;
  for (auto& tensor : tensors) {
    if (tensor.defined()) {
      if (!first) {
        os << ", ";
      }
      os << tensor.sizes();
      first = false;
    }
  }
  return os.str();
}

static C10_UNUSED std::vector<Tensor> ExpandTensorsMusa(
    const Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    bool& is_mask) {
  // If indices come in as ByteTensor or BoolTensor (masks), expand them into
  // the equivalent indexing by LongTensors
  std::vector<Tensor> result;
  for (c10::optional<Tensor> index_opt : indices) {
    if (!index_opt.has_value()) {
      result.emplace_back();
    } else {
      Tensor index = std::move(*index_opt);
      if (index.scalar_type() == kByte || index.scalar_type() == kBool) {
        if (index.scalar_type() == kByte) {
          TORCH_WARN(
              "indexing with dtype torch.uint8 is now deprecated,"
              " please use a dtype torch.bool instead.");
        }
        is_mask = true;
        // The sizes of the ByteTensor mask or bool tensor must match the sizes
        // of the corresponding dimensions in self
        for (const auto j : c10::irange(index.dim())) {
          int64_t src_idx = result.size() + j;
          if (index.size(j) != self.size(src_idx)) {
            at::native::invalid_mask(self, src_idx, index, j);
          }
        }
        // Replace with nonzeros
        auto nonzero = index.nonzero();
        for (const auto j : c10::irange(index.dim())) {
          result.emplace_back(nonzero.select(1, j));
        }
      } else {
        result.emplace_back(std::move(index));
      }
    }
  }
  return result;
}

std::vector<Tensor> MakeIndices(
    Tensor self,
    const torch::List<c10::optional<at::Tensor>>& orig,
    bool& is_mask) {
  at::native::checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = ExpandTensorsMusa(self, orig, is_mask);
  // next broadcast all index tensors together
  try {
    indices = expand_outplace(indices);

    // after expand indices shape to be same, need Contiguous check !!!
    for (auto& indice : indices) {
      indice = indice.contiguous();
      if (indice.numel() > 0 && indice.device() != self.device()) {
        indice = indice.to(self.device());
      }
    }
  } catch (std::exception& e) {
    TORCH_CHECK_INDEX(
        false,
        "shape mismatch: indexing tensors could not be broadcast together"
        " with shapes ",
        at::native::shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < static_cast<size_t>(self.dim())) {
    indices.emplace_back();
  }
  return indices;
}

static std::tuple<bool, Tensor> CanDispatchToMaskedFill(
    const Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices,
    const Tensor& value) {
  if (!(value.numel() == 1 && value.device().is_cpu())) {
    return std::make_tuple(false, Tensor());
  }
  int64_t num_idx = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor> i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      num_idx++;
    } else {
      Tensor index = std::move(*i);
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t src_idx = num_idx + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(src_idx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              src_idx);
        }
        num_idx += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_idx, self.ndimension())) {
    (void)i; // Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
}

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
  if (indices.size() >= 2 &&
      (this->src.device().type() == kMUSA ||
       this->src.device().type() == kMPS)) {
    if (!all_strides_match(indices)) {
      for (auto& indice : indices) {
        indice = indice.contiguous();
      }
    }
  }
}

} // namespace native

namespace musa {

inline void IndicesDTypeCheck(
    const std::vector<c10::optional<Tensor>>& indices) {
  for (const auto& ind : indices) {
    if (ind.has_value() && ind.value().numel() > 0) {
      TORCH_CHECK(
          ind.value().scalar_type() == at::ScalarType::Int ||
              ind.value().scalar_type() == at::ScalarType::Long ||
              ind.value().scalar_type() == at::ScalarType::Bool,
          "indices dtype should be int32/64 or bool");
    }
  }
}

std::vector<int64_t> compute_shapes(Tensor self, std::vector<Tensor> indices) {
  std::vector<int64_t> output_dims;
  auto self_dims = self.sizes().vec();
  auto indices_num = indices.size();
  std::vector<int64_t> indice_size;

  // check if defined indice has been calculated
  bool has_defined = false;

  // calculate output dims for indices
  for (size_t j = 0; j < indices_num; ++j) {
    if (indices[j].defined()) {
      if (!has_defined) {
        indice_size = indices[j].sizes().vec();
        output_dims.insert(
            output_dims.end(), indice_size.begin(), indice_size.end());
        has_defined = true;
      }
    } else {
      output_dims.emplace_back(self_dims[j]);
    }
  }
  return output_dims;
}

Tensor& IndexSelectOutPorting(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, out, "wrapper_CUDA_out_index_select_out", "out");
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA_out_index_select_out", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_CUDA_out_index_select_out", "index");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_out_cuda(self, dim, index, out);
}

Tensor& IndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  // TODO(@mt-ai): IndexSelect kernel doesn't support Short and Byte dtypes
  if (self.scalar_type() == at::ScalarType::Short ||
      self.scalar_type() == at::ScalarType::Byte) {
    return IndexSelectOutPorting(self, dim, index, out);
  }
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Double ||
          self.scalar_type() == at::ScalarType::Int ||
          self.scalar_type() == at::ScalarType::Long ||
          self.scalar_type() == at::ScalarType::BFloat16,
      "Unsupported IndexSelect input dtype: ",
      self.scalar_type());
  TORCH_CHECK(
      index.scalar_type() == at::ScalarType::Int ||
          index.scalar_type() == at::ScalarType::Long,
      "Unsupported IndexSelect index dtype: ",
      index.scalar_type());
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor contiguous_self = self.contiguous();
  Tensor contiguous_other = index.contiguous();
  TORCH_CHECK(
      dim < contiguous_self.dim() && dim >= -contiguous_self.dim(),
      "dim is invalid.");
  dim = (dim + contiguous_self.dim()) % contiguous_self.dim();

  at::native::indexselect_stub(
      kMUSA, dim, out, contiguous_other, contiguous_self);
  return out;
}

Tensor IndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  auto out_shape = std::vector<int64_t>(self.sizes().vec());
  int64_t index_len = index.numel();
  dim = (dim + self.dim()) % self.dim();
  out_shape[dim] = index_len;
  Tensor out = at::empty(
      out_shape,
      self.options()
          .dtype(self.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  out = IndexSelectOut(self, dim, index, out);
  return out;
}

Tensor IndexTensor(
    const Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& orig) {
  TORCH_CHECK_INDEX(
      orig.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      orig.size(),
      ")");
  IndicesDTypeCheck(orig.vec());
  bool is_mask = false;

  // For self.numel() == 0, we will still broadcast it when indices
  // are provided.
  std::vector<Tensor> indices = at::native::MakeIndices(self, orig, is_mask);
  std::vector<int> dims;

  int i = 0;
  for (const auto& indice : indices) {
    if (indice.numel() > 0) {
      dims.push_back(i);
    }
    i++;
  }

  auto out_shape = compute_shapes(self, indices);
  if (!dims.size()) {
    // when dim.size() == 0, out_shape = in_shape expected out_shape[dim] = 0
    return at::empty(out_shape, self.options());
  }
  if (dims.size() == 1 && indices[dims[0]].dim() == 1) {
    return IndexSelect(self, dims[0], indices[dims[0]]);
  }
  Tensor out = at::empty(out_shape, self.options());
  at::native::indexes_stub(kMUSA, out, indices.size(), indices, self);

  return out;
}

Tensor& IndexPut(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
  // Note: Tensors in "indices" are not on the same device, which is allowed.
  // such as: self-cpu, indices0-cpu, indices1-musa, value-cpu
  if (self.device() == DeviceType::CPU) {
    torch::List<c10::optional<Tensor>> indices_cpu;
    // Ensure indices are on the same device as self
    for (const c10::optional<Tensor>& index : indices) {
      auto index_cpu = c10::optional<Tensor>(index.value().to("cpu"));
      indices_cpu.push_back(index_cpu);
    }
    at::_index_put_impl_(self, indices_cpu, value, accumulate, unsafe);
    return self;
  }

  if (self.numel() == 0) {
    return self;
  }

  // borrowed from TensorAdvacedIndexing.cpp
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK_INDEX(
      indices.size() <= (size_t)self.dim(),
      "too many indices for tensor of dimension ",
      self.dim(),
      " (got ",
      indices.size(),
      ")");
  IndicesDTypeCheck(indices.vec());
  if (at::has_internal_overlap(self) == MemOverlap::Yes) {
    TORCH_WARN(
        "Use of index_put_ on expanded tensors is deprecated. "
        "Please clone() the tensor before performing this operation. "
        "This also applies to advanced indexing e.g. tensor[indices] = tensor");
  }
  if (!accumulate) {
    auto masked_fill_dispatch =
        at::native::CanDispatchToMaskedFill(self, indices, value);
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
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }

  bool is_mask = false;
  std::vector<Tensor> cgs_indices =
      at::native::MakeIndices(self, indices, is_mask);

  // IndexPut kernel doesn't support uncontiguous input yet.
  Tensor cgs_self = Contiguous(self);
  at::native::indexput_stub(kMUSA, cgs_self, cgs_indices, value_, accumulate);
  self.copy_(cgs_self);

  return self;
}

REGISTER_IMPL(
    aten,
    PrivateUse1,
    "as_strided",
    at::native::as_strided_tensorimpl,
    at_native_as_strided_tensorimpl)
REGISTER_IMPL(aten, PrivateUse1, "view", at::native::view, at_native_view)
REGISTER_IMPL(
    aten,
    PrivateUse1,
    "_reshape_alias",
    at::native::_reshape_alias,
    at_native__reshape_alias)
ADVANCED_REGISTER(aten, PrivateUse1, "index_select", IndexSelect)
ADVANCED_REGISTER(aten, PrivateUse1, "index_select.out", IndexSelectOut)
ADVANCED_REGISTER(aten, PrivateUse1, "index.Tensor", IndexTensor)
ADVANCED_REGISTER(aten, PrivateUse1, "_index_put_impl_", IndexPut)
REGISTER_IMPL(aten, PrivateUse1, "unfold", at::native::unfold, at_native_unfold)

} // namespace musa
} // namespace at
