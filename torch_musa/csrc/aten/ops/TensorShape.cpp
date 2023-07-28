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
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/view_native.h>
#include <c10/util/irange.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_conj_copy_native.h>
#include <ATen/ops/_convert_indices_from_coo_to_csr.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo.h>
#include <ATen/ops/_fw_primal_copy_native.h>
#include <ATen/ops/_index_put_impl.h>
#include <ATen/ops/_index_put_impl_ops.h>
#include <ATen/ops/_indices_copy_native.h>
#include <ATen/ops/_make_dual.h>
#include <ATen/ops/_make_dual_copy_native.h>
#include <ATen/ops/_mkldnn_reshape.h>
#include <ATen/ops/_mkldnn_transpose.h>
#include <ATen/ops/_neg_view_copy_native.h>
#include <ATen/ops/_reshape_alias_copy_native.h>
#include <ATen/ops/_reshape_alias_native.h>
#include <ATen/ops/_reshape_from_tensor_native.h>
#include <ATen/ops/_shape_as_tensor_native.h>
#include <ATen/ops/_sparse_broadcast_to.h>
#include <ATen/ops/_sparse_broadcast_to_copy_native.h>
#include <ATen/ops/_sparse_broadcast_to_native.h>
#include <ATen/ops/_sparse_compressed_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_coo_tensor_with_dims_and_tensors.h>
#include <ATen/ops/_sparse_csc_tensor_unsafe_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_stack_native.h>
#include <ATen/ops/_unsafe_view.h>
#include <ATen/ops/_unsafe_view_native.h>
#include <ATen/ops/_values_copy_native.h>
#include <ATen/ops/adjoint_native.h>
#include <ATen/ops/alias.h>
#include <ATen/ops/alias_copy_native.h>
#include <ATen/ops/alias_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/arange_native.h>
#include <ATen/ops/as_strided_copy_native.h>
#include <ATen/ops/as_strided_native.h>
#include <ATen/ops/as_strided_scatter_native.h>
#include <ATen/ops/atleast_1d.h>
#include <ATen/ops/atleast_2d.h>
#include <ATen/ops/atleast_3d.h>
#include <ATen/ops/block_diag_native.h>
#include <ATen/ops/broadcast_tensors_native.h>
#include <ATen/ops/broadcast_to_native.h>
#include <ATen/ops/cat.h>
#include <ATen/ops/cat_meta.h>
#include <ATen/ops/cat_native.h>
#include <ATen/ops/chunk_native.h>
#include <ATen/ops/col_indices_copy_native.h>
#include <ATen/ops/column_stack_native.h>
#include <ATen/ops/concat_native.h>
#include <ATen/ops/concatenate_native.h>
#include <ATen/ops/crow_indices_copy_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/detach_copy_native.h>
#include <ATen/ops/detach_native.h>
#include <ATen/ops/diag.h>
#include <ATen/ops/diag_embed.h>
#include <ATen/ops/diag_embed_native.h>
#include <ATen/ops/diag_native.h>
#include <ATen/ops/diagflat_native.h>
#include <ATen/ops/diagonal.h>
#include <ATen/ops/diagonal_backward.h>
#include <ATen/ops/diagonal_backward_native.h>
#include <ATen/ops/diagonal_copy.h>
#include <ATen/ops/diagonal_copy_native.h>
#include <ATen/ops/diagonal_native.h>
#include <ATen/ops/diagonal_scatter_native.h>
#include <ATen/ops/dsplit_native.h>
#include <ATen/ops/dstack_native.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/empty_quantized.h>
#include <ATen/ops/expand_as_native.h>
#include <ATen/ops/expand_copy_native.h>
#include <ATen/ops/expand_native.h>
#include <ATen/ops/flatten_dense_tensors_native.h>
#include <ATen/ops/flatten_native.h>
#include <ATen/ops/from_blob.h>
#include <ATen/ops/hsplit_native.h>
#include <ATen/ops/hstack.h>
#include <ATen/ops/hstack_native.h>
#include <ATen/ops/index_select.h>
#include <ATen/ops/index_select_native.h>
#include <ATen/ops/indices_copy_native.h>
#include <ATen/ops/lift_fresh_native.h>
#include <ATen/ops/lift_native.h>
#include <ATen/ops/mH_native.h>
#include <ATen/ops/mT_native.h>
#include <ATen/ops/matrix_H_native.h>
#include <ATen/ops/meshgrid_native.h>
#include <ATen/ops/moveaxis_native.h>
#include <ATen/ops/movedim.h>
#include <ATen/ops/movedim_native.h>
#include <ATen/ops/narrow.h>
#include <ATen/ops/narrow_copy.h>
#include <ATen/ops/narrow_copy_native.h>
#include <ATen/ops/narrow_native.h>
#include <ATen/ops/new_empty_native.h>
#include <ATen/ops/new_ones_native.h>
#include <ATen/ops/numpy_T_native.h>
#include <ATen/ops/permute_copy_native.h>
#include <ATen/ops/permute_native.h>
#include <ATen/ops/ravel_native.h>
#include <ATen/ops/repeat_native.h>
#include <ATen/ops/reshape_as_native.h>
#include <ATen/ops/reshape_native.h>
#include <ATen/ops/resize_native.h>
#include <ATen/ops/row_stack_native.h>
#include <ATen/ops/select.h>
#include <ATen/ops/select_backward_native.h>
#include <ATen/ops/select_copy_native.h>
#include <ATen/ops/select_native.h>
#include <ATen/ops/select_scatter_native.h>
#include <ATen/ops/set_native.h>
#include <ATen/ops/slice.h>
#include <ATen/ops/slice_backward_native.h>
#include <ATen/ops/slice_copy_native.h>
#include <ATen/ops/slice_native.h>
#include <ATen/ops/slice_scatter_native.h>
#include <ATen/ops/sparse_coo_tensor.h>
#include <ATen/ops/sparse_coo_tensor_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/split_copy_native.h>
#include <ATen/ops/split_native.h>
#include <ATen/ops/split_with_sizes.h>
#include <ATen/ops/split_with_sizes_copy_native.h>
#include <ATen/ops/split_with_sizes_native.h>
#include <ATen/ops/squeeze.h>
#include <ATen/ops/squeeze_copy_native.h>
#include <ATen/ops/squeeze_native.h>
#include <ATen/ops/stack_native.h>
#include <ATen/ops/sub.h>
#include <ATen/ops/sum.h>
#include <ATen/ops/sum_to_size_native.h>
#include <ATen/ops/swapaxes_native.h>
#include <ATen/ops/swapdims_native.h>
#include <ATen/ops/t_copy_native.h>
#include <ATen/ops/t_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/tensor_split.h>
#include <ATen/ops/tensor_split_native.h>
#include <ATen/ops/tile_native.h>
#include <ATen/ops/transpose.h>
#include <ATen/ops/transpose_copy_native.h>
#include <ATen/ops/transpose_native.h>
#include <ATen/ops/unbind.h>
#include <ATen/ops/unbind_copy_native.h>
#include <ATen/ops/unbind_native.h>
#include <ATen/ops/unflatten_dense_tensors_native.h>
#include <ATen/ops/unflatten_native.h>
#include <ATen/ops/unfold_copy_native.h>
#include <ATen/ops/unfold_native.h>
#include <ATen/ops/unsafe_chunk_native.h>
#include <ATen/ops/unsafe_split_native.h>
#include <ATen/ops/unsafe_split_with_sizes_native.h>
#include <ATen/ops/unsqueeze_copy_native.h>
#include <ATen/ops/unsqueeze_native.h>
#include <ATen/ops/values_copy_native.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_complex_copy_native.h>
#include <ATen/ops/view_as_native.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/view_as_real_copy_native.h>
#include <ATen/ops/view_copy_native.h>
#include <ATen/ops/view_native.h>
#include <ATen/ops/vsplit_native.h>
#include <ATen/ops/vstack.h>
#include <ATen/ops/vstack_native.h>
#include <ATen/ops/zeros.h>
#include <ATen/ops/zeros_like.h>
#include <ATen/ops/zeros_native.h>
#endif

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {

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

namespace native {
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

  // For CUDA/MPS tensors, force all index tensors to have the same striding to
  // simplify the CUDA/MPS kernel.
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

static std::tuple<bool, Tensor> canDispatchToMaskedFill(
    const Tensor& self,
    const torch::List<c10::optional<at::Tensor>>& indices,
    const Tensor& value) {
  if (!(value.numel() == 1 && value.device().is_cpu())) {
    return std::make_tuple(false, Tensor());
  }
  int64_t num_ind = 0;
  Tensor mask;
  auto self_device = self.device();
  for (const c10::optional<Tensor> i : indices) {
    if (!i.has_value() || !(*i).defined()) {
      num_ind++;
    } else {
      Tensor index = std::move(*i);
      if ((index.scalar_type() != kByte && index.scalar_type() != kBool) ||
          index.device() != self_device || mask.defined()) {
        return std::make_tuple(false, Tensor());
      } else {
        mask = index;
        for (const auto j : c10::irange(index.dim())) {
          int64_t srcIdx = num_ind + j;
          TORCH_CHECK_INDEX(
              index.size(j) == self.size(srcIdx),
              "The shape of the mask ",
              index.sizes(),
              " at index ",
              j,
              " does not match the shape of the indexed tensor ",
              self.sizes(),
              " at index ",
              srcIdx);
        }
        num_ind += mask.ndimension();
      }
    }
  }
  for (const auto i : c10::irange(num_ind, self.ndimension())) {
    (void)i; // Suppress unused variable warning
    mask = mask.unsqueeze(-1);
  }
  return std::make_tuple(true, mask);
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

namespace musa {

static C10_UNUSED std::vector<Tensor> expandTensorsMusa(
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
          int64_t srcIdx = result.size() + j;
          if (index.size(j) != self.size(srcIdx)) {
            at::native::invalid_mask(self, srcIdx, index, j);
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

std::vector<Tensor> make_indices(
    Tensor self,
    const torch::List<c10::optional<at::Tensor>>& orig,
    bool& is_mask) {
  at::native::checkIndexTensorTypes(orig);
  // first expand BoolTensor (masks) or ByteTensor (masks) into 1 or more
  // LongTensors
  auto indices = expandTensorsMusa(self, orig, is_mask);
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
        shapes_as_str(indices));
  }
  // add missing null Tensors so that it matches self.dim()
  while (indices.size() < static_cast<size_t>(self.dim())) {
    indices.emplace_back();
  }
  return indices;
}

void IndexSelectCall(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  TORCH_CHECK(dim < self.dim() && dim >= 0, "dim is invalid.");
  auto idx_mt = CreateMUTensor(index);
  auto in = CreateMUTensor(self);
  auto out_mt = CreateMUTensor(out);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::IndexSelect op;
  CHECK_MUDNN_STATUS(op.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(op.Run(h, out_mt, idx_mt, in), "Run");
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

Tensor IndexSelectPorting(
    const Tensor& self,
    int64_t dim,
    const Tensor& index) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, self, "wrapper_CUDA__index_select", "self");
  c10::impl::check_and_update_common_device(
      common_device, index, "wrapper_CUDA__index_select", "index");
  const OptionalDeviceGuard device_guard(device_of(self));
  return at::native::index_select_cuda(self, dim, index);
}

Tensor& IndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  if (self.scalar_type() != at::ScalarType::Float ||
      index.scalar_type() != at::ScalarType::Long) {
    return IndexSelectOutPorting(self, dim, index, out);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor contiguous_self = self.contiguous();
  Tensor contiguous_other = index.contiguous();
  TORCH_CHECK(
      dim < contiguous_self.dim() && dim >= -contiguous_self.dim(),
      "dim is invalid.");
  dim = (dim + contiguous_self.dim()) % contiguous_self.dim();
  IndexSelectCall(contiguous_self, dim, contiguous_other, out);
  return out;
}

Tensor IndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  if (self.scalar_type() != at::ScalarType::Float ||
      index.scalar_type() != at::ScalarType::Long) {
    return IndexSelectPorting(self, dim, index);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  Tensor contiguous_self = self.contiguous();
  Tensor contiguous_index = index.contiguous();
  auto out_shape = std::vector<int64_t>(contiguous_self.sizes().vec());
  int64_t index_len = contiguous_index.numel();
  TORCH_CHECK(
      dim < contiguous_self.dim() && dim >= -contiguous_self.dim(),
      "dim is invalid.");
  dim = (dim + contiguous_self.dim()) % contiguous_self.dim();
  out_shape[dim] = index_len;
  Tensor out = at::empty(
      out_shape,
      self.options()
          .dtype(contiguous_self.scalar_type())
          .memory_format(at::MemoryFormat::Contiguous));
  IndexSelectCall(contiguous_self, dim, contiguous_index, out);
  return out;
}

struct structured_index_out_functional final
    : public at::native::structured_index_out {
  void set_output_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_index_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  void set_output_raw_strided(
      int64_t output_idx,
      IntArrayRef sizes,
      IntArrayRef strides,
      TensorOptions options,
      DimnameList names) override {
    auto current_device = guard_.current_device();
    if (C10_UNLIKELY(current_device.has_value())) {
      TORCH_INTERNAL_ASSERT(
          *current_device == options.device(),
          "structured kernels don't support multi-device outputs");
    } else {
      guard_.reset_device(options.device());
    }
    outputs_[output_idx] = create_out(sizes, strides, options);
    if (!names.empty()) {
      namedinference::propagate_names(*outputs_[output_idx], names);
    }
    // super must happen after, so that downstream can use maybe_get_output
    // to retrieve the output
    at::native::structured_index_out::set_output_raw_strided(
        output_idx, sizes, strides, options, names);
  }
  const Tensor& maybe_get_output(int64_t output_idx) override {
    return *outputs_[output_idx];
  }
  std::array<c10::ExclusivelyOwned<Tensor>, 1> outputs_;
  c10::musa::OptionalMUSAGuard guard_;
};

Tensor IndexTensor(
    const Tensor& self,
    const c10::List<c10::optional<at::Tensor>>& indices) {
  // musa only supports contiguous indices in some case, which cuda doesn't need
  c10::musa::MUSAGuard device_guard(self.device());
  c10::List<c10::optional<at::Tensor>> indices_;
  for (c10::optional<at::Tensor> indice : indices) {
    Tensor indice_tmp = indice.value_or(Tensor());
    Tensor indice_ = indice_tmp.contiguous();
    indices_.push_back(c10::optional<Tensor>(indice_));
  }

  // porting code
  // No device check
  structured_index_out_functional op;
  auto precompute = op.meta(self, at::IOptTensorListRef(indices_));
  (void)precompute;
  op.impl(self, precompute.sizes, precompute.strides, *op.outputs_[0]);
  return std::move(op.outputs_[0]).take();
}

Tensor& IndexPut(
    Tensor& self,
    const torch::List<c10::optional<Tensor>>& indices,
    const Tensor& value,
    const bool accumulate,
    const bool unsafe) {
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
    auto masked_fill_dispatch = canDispatchToMaskedFill(self, indices, value);
    if (std::get<0>(masked_fill_dispatch)) {
      return self.masked_fill_(std::get<1>(masked_fill_dispatch), value.item());
    }
  }
  auto value_tmp = value;
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    value_tmp = value.to(self.device());
  }
  at::assert_no_overlap(self, value);
  // NOLINTNEXTLINE(performance-implicit-conversion-in-loop)
  for (const c10::optional<Tensor>& index : indices) {
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
    }
  }
  if (self.device().type() == kMUSA &&
      (accumulate || globalContext().deterministicAlgorithms())) {
    TORCH_CHECK(
        value_tmp.device() == self.device(),
        "expected device ",
        self.device(),
        " but got device ",
        value_tmp.device(),
        " for value tensor");
    ::at::native::index_put_with_sort_stub(
        self.device().type(), self, indices, value_tmp, accumulate, unsafe);
    return self;
  }

  auto info = at::native::make_info(self, indices);
  auto iter = at::native::make_index_put_iterator(info, value_tmp);
  ::at::native::index_put_stub(
      iter.device_type(),
      iter,
      info.indexed_sizes,
      info.indexed_strides,
      accumulate);
  return self;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", &at::native::as_strided_tensorimpl);
  m.impl("view", &at::native::view);
  m.impl("_reshape_alias", &at::native::_reshape_alias);
  m.impl("index_select", &IndexSelect);
  m.impl("index_select.out", &IndexSelectOut);
  m.impl("index.Tensor", &IndexTensor);
  m.impl("_index_put_impl_", &IndexPut);
  m.impl("unfold", &at::native::unfold);
}

} // namespace musa
} // namespace at
