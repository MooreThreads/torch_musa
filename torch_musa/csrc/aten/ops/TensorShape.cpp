#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/List.h>
#include <ATen/native/IndexingUtils.h>
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

void IndexCall(
    const Tensor& self,
    const std::vector<muTensor>& indexes,
    Tensor& out) {
  c10::musa::MUSAGuard device_guard(self.device());
  auto in = CreateMUTensor(self);
  auto out_mt = CreateMUTensor(out);
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Index op;
  CHECK_MUDNN_STATUS(
      op.Run(h, out_mt, indexes.size(), indexes.data(), in), "Index Run");
}

void IndexPutCall(
    Tensor& out,
    const std::vector<muTensor>& indexes,
    const Tensor& value,
    bool accumulate) {
  auto out_mt = CreateMUTensor(out, false);
  auto v = CreateMUTensor(value);
  TORCH_CHECK(
      value.scalar_type() != at::ScalarType::Long || accumulate == false,
      "index_put_ not support int64 when accumulate = true.but they are type: ",
      value.scalar_type(),
      ", accumulate:",
      accumulate);
  c10::musa::MUSAGuard device_guard(value.device());
  muHandle& h = GetMudnnHandle();
  ::musa::dnn::IndexPut op;
  CHECK_MUDNN_STATUS(op.SetAccumulate(accumulate), "Set Accumulate");
  // TODO(kang.chen): muDNN not provides SetOffset interface now,comment out
  // temporarily，
  // when migrate all ops to see if there's any impact.
  // CHECK_MUDNN_STATUS(op.SetOffset(out.storage_offset()), "Set Offset");
  CHECK_MUDNN_STATUS(
      op.Run(h, out_mt, indexes.size(), indexes.data(), v), "IndexPut Run");
}

Tensor& IndexSelectOut(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    Tensor& out) {
  Tensor contiguous_self = Contiguous(self);
  Tensor contiguous_other = Contiguous(index);
  TORCH_CHECK(
      dim < contiguous_self.dim() && dim >= -contiguous_self.dim(),
      "dim is invalid.");
  dim = (dim + contiguous_self.dim()) % contiguous_self.dim();
  IndexSelectCall(contiguous_self, dim, contiguous_other, out);
  return out;
}

Tensor IndexSelect(const Tensor& self, int64_t dim, const Tensor& index) {
  Tensor contiguous_self = Contiguous(self);
  Tensor contiguous_index = Contiguous(index);
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
  bool is_mask = false;
  if (self.numel() == 0) {
    return self;
  }
  auto indices = make_indices(self, orig, is_mask);

  std::vector<muTensor> tensors;
  std::vector<int> dims;
  bool is_one_element = false;
  Tensor out;

  for (size_t i = 0; i < indices.size(); i++) {
    muTensor indice;
    if (indices[i].numel() > 0) {
      // TODO(songtao.liu): where the index is a mask with only one true
      // element, mudnn doesn't work. I cannot figure where is the bug and use
      // select here instead.
      if ((indices[i].numel() == 1 || is_one_element) && is_mask) {
        if (!is_one_element) {
          out = self.clone();
          is_one_element = true;
        }
        TORCH_CHECK(
            indices[i].numel() == 1,
            "deprecated implementation for only one element indexing");
        out = out.select(0, indices[i].item().toLong());
      } else {
        dims.push_back(i);
        indices[i] = Contiguous(indices[i]);
        indice = CreateMUTensor(indices[i]);
      }
    } else {
      indice = muTensor();
    }
    tensors.emplace_back(indice);
  }

  if (is_one_element) {
    return tensors.empty() ? out : out.unsqueeze(0);
  }

  auto out_shape = compute_shapes(self, indices);
  if (!(dims.size())) {
    // when dim.size() == 0, out_shape = in_shape expected out_shape[dim] = 0
    return at::empty(
        out_shape, self.options().memory_format(at::MemoryFormat::Contiguous));
  }
  if (dims.size() == 1 && indices[dims[0]].dim() == 1) {
    return at::index_select(self, dims[0], indices[dims[0]]);
  }
  out = at::empty(
      out_shape, self.options().memory_format(at::MemoryFormat::Contiguous));
  Tensor contiguous_self = Contiguous(self);
  Tensor contiguous_out = Contiguous(out);
  IndexCall(contiguous_self, tensors, contiguous_out);
  return contiguous_out;
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
  c10::musa::MUSAGuard device_guard(self.device());

  if (indices[0].has_value()) {
    if (indices[0]->scalar_type() == ScalarType::Bool) {
      auto contiguous_self = self.to("cpu");
      torch::List<c10::optional<Tensor>> indices_;
      for (const c10::optional<Tensor>& index : indices) {
        auto contiguous_index = c10::optional<Tensor>(index.value().to("cpu"));
        indices_.push_back(contiguous_index);
      }
      auto contiguous_value = value.to("cpu");
      at::_index_put_impl_(
          contiguous_self, indices_, contiguous_value, accumulate, unsafe);
      self.copy_(contiguous_self.to(value.device()));
      return self;
    }
  }

  // borrowed from TensorAdvacedIndexing.cpp
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
  at::assert_no_overlap(self, value);
  Tensor contiguous_value = Contiguous(value);
  if (value.device() != self.device() && value.numel() == 1 &&
      value.dim() == 0) {
    contiguous_value = value.to(self.device());
  }
  std::vector<muTensor> tensors;
  std::vector<Tensor> cgs_tensors(indices.size());
  int idx = 0;
  for (const c10::optional<Tensor>& index : indices) {
    muTensor indice;
    if (index.has_value()) {
      at::assert_no_overlap(self, *index);
      cgs_tensors[idx] = index->to(value.device());
      auto contiguous_index = Contiguous(cgs_tensors[idx]);
      indice = CreateMUTensor(contiguous_index);
      tensors.emplace_back(indice);
      idx++;
    } else {
      TORCH_CHECK(false, "Only support continue index tensors now in mtPytorch")
    }
  }
  auto contiguosu_self = Contiguous(self);
  IndexPutCall(contiguosu_self, tensors, contiguous_value, accumulate);
  self.copy_(contiguosu_self);
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
