#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/List.h>
#include <ATen/native/IndexingUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

// Restore disabled warnings
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

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

// borrowed from Aten/native/TensorAdvancedIndexing.cpp

extern Tensor as_strided_tensorimpl(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_);

extern Tensor _reshape_alias(
    const Tensor& self,
    IntArrayRef sizes,
    IntArrayRef strides);

extern Tensor view(const Tensor& self, IntArrayRef size);

extern Tensor unfold(
    const Tensor& self,
    int64_t dimension,
    int64_t size,
    int64_t step);

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
            invalid_mask(self, srcIdx, index, j);
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
  checkIndexTensorTypes(orig);
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
  TORCH_CHECK(dim < self.dim() && dim >= 0, "dim is invalid.");
  auto idx_mt = CreateMUTensor(index);
  auto in = CreateMUTensor(self);
  auto out_mt = CreateMUTensor(out);
  muHandle h;
  ::musa::dnn::IndexSelect op;
  CHECK_MUDNN_STATUS(op.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(op.Run(h, out_mt, idx_mt, in), "Run");
}

void IndexCall(
    const Tensor& self,
    const std::vector<muTensor>& indexes,
    Tensor& out) {
  auto in = CreateMUTensor(self);
  auto out_mt = CreateMUTensor(out);
  muHandle h;
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
  muHandle h;
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
  Tensor out = empty_mtgpu(
      out_shape,
      contiguous_self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
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
    return empty_mtgpu(
        out_shape,
        self.scalar_type(),
        c10::nullopt,
        kMUSA,
        c10::nullopt,
        at::MemoryFormat::Contiguous);
  }
  if (dims.size() == 1 && indices[dims[0]].dim() == 1) {
    return at::index_select(self, dims[0], indices[dims[0]]);
  }
  out = empty_mtgpu(
      out_shape,
      self.scalar_type(),
      c10::nullopt,
      kMUSA,
      c10::nullopt,
      at::MemoryFormat::Contiguous);
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
      self.copy_(contiguous_self.to("musa"));
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
      cgs_tensors[idx] = index->to("musa");
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
  m.impl("as_strided", &as_strided_tensorimpl);
  m.impl("view", &view);
  m.impl("_reshape_alias", &_reshape_alias);
  m.impl("index_select", &IndexSelect);
  m.impl("index_select.out", &IndexSelectOut);
  m.impl("index.Tensor", &IndexTensor);
  m.impl("_index_put_impl_", &IndexPut);
  m.impl("unfold", &unfold);
}

} // namespace musa
} // namespace native
} // namespace at
