#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/List.h>
#include <ATen/native/IndexingUtils.h>
#include <c10/util/irange.h>
#include <torch/library.h>

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

Tensor AsStrided(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    optional<int64_t> storage_offset_) {
  return as_strided_tensorimpl(self, size, stride, storage_offset_);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("as_strided", &AsStrided);
  m.impl("view", &view);
  m.impl("_reshape_alias", &_reshape_alias);
  m.impl("unfold", &unfold);
}

} // namespace musa
} // namespace native
} // namespace at
