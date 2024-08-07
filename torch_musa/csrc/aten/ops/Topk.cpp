#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace musa {

using Status = ::musa::dnn::Status;

std::tuple<Tensor&, Tensor&> TopkOut(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted,
    Tensor& values,
    Tensor& indices) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of TopK must be MTGPU, but now is ",
      self.device());
  TORCH_CHECK(
      values.device().type() == kMUSA,
      "Device of value tensor of TopK must be MTGPU, but now is ",
      values.device());
  TORCH_CHECK(
      indices.device().type() == kMUSA,
      "Device of indices tensor of TopK must be MTGPU, but now is ",
      indices.device());
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half ||
          self.scalar_type() == at::ScalarType::BFloat16,
      "Dtype of input tensor of topk only support Float32/Half/BFloat16, but "
      "now it is ",
      self.scalar_type());

  c10::musa::MUSAGuard device_guard(self.device());
  int64_t wraped_dim = maybe_wrap_dim(dim, self.dim(), /*wrap_scalar=*/true);
  TORCH_CHECK(
      k >= 0 && k <= (self.dim() > 0 ? self.size(wraped_dim) : 1),
      "selected index k out of range");
  int64_t slice_size = self.dim() == 0 ? 1 : self.size(wraped_dim);
  TORCH_CHECK(k >= 0 && k <= slice_size, "k not in range for dimension");

  // Build the output size, which is the dim being selected set to
  // size k
  DimVector topk_size(self.sizes().vec());
  if (topk_size.size() > 0) {
    topk_size[wraped_dim] = k;
  }
  values.resize_(topk_size);
  indices.resize_(topk_size);

  if (self.numel() == 0 || k == 0) {
    return std::forward_as_tuple(values, indices);
  }

  auto self_contiguous = FormatContiguous(self, MemoryFormat::Contiguous);

  auto mt_input = CreateMUTensor(self_contiguous);
  muTensor mt_values = CreateMUTensor(values);
  muTensor mt_indices = CreateMUTensor(indices);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::TopK mTopk;
  CHECK_MUDNN_STATUS(mTopk.SetK(k), "SetK");
  CHECK_MUDNN_STATUS(mTopk.SetDim(wraped_dim), "SetDim");
  CHECK_MUDNN_STATUS(mTopk.SetLargest(largest), "SetLargest");
  CHECK_MUDNN_STATUS(mTopk.SetSorted(sorted), "SetSorted");

  CHECK_MUDNN_STATUS(
      mTopk.Run(h, mt_values, mt_indices, mt_input, InternalMemAlloc), "Run");
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> Topk(
    const Tensor& self,
    int64_t k,
    int64_t dim,
    bool largest,
    bool sorted) {
  auto values =
      at::empty({0}, self.options().memory_format(MemoryFormat::Contiguous));
  auto indices = at::empty(
      {0}, self.options().memory_format(MemoryFormat::Contiguous).dtype(kLong));
  TopkOut(self, k, dim, largest, sorted, values, indices);
  return std::forward_as_tuple(values, indices);
}

} // namespace musa
} // namespace at
