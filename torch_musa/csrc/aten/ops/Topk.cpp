#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {
namespace native {
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
      self.scalar_type() == at::ScalarType::Float,
      "Dtype of input tensor of topk only support Float32, but "
      "now it is ",
      self.scalar_type());

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

  auto self_contiguous = Contiguous(self);
  auto mt_input = CreateMUTensor(self_contiguous);
  muTensor mt_values;
  muTensor mt_indices;
  muTensor mt_values_sorted;
  muTensor mt_indices_sorted;
  auto values_not_use = at::empty_like(values);
  auto indices_not_use = at::empty_like(indices);
  auto indices_tmp = at::empty_like(indices);
  auto mt_indices_tmp = CreateMUTensor(indices_tmp);
  if (sorted && k > 1) {
    // when sorted=True, the results value/indices of topk are carried by
    // mt_values_sorted/mt_indices_sorted。
    mt_values_sorted = CreateMUTensor(values);
    mt_indices_sorted = CreateMUTensor(indices);
    mt_values = CreateMUTensor(values_not_use);
    mt_indices = CreateMUTensor(indices_not_use);
  } else {
    mt_values_sorted = CreateMUTensor(values_not_use);
    mt_indices_sorted = CreateMUTensor(indices_not_use);
    mt_values = CreateMUTensor(values);
    mt_indices = CreateMUTensor(indices);
  }

  muHandle h;
  ::musa::dnn::TopK mTopk;
  CHECK_MUDNN_STATUS(mTopk.SetK(k), "SetK");
  CHECK_MUDNN_STATUS(mTopk.SetDim(wraped_dim), "SetDim");
  CHECK_MUDNN_STATUS(mTopk.SetLargest(largest), "SetLargest");
  CHECK_MUDNN_STATUS(mTopk.SetSorted(sorted), "SetSorted");

  CHECK_MUDNN_STATUS(
      mTopk.Run(
          h,
          mt_values,
          mt_indices,
          mt_input,
          mt_values_sorted,
          mt_indices_tmp,
          mt_indices_sorted,
          InternalMemAlloc),
      "Run");
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

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("topk", &Topk);
  m.impl("topk.values", &TopkOut);
}

} // namespace musa
} // namespace native
} // namespace at
