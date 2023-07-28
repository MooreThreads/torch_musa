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

void SortCall(
    Tensor& values,
    Tensor& indices,
    const Tensor& in,
    int64_t dim,
    bool descending,
    bool stable) {
  c10::musa::MUSAGuard device_guard(in.device());
  auto input_ = CreateMUTensor(in);
  auto values_ = CreateMUTensor(values);
  auto indices_ = CreateMUTensor(indices);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Sort mSort;
  TORCH_CHECK(
      Status::SUCCESS == mSort.SetDim(dim), "Sort set dim param failed");

  TORCH_CHECK(
      Status::SUCCESS == mSort.SetDescending(descending),
      "Sort set descending flag failed");

  TORCH_CHECK(
      Status::SUCCESS == mSort.SetStable(stable),
      "Sort set stable flag failed");

  TORCH_CHECK(
      Status::SUCCESS ==
          mSort.Run(h, values_, indices_, input_, InternalMemAlloc),
      "Sort run kernel failed");
}

std::tuple<Tensor, Tensor> Sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  Tensor values = at::empty(self_.sizes(), self_.options()).copy_(self_);
  Tensor indices = at::empty(self_.sizes(), self_.options().dtype(kLong));

  if (self_.dim() == 0 && self_.numel() == 1) {
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  // default statble is false
  SortCall(values, indices, self_, dim_, descending, false);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> SortOut(
    const Tensor& self,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  values.resize_(self_.sizes()).copy_(self_);
  indices.resize_(self_.sizes());

  if (self_.dim() == 0 && self_.numel() == 1) {
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  // default statble is false
  SortCall(values, indices, self_, dim_, descending, false);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> SortStable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  Tensor values = at::empty(self_.sizes(), self_.options()).copy_(self_);
  Tensor indices = at::empty(self_.sizes(), self_.options().dtype(kLong));

  if (self_.dim() == 0 && self_.numel() == 1) {
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }

  TORCH_INTERNAL_ASSERT(
      stable.has_value(),
      "sort_out(): c10::optional<bool> for stable has to have value.");
  bool stable_ = stable.value();
  SortCall(values, indices, self_, dim_, descending, stable_);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor&, Tensor&> SortStableOut(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  values.resize_(self_.sizes()).copy_(self_);
  indices.resize_(self_.sizes());

  if (self_.dim() == 0 && self_.numel() == 1) {
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  TORCH_INTERNAL_ASSERT(
      stable.has_value(),
      "sort_out(): c10::optional<bool> for stable has to have value.");
  bool stable_ = stable.value();
  SortCall(values, indices, self_, dim_, descending, stable_);
  return std::forward_as_tuple(values, indices);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("sort", &Sort);
  m.impl("sort.values", &SortOut);
  m.impl("sort.stable", &SortStable);
  m.impl("sort.values_stable", &SortStableOut);
}

} // namespace musa
} // namespace at
