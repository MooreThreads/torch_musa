#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/sort.h>
#include <ATen/ops/sort_native.h>
#include <ATen/ops/sort_ops.h>
#endif

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
  if (in.dim() == 0 && in.numel() == 1) {
    values.copy_(in);
    indices.zero_();
    return;
  }
  TORCH_CHECK(
      in.scalar_type() == at::ScalarType::Half ||
          in.scalar_type() == at::ScalarType::BFloat16 ||
          in.scalar_type() == at::ScalarType::Float ||
          in.scalar_type() == at::ScalarType::Int ||
          in.scalar_type() == at::ScalarType::Long,
      "Sort only support half/bfloat16/float/int32/int64, got ",
      in.scalar_type());
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

std::tuple<Tensor&, Tensor&> SortOut(
    const Tensor& self,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  if (C10_UNLIKELY(self.numel() == 0)) {
    return std::forward_as_tuple(values, indices);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  values.resize_(self_.sizes());
  indices.resize_(self_.sizes());

  // default statble is false
  SortCall(values, indices, self_, dim_, descending, false);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> Sort(
    const Tensor& self,
    int64_t dim,
    bool descending) {
  auto self_ = self.contiguous();
  Tensor values = at::empty(self_.sizes(), self_.options());
  Tensor indices = at::empty(self_.sizes(), self_.options().dtype(kLong));

  return SortOut(self_, dim, descending, values, indices);
}

std::tuple<Tensor&, Tensor&> SortStableOut(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending,
    Tensor& values,
    Tensor& indices) {
  if (C10_UNLIKELY(self.numel() == 0)) {
    return std::forward_as_tuple(values, indices);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  auto self_ = self.contiguous();
  values.resize_(self_.sizes());
  indices.resize_(self_.sizes());

  TORCH_INTERNAL_ASSERT(
      stable.has_value(),
      "sort_out(): c10::optional<bool> for stable has to have value.");
  bool stable_ = stable.value();
  SortCall(values, indices, self_, dim_, descending, stable_);
  return std::forward_as_tuple(values, indices);
}

std::tuple<Tensor, Tensor> SortStable(
    const Tensor& self,
    c10::optional<bool> stable,
    int64_t dim,
    bool descending) {
  auto self_ = self.contiguous();
  Tensor values = at::empty(self_.sizes(), self_.options());
  Tensor indices = at::empty(self_.sizes(), self_.options().dtype(kLong));

  return SortStableOut(self_, stable, dim, descending, values, indices);
}

Tensor ArgsortStable(
    const Tensor& self,
    bool stable,
    int64_t dim,
    bool descending) {
  return std::get<1>(at::sort(self, stable, dim, descending));
}

} // namespace musa
} // namespace at
