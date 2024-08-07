#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/ops/musa/UniqueCub.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {
namespace {
void SortCall(
    Tensor& values,
    Tensor& indices,
    const Tensor& in,
    int64_t dim,
    bool descending,
    bool stable) {
  auto input_ = CreateMUTensor(in);
  auto values_ = CreateMUTensor(values);
  auto indices_ = CreateMUTensor(indices);

  muHandle& h = GetMudnnHandle();
  ::musa::dnn::Sort mSort;
  CHECK_MUDNN_STATUS(mSort.SetDim(dim), "Sort set dim param failed");
  CHECK_MUDNN_STATUS(
      mSort.SetDescending(descending), "Sort set descending flag failed");
  CHECK_MUDNN_STATUS(mSort.SetStable(stable), "Sort set stable flag failed");
  CHECK_MUDNN_STATUS(
      mSort.Run(h, values_, indices_, input_, InternalMemAlloc),
      "Sort run kernel failed");
}

std::tuple<Tensor, Tensor> SortHelper(const Tensor& self, int64_t dim = -1) {
  auto self_ = self.contiguous();
  Tensor values = at::empty(self_.sizes(), self_.options());
  Tensor indices = at::empty(self_.sizes(), self_.options().dtype(kLong));
  if (C10_UNLIKELY(self.numel() == 0)) {
    return std::forward_as_tuple(values, indices);
  }
  c10::musa::MUSAGuard device_guard(self.device());
  int64_t dim_ = maybe_wrap_dim(dim, self.dim(), true);
  if (self_.dim() == 0 && self_.numel() == 1) {
    values.copy_(self_);
    indices.zero_();
    return std::forward_as_tuple(values, indices);
  }
  // default statble is false
  SortCall(values, indices, self_, dim_, false, false);
  return std::forward_as_tuple(values, indices);
}
} // anonymous namespace

std::tuple<Tensor, Tensor, Tensor> Unique2(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return AT_DISPATCH_ALL_TYPES_AND(kHalf, self.scalar_type(), "unique", [&] {
    // The current MUSA implementation of unique always sort due to the
    // lack of hashtable implementation in thrust, and we use muDNN Sort
    // instead of cub which has worse latency.
    auto num_inp = self.numel();
    TORCH_CHECK(num_inp <= INT_MAX, "num_inp ", num_inp, " is too big");
    if (num_inp == 0) {
      Tensor output = at::empty({0}, self.options());
      Tensor inverse_indices =
          at::empty(self.sizes(), self.options().dtype(kLong));
      Tensor counts = at::empty({0}, self.options().dtype(kLong));
      return std::tuple<Tensor, Tensor, Tensor>(
          output, inverse_indices, counts);
    }
    Tensor self_c = self.contiguous();
    Tensor sorted_self, sorted_indices;
    std::tie(sorted_self, sorted_indices) = SortHelper(self_c.view(-1));
    sorted_self = sorted_self.view_as(self);
    sorted_indices = sorted_indices.view_as(self);
    return compute_unique<scalar_t>(
        sorted_self, sorted_indices, return_inverse, return_counts, false);
  });
}

std::tuple<Tensor, Tensor> Unique(
    const Tensor& self,
    const bool sorted,
    const bool return_inverse) {
  Tensor output, inverse;
  std::tie(output, inverse, std::ignore) =
      Unique2(self, sorted, return_inverse, false);
  return std::forward_as_tuple(output, inverse);
}

std::tuple<Tensor, Tensor, Tensor> UniqueDim(
    const Tensor& self,
    const int64_t dim,
    const bool sorted,
    const bool return_inverse,
    const bool return_counts) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return AT_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
        Tensor sorted_self;
        std::tie(sorted_self, std::ignore) = SortHelper(self, dim);
        return unique_dim_musa_template<scalar_t>(
            sorted_self, dim, true, return_inverse, return_counts);
      });
}

std::tuple<Tensor, Tensor, Tensor> UniqueDimConsecutive(
    const Tensor& self,
    const int64_t dim,
    const bool return_inverse,
    const bool return_counts) {
  const OptionalDeviceGuard device_guard(device_of(self));
  return AT_DISPATCH_ALL_TYPES_AND2(
      kBool, kHalf, self.scalar_type(), "unique_dim", [&] {
        return unique_dim_musa_template<scalar_t>(
            self, dim, true, return_inverse, return_counts);
      });
}

std::tuple<Tensor, Tensor, Tensor> UniqueConsecutive(
    const Tensor& self,
    const bool return_inverse,
    const bool return_counts,
    c10::optional<int64_t> dim) {
  const OptionalDeviceGuard device_guard(device_of(self));
  if (!dim.has_value()) {
    return AT_DISPATCH_ALL_TYPES_AND(kHalf, self.scalar_type(), "unique", [&] {
      auto num_inp = self.numel();
      TORCH_CHECK(num_inp <= INT_MAX, "num_inp ", num_inp, " is too big");
      if (num_inp == 0) {
        Tensor output = at::empty({0}, self.options());
        Tensor inverse_indices =
            at::empty(self.sizes(), self.options().dtype(kLong));
        Tensor counts = at::empty({0}, self.options().dtype(kLong));
        return std::tuple<Tensor, Tensor, Tensor>(
            output, inverse_indices, counts);
      }
      Tensor self_c = self.contiguous();
      Tensor indices;
      return compute_unique<scalar_t>(
          self_c, indices, return_inverse, return_counts, true);
    });
  }
  return UniqueDimConsecutive(self, dim.value(), return_inverse, return_counts);
}

} // namespace musa
} // namespace at
