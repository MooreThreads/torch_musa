#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/ThrustAllocator.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <c10/util/Load.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/_unique2_native.h>
#include <ATen/ops/_unique_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/unique_consecutive_native.h>
#include <ATen/ops/unique_dim_consecutive_native.h>
#include <ATen/ops/unique_dim_native.h>
#endif

#include <thrust/adjacent_difference.h>
#include <thrust/execution_policy.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <iterator>
#include <tuple>

#include "torch_musa/csrc/aten/ops/musa/UniqueCub.muh"

namespace at::musa {

namespace {

template <
    typename policy_t,
    typename scalar_t,
    typename equal_t,
    typename not_equal_t>
std::tuple<Tensor, Tensor, int64_t> compute_unique(
    const policy_t& policy,
    scalar_t* data,
    int64_t num_inp,
    const Tensor& sorted_indices,
    const bool return_inverse,
    const bool return_counts,
    TensorOptions options,
    equal_t equal,
    not_equal_t not_equal) {
  // inverse indices
  Tensor inverse_indices;
  if (!return_inverse || num_inp == 0) {
    inverse_indices = at::empty({0}, options);
  } else {
    TORCH_CHECK(
        sorted_indices.defined(),
        "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
    const int64_t* sorted_indices_ptr = sorted_indices.data_ptr<int64_t>();
    Tensor inv_loc = at::empty({num_inp}, options);
    inverse_indices = at::empty({num_inp}, options);
    int64_t* inv_loc_ptr = inv_loc.data_ptr<int64_t>();
    int64_t* inverse_indices_ptr = inverse_indices.data_ptr<int64_t>();
    thrust::adjacent_difference(
        policy, data, data + num_inp, inv_loc_ptr, not_equal);
    inv_loc[0] = 0;
    thrust::inclusive_scan(
        policy, inv_loc_ptr, inv_loc_ptr + num_inp, inv_loc_ptr);
    thrust::scatter(
        policy,
        inv_loc_ptr,
        inv_loc_ptr + num_inp,
        sorted_indices_ptr,
        inverse_indices_ptr);
  }

  // unique and count
  Tensor counts = at::empty({0}, options);
  int64_t num_out;
  if (!return_counts) {
    num_out = thrust::unique(policy, data, data + num_inp, equal) - data;
  } else {
    Tensor range = at::arange(0, num_inp + 1, options);
    int64_t* range_ptr = range.data_ptr<int64_t>();
    num_out =
        thrust::unique_by_key(policy, data, data + num_inp, range_ptr, equal)
            .first -
        data;
    range[num_out] = num_inp;
    counts.resize_(num_out);
    int64_t* counts_ptr = counts.data_ptr<int64_t>();
    thrust::adjacent_difference(
        policy, range_ptr + 1, range_ptr + num_out + 1, counts_ptr);
  }

  AT_MUSA_CHECK(musaGetLastError());
  return std::tuple<Tensor, Tensor, int64_t>(inverse_indices, counts, num_out);
}
} // namespace

template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> unique_dim_musa_template(
    const Tensor& self,
    const int64_t dim,
    const bool consecutive,
    const bool return_inverse,
    const bool return_counts) {
  /**
   * The idea for implementing this is basically the same as unique.
   * For unique_dim, we are taking the unique with respect to a index
   * tensor, but during the processes, we override the compare and equal
   * operator by checking the data underlying it instead. After the
   * algorithm, we would use index_select to map the resulting indicies
   * to the result on the actual data.
   */

  musaStream_t stream = at::musa::getCurrentMUSAStream();
  at::musa::ThrustAllocator allocator;
  auto policy = thrust::musa::par(allocator).on(stream);

  auto sizes = self.sizes().vec();
  // check how many zero dimensions exist
  auto num_zero_dims = std::count(sizes.begin(), sizes.end(), 0);

  // tensor is not well formed as it has 0 sized dimensions
  if (self.size(dim) == 0) {
    TORCH_CHECK(
        num_zero_dims == 1,
        "Number of zero sized dimensions is more than one, so unique cannot be applied ")
    Tensor output = at::empty(sizes, self.options());
    Tensor inverse_indices = at::empty({0}, self.options().dtype(kLong));
    Tensor counts = at::empty({0}, self.options().dtype(kLong));

    return std::make_tuple(output, inverse_indices, counts);
  }

  TORCH_CHECK(
      num_zero_dims == 0,
      "There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");

  int64_t num_inp = self.size(dim);
  auto options = self.options().dtype(kLong);
  Tensor input_flat = self.transpose(dim, 0).contiguous().view({num_inp, -1});
  int64_t n = input_flat.size(1);
  scalar_t* input_flat_ptr = input_flat.data_ptr<scalar_t>();

  Tensor indices = at::arange(0, num_inp, options);
  int64_t* indices_data = indices.data_ptr<int64_t>();

  if (!consecutive) {
    thrust::sort(
        policy,
        indices_data,
        indices_data + num_inp,
        [=] __device__(int64_t a, int64_t b) -> bool {
          for (int64_t i = 0; i < n; ++i) {
            scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
            scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
            if (lhs < rhs) {
              return true;
            } else if (lhs > rhs) {
              return false;
            }
          }
          return false;
        });
  }

  Tensor inverse_indices, counts;
  int64_t num_out;
  std::tie(inverse_indices, counts, num_out) = compute_unique(
      policy,
      indices_data,
      num_inp,
      indices,
      return_inverse,
      return_counts,
      options,
      [=] __device__(int64_t a, int64_t b) -> bool {
        for (int64_t i = 0; i < n; ++i) {
          scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
          scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
          if (lhs != rhs) {
            return false;
          }
        }
        return true;
      },
      [=] __device__(int64_t a, int64_t b) -> int64_t {
        for (int64_t i = 0; i < n; ++i) {
          scalar_t lhs = c10::load(&input_flat_ptr[i + a * n]);
          scalar_t rhs = c10::load(&input_flat_ptr[i + b * n]);
          if (lhs != rhs) {
            return 1;
          }
        }
        return 0;
      });
  indices.resize_(num_out);

  return std::tuple<Tensor, Tensor, Tensor>(
      self.index_select(dim, indices), inverse_indices, counts);
}

#define INSTANTIATE_UNIQUE_MUSA_TEMPLATE(TYPE)                                \
  template std::tuple<Tensor, Tensor, Tensor> unique_dim_musa_template<TYPE>( \
      const Tensor& self,                                                     \
      const int64_t dim,                                                      \
      const bool consecutive,                                                 \
      const bool return_inverse,                                              \
      const bool return_counts)

INSTANTIATE_UNIQUE_MUSA_TEMPLATE(uint8_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int8_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(double);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(float);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int32_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int64_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int16_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(bool);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(at::Half);

#undef INSTANTIATE

} // namespace at::musa
