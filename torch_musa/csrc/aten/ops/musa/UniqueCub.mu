#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/musa/detail/KernelUtils.h>
#include <ATen/musa/MUSA_PORT_ApplyUtils.muh>
#include <ATen/musa/cub.muh>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <c10/core/DeviceArray.h>
#include <c10/util/Load.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/empty.h>
#endif

#include "torch_musa/csrc/aten/ops/musa/UniqueCub.muh"

namespace at::musa {

namespace {

template <typename InputIteratorT>
__global__ void adjacent_difference_kernel(
    int64_t n,
    InputIteratorT input,
    int* output) {
  CUDA_KERNEL_LOOP(i, n) {
    output[i] = i > 0 ? input[i] != input[i - 1] : 0;
  }
}

__global__ void scatter_kernel(
    int64_t n,
    const int64_t* input,
    const int64_t* indices,
    int64_t* output) {
  CUDA_KERNEL_LOOP(i, n) {
    output[indices[i]] = input[i];
  }
}

template <typename scalar_t>
const scalar_t* wrap_input_iterator(const scalar_t* data) {
  return data;
}

struct LoadBoolOp {
  __device__ bool operator()(uint8_t x) const {
    return static_cast<bool>(x);
  }
};

auto wrap_input_iterator(const bool* data) {
  // See NOTE [Loading boolean values]
  LoadBoolOp op;
  return NO_ROCM(at_cuda_detail)::cub::
      TransformInputIterator<bool, LoadBoolOp, const uint8_t*, int>(
          reinterpret_cast<const uint8_t*>(data), op);
}

} // namespace

// A variation of compute_unique (defined in Unique.mu) that doesn't allow
// customizing equal and not_equal (CUB doesn't allow them).
template <typename scalar_t>
std::tuple<Tensor, Tensor, Tensor> compute_unique(
    const Tensor& sorted,
    const Tensor& sorted_indices,
    const bool return_inverse,
    const bool return_counts,
    const bool consecutive) {
  int64_t num_inp = sorted.numel();
  auto options = sorted.options().dtype(kLong);
  auto data = wrap_input_iterator(sorted.data_ptr<scalar_t>());
  musaStream_t stream = at::musa::getCurrentMUSAStream();

  // inverse indices
  Tensor inverse_indices;
  if (!return_inverse) {
    inverse_indices = at::empty({0}, options);
  } else {
    inverse_indices = at::empty(sorted.sizes(), options);
    Tensor inv_loc = consecutive ? at::empty({num_inp}, options.dtype(kInt))
                                 : inverse_indices;
    int* inv_loc_ptr = static_cast<int*>(inv_loc.data_ptr());
    const dim3 block =
        dim3(std::min(static_cast<int64_t>(musa::getApplyBlock().x), num_inp));
    dim3 grid;
    int curDevice = -1;
    musaGetDevice(&curDevice);
    musa::getApplyGrid(num_inp, grid, curDevice);
    adjacent_difference_kernel<<<grid, block, 0, stream>>>(
        num_inp, data, inv_loc_ptr);
    C10_MUSA_KERNEL_LAUNCH_CHECK();

    Tensor inv_loc_out =
        consecutive ? inverse_indices : at::empty({num_inp}, options);
    at::musa::cub::inclusive_sum_truncating(
        inv_loc_ptr, inv_loc_out.data_ptr<int64_t>(), num_inp);

    if (!consecutive) {
      TORCH_INTERNAL_ASSERT(
          sorted_indices.defined(),
          "return_inverse is set to true, but sorted_indices is undefined. Send a bug report!");
      scatter_kernel<<<grid, block, 0, stream>>>(
          num_inp,
          inv_loc_out.data_ptr<int64_t>(),
          sorted_indices.data_ptr<int64_t>(),
          inverse_indices.data_ptr<int64_t>());
      C10_MUSA_KERNEL_LAUNCH_CHECK();
    }
  }

  // unique and count
  Tensor data_out = at::empty({num_inp}, sorted.options());
  Tensor counts = at::empty({0}, options);
  Tensor length = at::empty({1}, options);
  int64_t num_out;
  if (!return_counts) {
    musa::cub::unique(
        data,
        data_out.data_ptr<scalar_t>(),
        length.data_ptr<int64_t>(),
        num_inp);
    num_out = length.item<int64_t>();
  } else {
    counts.resize_(num_inp);
    at::musa::cub::run_length_encode(
        data,
        data_out.data_ptr<scalar_t>(),
        counts.data_ptr<int64_t>(),
        length.data_ptr<int64_t>(),
        num_inp);
    num_out = length.item<int64_t>();
    counts.resize_(num_out);
  }

  data_out.resize_(num_out);
  return std::tuple<Tensor, Tensor, Tensor>(data_out, inverse_indices, counts);
}

#define INSTANTIATE_UNIQUE_MUSA_TEMPLATE(TYPE)                      \
  template std::tuple<Tensor, Tensor, Tensor> compute_unique<TYPE>( \
      const Tensor& sorted,                                         \
      const Tensor& sorted_indices,                                 \
      const bool return_inverse,                                    \
      const bool return_counts,                                     \
      const bool consecutive)

INSTANTIATE_UNIQUE_MUSA_TEMPLATE(uint8_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int8_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(double);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(float);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int32_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int64_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(int16_t);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(at::Half);
INSTANTIATE_UNIQUE_MUSA_TEMPLATE(at::BFloat16);

#undef INSTANTIATE_UNIQUE_MUSA_TEMPLATE

} // namespace at::musa
