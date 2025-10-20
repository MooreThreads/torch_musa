#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
// clang-format off
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/core/Tensor.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InitialTensorOptions.h>
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/WrapDimUtilsMulti.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/native/Resize.h>
#include <ATen/native/SparseTensorUtils.h>
#include <algorithm>
#include <ATen/AccumulateType.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_convert_indices_from_coo_to_csr_native.h>
#include <ATen/ops/_convert_indices_from_csr_to_coo_native.h>
#include <ATen/ops/_sparse_csr_tensor_unsafe_native.h>
#include <ATen/ops/_unique.h>
#include <ATen/ops/add_native.h>
#include <ATen/ops/resize_as_sparse_native.h>
#include <ATen/ops/tensor.h>
#include <ATen/ops/zeros.h>
#endif

#include <musa_runtime.h>
#include <type_traits>


#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <ATen/musa/MUSAUtils.h>
#include <ATen/musa/ThrustAllocator.h>
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

#include <ATen/native/musa/Reduce.muh>
#include <ATen/native/sparse/musa/SparseBlasImpl.h>
#include <ATen/native/sparse/musa/SparseMUSABlas.h>
#include <ATen/native/sparse/musa/SparseMUSATensorMath.muh>

#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/sequence.h>

namespace at::native {

namespace {

template <typename input_t, typename output_t>
__global__ void convert_indices_from_coo_to_csr_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t size, const int64_t numel) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid == 0) {
    for (int64_t i = 0; i <= data_in[0]; i++)
      data_out[i] = static_cast<output_t>(0);
  } else if (tid < numel) {
    for (int64_t i = data_in[tid - 1]; i < data_in[tid]; i++)
      data_out[i + 1] = static_cast<output_t>(tid);
  } else if (tid == numel) {
    for (int64_t i = data_in[numel - 1] + 1; i < size + 1; i++)
      data_out[i] = static_cast<output_t>(numel);
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_coo_to_csr_cuda(const Tensor& result, const Tensor& input, const int64_t size) {
  int64_t numel = input.numel();
  const input_t* data_in = input.const_data_ptr<input_t>();
  output_t* data_out = result.data_ptr<output_t>();

  if (numel == 0) {
    result.zero_();
    return;
  }

  // Run (numel + 1) threads...
  int64_t THREADS = at::musa::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (numel + THREADS) / THREADS;
  at::musa::MUSAStream stream = at::musa::getCurrentMUSAStream();
  convert_indices_from_coo_to_csr_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, data_in, size, numel);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

template <typename input_t, typename output_t>
__global__ void convert_indices_from_csr_to_coo_cuda_kernel(output_t* data_out, const input_t* data_in, const int64_t nrows, const int64_t nnz, const int64_t nbatches) {
  int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;

  if (tid < nrows * nbatches) {
    int64_t b = tid / nrows;
    int64_t i_ = b * (nrows + 1) + tid % nrows;
    for (int64_t i = data_in[i_]; i < data_in[i_ + 1]; i++) {
      data_out[b * nnz + i] = static_cast<output_t>(tid % nrows);
    }
  }
}

template <typename input_t, typename output_t>
void convert_indices_from_csr_to_coo_cuda(const Tensor& indices, const Tensor& crow_indices, const Tensor& col_indices, const bool transpose=false) {
  int64_t nrows = crow_indices.size(-1) - 1;
  int64_t nnz = col_indices.size(-1);
  if (nrows == 0 || nnz == 0) {
    indices.zero_();
    return;
  }
  int64_t total_nnz = col_indices.numel();
  int64_t batch_ndim = crow_indices.dim() - 1;
  if (batch_ndim > 0) {
    auto batch_indices = indices.narrow(0, 0, batch_ndim);
    batch_indices.copy_(at::sparse::full_coo_indices(crow_indices.sizes().slice(0, batch_ndim), indices.options())
                        .repeat_interleave(nnz, 1));
  }

  auto crow_indices_ = crow_indices.expect_contiguous();
  const input_t* crow_indices_data_in = crow_indices_->const_data_ptr<input_t>();
  TORCH_INTERNAL_ASSERT(indices.is_contiguous());
  auto row0 = indices.select(0, transpose?batch_ndim + 1:batch_ndim + 0);
  auto row1 = indices.select(0, transpose?batch_ndim + 0:batch_ndim + 1);
  auto col_indices_ = col_indices.expect_contiguous();
  row1.copy_(col_indices_->view({-1}));
  output_t* data_out = row0.data_ptr<output_t>();

  // Run nrows * nbatches threads...
  int64_t nbatches = total_nnz / nnz;
  int64_t THREADS = at::musa::getCurrentDeviceProperties()->maxThreadsPerBlock;
  int64_t BLOCKS = (nrows * nbatches + THREADS) / THREADS;
  at::musa::MUSAStream stream = at::musa::getCurrentMUSAStream();
  convert_indices_from_csr_to_coo_cuda_kernel<<<BLOCKS, THREADS, 0, stream>>>(data_out, crow_indices_data_in, nrows, nnz, nbatches);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

} // namespace

using namespace at::sparse_csr;
// certain utility functions are usable from sparse COO.
using namespace at::sparse;

TORCH_IMPL_FUNC(_convert_indices_from_coo_to_csr_structured_cuda) (
  const Tensor& input, const int64_t size, const bool out_int32, const Tensor& result
) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int>(result, input, size);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(input.scalar_type(), "convert_indices_from_coo_to_csr_cuda", [&] {
      convert_indices_from_coo_to_csr_cuda<scalar_t, int64_t>(result, input, size);
    });
  }
}

TORCH_IMPL_FUNC(_convert_indices_from_csr_to_coo_structured_cuda) (
  const Tensor& crow_indices, const Tensor& col_indices, const bool out_int32, const bool transpose, const Tensor& result
) {
  if (out_int32) {
    AT_DISPATCH_INTEGRAL_TYPES(crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cuda", [&] {
      convert_indices_from_csr_to_coo_cuda<scalar_t, int32_t>(result, crow_indices, col_indices, transpose);
    });
  } else {
    AT_DISPATCH_INTEGRAL_TYPES(crow_indices.scalar_type(), "convert_indices_from_csr_to_coo_cuda", [&] {
      convert_indices_from_csr_to_coo_cuda<scalar_t, int64_t>(result, crow_indices, col_indices, transpose);
    });
  }
}

} // namespace at::native
