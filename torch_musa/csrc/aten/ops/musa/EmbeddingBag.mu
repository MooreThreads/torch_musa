#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/List.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/Embedding.h"
#include "torch_musa/csrc/aten/ops/musa/EmbeddingBagHelper.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <ATen/AccumulateType.h>
#include <ATen/native/musa/thread_constants.h>
#include <ATen/musa/cub.muh>
#include <ATen/native/musa/EmbeddingBackwardKernel.muh>
#include <ATen/native/musa/KernelUtils.muh>
#include <ATen/native/musa/SortingCommon.muh>
#include <ATen/native/musa/block_reduce.muh>

#if CUB_SUPPORTS_SCAN_BY_KEY()
#include <thrust/iterator/reverse_iterator.h>
#endif

namespace at {
namespace native {

#if !CUB_SUPPORTS_SCAN_BY_KEY()
template <typename index_t>
void embedding_dense_backward_cuda_scan(Tensor& sorted_indices, Tensor& count);
#endif

namespace {

template <
    typename DataType,
    typename IndexType,
    int BLOCK_X,
    int BLOCK_Y,
    EmbeddingBagMode mode>
__global__ void EmbeddingBag1DKernel(
    DataType* out,
    const DataType* tbl,
    const IndexType* idx,
    const IndexType* offsets,
    const int num_indices,
    const int num_bags,
    const int tbl_w,
    const int padding_idx) {
  int chunksPerBag = at::musa::ceil_div(tbl_w, BLOCK_X);
  int numChunks = num_bags * chunksPerBag;

  int chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  int chunkStride = gridDim.x * blockDim.y;

  for (int chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    for (int dim = (chunk % chunksPerBag) * blockDim.x + threadIdx.x;
         dim < tbl_w;
         dim += blockDim.x) {
      int bag = chunk / chunksPerBag;
      IndexType begin = offsets[bag];
      IndexType end = (bag == num_bags - 1) ? num_indices : (offsets[bag + 1]);

      DataType res = 0.f;
      int bag_size = 0;
      for (IndexType emb = begin; emb < end; emb++) {
        IndexType id = idx[emb];
        bool padding_flag = (id == padding_idx);
        if (mode == EmbeddingBagMode::MAX) {
          DataType value = tbl[id * tbl_w + dim];
          if (bag_size == 0 || value > res) {
            res = padding_flag ? res : value;
          }
          bag_size += padding_flag ? 0 : 1;
        } else {
          DataType value = padding_flag ? 0.f : tbl[id * tbl_w + dim];
          res += value;
          bag_size += padding_flag ? 0 : 1;
        }
      }
      if (mode == EmbeddingBagMode::MEAN) {
        if (bag_size != 0) {
          res = res / bag_size;
        }
      }

      out[bag * tbl_w + dim] = res;
    }
  }
}

template <
    typename DataType,
    typename IndexType,
    int BLOCK_X,
    int BLOCK_Y,
    typename ReduceOp>
__global__ void EmbeddingBag2DKernel(
    DataType* out,
    const DataType* tbl,
    const IndexType* idx,
    const int A,
    const int B,
    const int C,
    const int tbl_h,
    const int padding_idx,
    DataType param,
    ReduceOp op) {
  __shared__ DataType share_array[BLOCK_Y][BLOCK_X];

  int tidx = threadIdx.x;
  int tidy = threadIdx.y;

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < A * C;
       index += gridDim.x * blockDim.x) {
    int row = index / C;
    int col = index % C;

    IndexType id = tidy < B ? idx[row * B + tidy] : -1;
    bool valid_id = (id >= 0 && id < tbl_h);
    DataType v1, v2;
    bool padding_flag = (id == padding_idx);
    v1 = padding_flag ? op.init_value
                      : (valid_id ? tbl[id * C + col] : op.init_value);

    for (int b = tidy + BLOCK_Y; b < B; b += BLOCK_Y) {
      id = idx[row * B + b];
      valid_id = (id >= 0 && id < tbl_h);
      padding_flag = (id == padding_idx);
      v2 = padding_flag ? op.init_value
                        : (valid_id ? tbl[id * C + col] : op.init_value);
      op.apply(v1, v2);
    }
    share_array[tidy][tidx] = v1;
    __SYNCTHREADS;

#pragma unroll
    for (int s = BLOCK_Y / 2; s >= 2; s >>= 1) {
      if (tidy < s) {
        op.apply(share_array[tidy][tidx], share_array[tidy + s][tidx]);
      }
      __SYNCTHREADS;
    }

    if (tidy == 0) {
      v1 = share_array[0][tidx];
      v2 = share_array[1][tidx];
      op.apply(v1, v2);
      op.postOp(v1, param);
      out[row * C + col] = v1;
    }
  }
}

} // namespace

void _EmbeddingBag1DRun(
    const Tensor& o,
    const Tensor& t,
    const Tensor& i,
    const Tensor& offsets,
    const int64_t mode,
    const int64_t padding_idx_) {
  TORCH_CHECK(offsets.data_ptr() != nullptr, "1D input must have offsets");
  auto stream = at::musa::getCurrentMUSAStream();

  int tbl_w = t.sizes()[1];
  int num_bags = offsets.numel();
  int num_indices = i.numel();
  int padding_idx = static_cast<int>(padding_idx_);

  const uint32_t block_x = 128;
  const uint32_t block_y = 8;
  const uint32_t grid_x = at::musa::ceil_div(tbl_w, static_cast<int>(block_x));
  const uint32_t grid_y = 1;

  dim3 block_size{block_x, block_y, 1};
  dim3 grid_size{grid_x, grid_y, 1};

#define cb(_embbag_mode, _idx_type)                                      \
  EmbeddingBag1DKernel<float, _idx_type, block_x, block_y, _embbag_mode> \
      <<<grid_size, block_size, 0, stream>>>(                            \
          static_cast<float*>(o.data_ptr()),                             \
          static_cast<const float*>(t.data_ptr()),                       \
          static_cast<const _idx_type*>(i.data_ptr()),                   \
          static_cast<const _idx_type*>(offsets.data_ptr()),             \
          num_indices,                                                   \
          num_bags,                                                      \
          tbl_w,                                                         \
          padding_idx);

  switch (mode) {
    case 0: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::SUM, int32_t);
      } else {
        cb(EmbeddingBagMode::SUM, int64_t);
      }
      break;
    }
    case 1: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::MEAN, int32_t);
      } else {
        cb(EmbeddingBagMode::MEAN, int64_t);
      }
      break;
    }
    case 2: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::MAX, int32_t);
      } else {
        cb(EmbeddingBagMode::MAX, int64_t);
      }
      break;
    }
    default:
      TORCH_CHECK(false, "EmbeddingBag doesn't support mode: ", mode);
  }
#undef cb
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

void _EmbeddingBag2DRun(
    const Tensor& o,
    const Tensor& t,
    const Tensor& i,
    const int64_t mode,
    const int64_t padding_idx_) {
  auto stream = at::musa::getCurrentMUSAStream();
  int tbl_h = t.sizes()[0];
  int tbl_w = t.sizes()[1];
  int padding_idx = static_cast<int>(padding_idx_);

  int A = i.sizes()[0];
  int B = i.sizes()[1];
  int C = tbl_w;

  const uint32_t block_x = 128;
  const uint32_t block_y = 8;
  const uint32_t grid_x = at::musa::ceil_div(A * C, static_cast<int>(block_x));
  const uint32_t grid_y = 1;

  dim3 block_size{block_x, block_y, 1};
  dim3 grid_size{grid_x, grid_y, 1};

#define cb(_embbag_mode, _idx_type)                                  \
  ReduceOp<_embbag_mode> op;                                         \
  EmbeddingBag2DKernel<                                              \
      float,                                                         \
      _idx_type,                                                     \
      block_x,                                                       \
      block_y,                                                       \
      ReduceOp<_embbag_mode>><<<grid_size, block_size, 0, stream>>>( \
      static_cast<float*>(o.data_ptr()),                             \
      static_cast<const float*>(t.data_ptr()),                       \
      static_cast<const _idx_type*>(i.data_ptr()),                   \
      A,                                                             \
      B,                                                             \
      C,                                                             \
      tbl_h,                                                         \
      padding_idx,                                                   \
      1.0f / B,                                                      \
      op);

  switch (mode) {
    case 0: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::SUM, int32_t);
      } else {
        cb(EmbeddingBagMode::SUM, int64_t);
      }
      break;
    }
    case 1: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::MEAN, int32_t);
      } else {
        cb(EmbeddingBagMode::MEAN, int64_t);
      }
      break;
    }
    case 2: {
      if (at::ScalarType::Int == i.scalar_type()) {
        cb(EmbeddingBagMode::MAX, int32_t);
      } else {
        cb(EmbeddingBagMode::MAX, int64_t);
      }
      break;
    }
    default:
      TORCH_CHECK(false, "EmbeddingBag doesn't support mode: ", mode);
  }
#undef cb
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

void EmbeddingBagRun(
    Tensor& out,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const int64_t mode,
    const int64_t padding_idx) {
  TORCH_CHECK(
      weight.dim() == 2,
      "EmbeddingBag only supports table dimension 2, but got: ",
      weight.dim());
  TORCH_CHECK(
      indices.dim() == 1 || indices.dim() == 2,
      "EmbeddingBag only supports input dimension 1 or 2, but got: ",
      indices.dim());
  TORCH_CHECK(
      at::ScalarType::Int == indices.scalar_type() ||
          at::ScalarType::Long == indices.scalar_type(),
      "Unsupported input data type");
  TORCH_CHECK(
      out.scalar_type() == weight.scalar_type(),
      "output dtype table dtype must be same");
  TORCH_CHECK(
      at::ScalarType::Float == out.scalar_type(),
      "Unsupported output data type");

  if (indices.dim() == 1) {
    _EmbeddingBag1DRun(out, weight, indices, offsets, mode, padding_idx);
  } else {
    _EmbeddingBag2DRun(out, weight, indices, mode, padding_idx);
  }
}

REGISTER_MUSA_DISPATCH(embedding_bag_stub, &EmbeddingBagRun);

} // namespace native
} // namespace at
