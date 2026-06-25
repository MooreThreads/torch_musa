#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include <ATen/Dispatch.h>
#include "torch_musa/csrc/aten/musa/MUSAContextLight.h"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/musa/EmbeddingBag.muh"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::musa {

namespace {

template <
    typename DataType,
    typename IndexType,
    typename OffsetType,
    native::EmbeddingBagMode mode>
__global__ void EmbeddingBagSumMeanKernel(
    DataType* out,
    int64_t* offset2bag,
    int64_t* bag_size,
    const DataType* weight,
    const IndexType* indices,
    const OffsetType* offsets,
    const DataType* scales,
    const int64_t padding_idx,
    const int64_t numBags,
    const int64_t featureSize,
    const int64_t numIndices,
    const int numChunks,
    FastDivmod fdm) {
  using Reduce = ReduceOp<DataType, mode>;
  using CType = typename Reduce::CType;
  const int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    uint32_t bag, chunk_id;
    fdm(bag, chunk_id, (uint32_t)chunk);
    int64_t dim = chunk_id * blockDim.x + threadIdx.x;
    if (dim < featureSize) {
      int64_t begin = offsets[bag];
      int64_t end = (bag == numBags - 1) ? numIndices : (offsets[bag + 1]);
      Reduce rd;
      for (int64_t emb = begin; emb < end; emb++) {
        int64_t id = indices[emb];
        if (id != padding_idx) {
          auto value = (CType)(weight[id * featureSize + dim]);
          if constexpr (mode == native::EmbeddingBagMode::SUM) {
            if (scales != nullptr) {
              value = value * (CType)(scales[emb]);
            }
          }
          rd.step(value);
        }
        if (dim == 0) {
          offset2bag[emb] = bag;
        }
      }
      if (dim == 0) {
        bag_size[bag] = rd.count();
      }
      out[bag * featureSize + dim] = (DataType)(rd.reduce());
    }
  }
}

template <typename DataType, typename IndexType, typename OffsetType>
__global__ void EmbeddingBagMaxKernel(
    DataType* out,
    int64_t* offset2bag,
    int64_t* bag_size,
    int64_t* max_indices,
    const DataType* weight,
    const IndexType* indices,
    const OffsetType* offsets,
    const int64_t padding_idx,
    const int64_t numBags,
    const int64_t featureSize,
    const int64_t numIndices,
    const int numChunks,
    FastDivmod fdm) {
  using Reduce = ReduceOp<DataType, native::EmbeddingBagMode::MAX>;
  using CType = typename Reduce::CType;
  const int64_t chunkOffset = blockIdx.x * blockDim.y + threadIdx.y;
  const int64_t chunkStride = gridDim.x * blockDim.y;

  for (int64_t chunk = chunkOffset; chunk < numChunks; chunk += chunkStride) {
    uint32_t bag, chunk_id;
    fdm(bag, chunk_id, (uint32_t)chunk);
    int64_t dim = chunk_id * blockDim.x + threadIdx.x;
    if (dim < featureSize) {
      int64_t begin = offsets[bag];
      int64_t end = (bag == numBags - 1) ? numIndices : (offsets[bag + 1]);
      Reduce rd;
      for (int64_t emb = begin; emb < end; emb++) {
        int64_t id = indices[emb];
        if (id != padding_idx) {
          auto value = (CType)(weight[id * featureSize + dim]);
          rd.step(value, id);
        }
        if (dim == 0) {
          offset2bag[emb] = bag;
        }
      }
      if (dim == 0) {
        bag_size[bag] = rd.count();
      }
      dim += bag * featureSize;
      max_indices[dim] = rd.max_idx();
      out[dim] = (DataType)(rd.reduce());
    }
  }
}

void LaunchEmbeddingBagKnernel(
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& per_sample_weights,
    const int64_t mode,
    const int64_t padding_idx,
    const int64_t numBags,
    const int64_t featureSize,
    const int64_t numIndices) {
  auto stream = getCurrentMUSAStream();
  const int64_t mpc = getCurrentDeviceProperties()->multiProcessorCount;

  const int64_t block_x = 32;
  int64_t block_y = 32;

  const auto chunksPerBag = ceil_div(featureSize, block_x);
  const auto numChunks = numBags * chunksPerBag;
  TORCH_INTERNAL_ASSERT(
      numChunks > 0 && numChunks <= std::numeric_limits<int32_t>::max());
  const auto fdm = FastDivmod(static_cast<uint32_t>(chunksPerBag));

  int64_t grid_x = ceil_div(numChunks, block_y);
  while (block_y > 16 && grid_x < mpc) {
    block_y = (block_y >> 1);
    grid_x = (grid_x << 1);
  }

  const dim3 block{(uint32_t)block_x, (uint32_t)block_y, 1};
  const dim3 grid{(uint32_t)grid_x, 1, 1};

#define CALL_SUM_MEAN(DATA, IDX, OFF, MODE)                                    \
  EmbeddingBagSumMeanKernel<DATA, IDX, OFF, MODE><<<grid, block, 0, stream>>>( \
      static_cast<DATA*>(out.data_ptr()),                                      \
      static_cast<int64_t*>(offset2bag.data_ptr()),                            \
      static_cast<int64_t*>(bag_size.data_ptr()),                              \
      static_cast<const DATA*>(weight.data_ptr()),                             \
      static_cast<const IDX*>(indices.data_ptr()),                             \
      static_cast<const OFF*>(offsets.data_ptr()),                             \
      static_cast<const DATA*>(psw_data),                                      \
      padding_idx,                                                             \
      numBags,                                                                 \
      featureSize,                                                             \
      numIndices,                                                              \
      (int)numChunks,                                                          \
      fdm);

#define DISPATCH_SUM_MEAN(MODE)                      \
  if (is_short_idx && is_short_off) {                \
    CALL_SUM_MEAN(scalar_t, int, int, MODE);         \
  } else if (is_short_idx) {                         \
    CALL_SUM_MEAN(scalar_t, int, int64_t, MODE);     \
  } else if (is_short_off) {                         \
    CALL_SUM_MEAN(scalar_t, int64_t, int, MODE);     \
  } else {                                           \
    CALL_SUM_MEAN(scalar_t, int64_t, int64_t, MODE); \
  }

#define CALL_MAX(DATA, IDX, OFF)                                     \
  EmbeddingBagMaxKernel<DATA, IDX, OFF><<<grid, block, 0, stream>>>( \
      static_cast<DATA*>(out.data_ptr()),                            \
      static_cast<int64_t*>(offset2bag.data_ptr()),                  \
      static_cast<int64_t*>(bag_size.data_ptr()),                    \
      static_cast<int64_t*>(max_indices.data_ptr()),                 \
      static_cast<const DATA*>(weight.data_ptr()),                   \
      static_cast<const IDX*>(indices.data_ptr()),                   \
      static_cast<const OFF*>(offsets.data_ptr()),                   \
      padding_idx,                                                   \
      numBags,                                                       \
      featureSize,                                                   \
      numIndices,                                                    \
      (int)numChunks,                                                \
      fdm);

  AT_DISPATCH_FLOATING_TYPES_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      weight.scalar_type(),
      "embedding_bag_musa",
      [&] {
        const bool is_short_idx = indices.scalar_type() == ScalarType::Int;
        const bool is_short_off = offsets.scalar_type() == ScalarType::Int;

        const auto* psw_data = per_sample_weights.defined()
            ? per_sample_weights.data_ptr()
            : nullptr;

        if (mode == native::EmbeddingBagMode::SUM) {
          DISPATCH_SUM_MEAN(native::EmbeddingBagMode::SUM);
        } else if (mode == native::EmbeddingBagMode::MEAN) {
          DISPATCH_SUM_MEAN(native::EmbeddingBagMode::MEAN);
        } else {
          if (is_short_idx && is_short_off) {
            CALL_MAX(scalar_t, int, int);
          } else if (is_short_idx) {
            CALL_MAX(scalar_t, int, int64_t);
          } else if (is_short_off) {
            CALL_MAX(scalar_t, int64_t, int);
          } else {
            CALL_MAX(scalar_t, int64_t, int64_t);
          }
        }
      });

  C10_MUSA_KERNEL_LAUNCH_CHECK();

#undef CALL_MAX
#undef DISPATCH_SUM_MEAN
#undef CALL_SUM_MEAN
}

} // anonymous namespace

void EmbeddingBagRun(
    Tensor& out,
    Tensor& offset2bag,
    Tensor& bag_size,
    Tensor& max_indices,
    const Tensor& weight,
    const Tensor& indices,
    const Tensor& offsets,
    const Tensor& per_sample_weights,
    const int64_t mode,
    const int64_t padding_idx) {
  TORCH_INTERNAL_ASSERT(weight.dim() == 2 && weight.is_contiguous());
  TORCH_INTERNAL_ASSERT(out.dim() == 2 && out.is_contiguous());
  TORCH_INTERNAL_ASSERT(out.scalar_type() == weight.scalar_type());

  TORCH_INTERNAL_ASSERT(offset2bag.dim() == 1 && offset2bag.is_contiguous());
  TORCH_INTERNAL_ASSERT(offset2bag.scalar_type() == ScalarType::Long);
  TORCH_INTERNAL_ASSERT(bag_size.dim() == 1 && bag_size.is_contiguous());
  TORCH_INTERNAL_ASSERT(bag_size.scalar_type() == ScalarType::Long);

  if (mode == native::EmbeddingBagMode::MAX) {
    TORCH_INTERNAL_ASSERT(max_indices.data_ptr() != nullptr);
    TORCH_INTERNAL_ASSERT(
        max_indices.dim() == 2 && max_indices.is_contiguous());
    TORCH_INTERNAL_ASSERT(bag_size.scalar_type() == ScalarType::Long);
  }

  TORCH_INTERNAL_ASSERT(indices.dim() == 1 && indices.is_contiguous());
  TORCH_INTERNAL_ASSERT(offsets.dim() == 1 && offsets.is_contiguous());

  if (mode != native::EmbeddingBagMode::SUM) {
    TORCH_INTERNAL_ASSERT(!per_sample_weights.defined());
  }
  if (per_sample_weights.defined()) {
    TORCH_INTERNAL_ASSERT(
        per_sample_weights.scalar_type() == weight.scalar_type());
  }

  const int64_t numBags = out.size(0);
  const int64_t featureSize = out.size(1);
  const int64_t numIndices = offset2bag.size(0);

  LaunchEmbeddingBagKnernel(
      out,
      offset2bag,
      bag_size,
      max_indices,
      weight,
      indices,
      offsets,
      per_sample_weights,
      mode,
      padding_idx,
      numBags,
      featureSize,
      numIndices);
}

} // namespace at::musa
