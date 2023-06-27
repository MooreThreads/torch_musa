#include <ATen/core/Tensor.h>
#include "torch_musa/csrc/aten/ops/Embedding.h"
#include <mudnn.h>
#include <ATen/musa/MUSAConfig.h>
#include <ATen/musa/cub.muh>
#include <musa_fp16.h>


namespace at {
namespace musa {
  
template <typename DataType, typename IndexType>
__global__ void EmbeddingKernel(DataType* out, const DataType* tbl,
                                const IndexType* idx, const int num_indices,
                                const int tbl_h, const int tbl_w,
                                const int padding_idx) {
  int ox = threadIdx.x;
  int oy = blockIdx.x + threadIdx.y * gridDim.x;

  for (; oy < num_indices; oy += blockDim.y * gridDim.x) {
    IndexType id = idx[oy];
    DataType* po = out + oy * tbl_w;
    const DataType* pt = tbl + id * tbl_w;
    bool valid_id = (id >= 0 && id < tbl_h && id != padding_idx);
    for (int i = ox; i < tbl_w; i += blockDim.x) {
      po[i] = valid_id ? pt[i] : DataType(0);
    }
  }
}

void EmbeddingRun(const Tensor& o, const Tensor& t, const Tensor& i, int64_t padding) {
    auto stream = c10::musa::getCurrentMUSAStream();

    int num_indices = i.numel();
    int tbl_h = t.size(0);
    int tbl_w = t.size(1);
    int padding_idx = static_cast<int>(padding);

    const uint32_t block_x = 128;
    const uint32_t block_y = 8;
    const uint32_t grid_x = (tbl_w + 128 - 1) / 128;
    const uint32_t grid_y = 1;

    dim3 block_size{block_x, block_y, 1};
    dim3 grid_size{grid_x, grid_y, 1};

    if (ScalarType::Float == t.scalar_type()) {
      if (ScalarType::Int == i.scalar_type()) {
        EmbeddingKernel<float, int32_t><<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(o.data_ptr()), static_cast<const float*>(t.data_ptr()),
            static_cast<const int32_t*>(i.data_ptr()), num_indices, tbl_h, tbl_w,
            padding_idx);
      } else {
        EmbeddingKernel<float, int64_t><<<grid_size, block_size, 0, stream>>>(
            static_cast<float*>(o.data_ptr()), static_cast<const float*>(t.data_ptr()),
            static_cast<const int64_t*>(i.data_ptr()), num_indices, tbl_h, tbl_w,
            padding_idx);
      }
    } else {
      if (ScalarType::Int == i.scalar_type()) {
        EmbeddingKernel<half, int32_t><<<grid_size, block_size, 0, stream>>>(
            static_cast<half*>(o.data_ptr()), static_cast<const half*>(t.data_ptr()),
            static_cast<const int32_t*>(i.data_ptr()), num_indices, tbl_h, tbl_w,
            padding_idx);
      } else {
        EmbeddingKernel<half, int64_t><<<grid_size, block_size, 0, stream>>>(
            static_cast<half*>(o.data_ptr()), static_cast<const half*>(t.data_ptr()),
            static_cast<const int64_t*>(i.data_ptr()), num_indices, tbl_h, tbl_w,
            padding_idx);
      }
    }
}

}  // namespace musa
}  // namespace at

