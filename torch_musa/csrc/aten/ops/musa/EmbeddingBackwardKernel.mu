#include <ATen/ATen.h>
//#include <ATen/AccumulateType.h>
#include <ATen/Dispatch.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/cub.h>

#include <ATen/musa/ThrustAllocator.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/ops/musa/EmbeddingBackwardKernel.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

namespace {

using float16_t = musa::float16_t;
using bfloat16_t = musa::bfloat16_t;

#if TORCH_MUSA_ARCH == 110
constexpr int MAX_BLOCK_SIZE = 16;
#elif TORCH_MUSA_ARCH == 210
constexpr int MAX_BLOCK_SIZE = 32;
#else
constexpr int MAX_BLOCK_SIZE = 64;
#endif

constexpr int NROWS_PER_THREAD = 10;

__host__ __device__ __forceinline__ int64_t ceil_div(int64_t x, int64_t y) {
  return (x + y - 1) / y;
}

template <typename index_t>
int64_t EmbeddingBackwardMUSAKernelUniqueByKey(
    const Tensor& sorted_indices,
    Tensor& segment_offsets) {
  auto stream = at::musa::getCurrentMUSAStream();
  at::musa::ThrustAllocator allocator;
  auto policy = thrust::musa::par(allocator).on(stream);
  const size_t numel = sorted_indices.numel();
  auto sorted_indices_dev =
      thrust::device_ptr<index_t>(sorted_indices.data_ptr<index_t>());
  Tensor dummy =
      at::empty_like(sorted_indices, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
  auto dummy_dev = thrust::device_ptr<index_t>(dummy.data_ptr<index_t>());
  auto ends = thrust::unique_by_key_copy(
      policy,
      sorted_indices_dev,
      sorted_indices_dev + numel,
      thrust::make_counting_iterator(0),
      dummy_dev,
      thrust::device_ptr<index_t>(segment_offsets.data_ptr<index_t>()));
  return thrust::get<0>(ends) - dummy_dev;
}

template <typename index_t>
__global__ void PartialsPerSegment(
    index_t* partials_per_segment,
    const index_t* segment_offsets,
    int32_t num_of_segments,
    const int64_t numel) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int gsz = blockDim.x * gridDim.x;
  for (int i = gid; i < num_of_segments; i += gsz) {
    const int64_t idx_start = segment_offsets[i];
    const int64_t idx_end =
        (i == num_of_segments - 1) ? numel : segment_offsets[i + 1];
    const int64_t size = idx_end - idx_start;
    partials_per_segment[i] = ceil_div(size, NROWS_PER_THREAD);
  }
}

template <typename index_t>
__global__ void PartialsSegmentOffset(
    int64_t* num_of_partial_segments,
    index_t* partial_segment_offset,
    const index_t* partials_per_segment,
    const index_t* partials_per_segment_offset,
    const index_t* segment_offsets,
    const int32_t num_of_segments) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int gsz = blockDim.x * gridDim.x;

  if (0 == gid) {
    // reduce one kernel launch overhead
    *num_of_partial_segments = partials_per_segment[num_of_segments - 1] +
        partials_per_segment_offset[num_of_segments - 1];
  }

  for (int i = gid; i < num_of_segments; i += gsz) {
    index_t idx = partials_per_segment_offset[i];
    const index_t num_partials = partials_per_segment[i];
    const index_t segment_offset = segment_offsets[i];
    for (int64_t j = 0; j < num_partials; j++) {
      partial_segment_offset[idx++] = segment_offset + j * NROWS_PER_THREAD;
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void ComputeDwSegment(
    scalar_t* dw_seg,
    const scalar_t* dy,
    const index_t* origin_idx,
    const index_t* partials_segment_offset,
    const int num_of_partial_segments,
    const int elements,
    const int tbl_w) {
  int m_idx = blockIdx.x;
  int m_step = gridDim.x;
  int n_idx = threadIdx.x;
  int n_step = blockDim.x;

  for (int m = m_idx; m < num_of_partial_segments; m += m_step) {
    int idx_start = partials_segment_offset[m];
    int idx_end = (m == num_of_partial_segments - 1)
        ? elements
        : partials_segment_offset[m + 1];
    scalar_t* pdw = dw_seg + m * tbl_w;
    for (int i = idx_start; i < idx_end; i++) {
      int origin_id = origin_idx[i];
      const scalar_t* pdy = dy + origin_id * tbl_w;
      for (int j = n_idx; j < tbl_w; j += n_step) {
        pdw[j] += pdy[j];
      }
    }
  }
}

template <typename scalar_t, typename index_t>
__global__ void SumAndScater(
    scalar_t* dw,
    const scalar_t* dw_segments,
    const index_t* idx,
    const index_t* segment_offset,
    const index_t* partials_per_segment_offset,
    const int num_of_segments,
    const int num_of_partial_segments,
    const int tbl_h,
    const int tbl_w,
    const int padding_idx) {
  int m_idx = blockIdx.x;
  int m_step = gridDim.x;
  int n_idx = threadIdx.x;
  int n_step = blockDim.x;

  for (int m = m_idx; m < num_of_segments; m += m_step) {
    int idx_start = partials_per_segment_offset[m];
    int idx_end = (m == num_of_segments - 1)
        ? num_of_partial_segments
        : partials_per_segment_offset[m + 1];
    index_t id = idx[segment_offset[m]];
    scalar_t* pdw = dw + id * tbl_w;
    bool valid_id = (id >= 0 && id < tbl_h && id != padding_idx);
    for (int i = idx_start; i < idx_end; i++) {
      const scalar_t* pdw_seg = dw_segments + i * tbl_w;
      for (int j = n_idx; j < tbl_w; j += n_step) {
        if (valid_id) {
          pdw[j] += pdw_seg[j];
        }
      }
    }
  }
}
} // namespace

Tensor EmbeddingBackwardMUSAKernel(
    const Tensor& grad,
    const Tensor& orig_indices,
    const Tensor& sorted_indices,
    int64_t num_weights,
    int padding_idx) {
  auto stream = at::musa::getCurrentMUSAStream();
  const ptrdiff_t numel = sorted_indices.numel();
  Tensor grad_weight = at::zeros({num_weights, grad.size(-1)}, grad.options());

  int tbl_h = grad_weight.size(0);
  int tbl_w = grad_weight.size(1);

  // Compute the number of segments and segment offsets
  auto segment_offsets = at::empty({numel}, orig_indices.options());
  int32_t num_of_segments = 0;

  AT_DISPATCH_INDEX_TYPES(
      orig_indices.scalar_type(), "EmbeddingBackwardMUSAKernel", [&]() {
        num_of_segments = EmbeddingBackwardMUSAKernelUniqueByKey<index_t>(
            sorted_indices, segment_offsets);
      });

  // Split the segments up into size of `NROWS_PER_THREAD`
  // Compute the number partial-segments per segment
  int64_t max_segment = std::min<int64_t>(numel, num_weights);
  auto partials_per_segment = at::empty({max_segment}, orig_indices.options());
  const int max_blocks = MAX_BLOCK_SIZE;
  uint32_t threads = 1024;
  uint32_t blocks =
      std::min(ceil_div(numel, threads), static_cast<int64_t>(max_blocks));

  AT_DISPATCH_INDEX_TYPES(
      orig_indices.scalar_type(), "EmbeddingBackwardMUSAKernel", [&]() {
        {
          PartialsPerSegment<<<blocks, threads, 0, stream>>>(
              partials_per_segment.data_ptr<index_t>(),
              segment_offsets.data_ptr<index_t>(),
              num_of_segments,
              numel);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        }

        auto partials_per_segment_offset =
            at::empty({max_segment}, orig_indices.options());

        at::musa::cub::exclusive_sum(
            partials_per_segment.data_ptr<index_t>(),
            partials_per_segment_offset.data_ptr<index_t>(),
            max_segment);

        // The total number of partial-segments is the sum of
        // `partials_per_segment_offset`
        auto num_of_partial_segments_tensor =
            at::empty({}, grad.options().dtype(kLong));
        int64_t num_of_partial_segments = 0;

        auto max_partial_segment = numel / NROWS_PER_THREAD + max_segment;

        auto partial_segment_offset =
            at::empty({max_partial_segment}, orig_indices.options());
        int64_t* num_of_partial_segments_ptr =
            num_of_partial_segments_tensor.data_ptr<int64_t>();

        {
          PartialsSegmentOffset<<<blocks, threads, 0, stream>>>(
              num_of_partial_segments_ptr,
              partial_segment_offset.data_ptr<index_t>(),
              partials_per_segment.data_ptr<index_t>(),
              partials_per_segment_offset.data_ptr<index_t>(),
              segment_offsets.data_ptr<index_t>(),
              num_of_segments);
          C10_MUSA_KERNEL_LAUNCH_CHECK();
        }

        musaMemcpyAsync(
            &num_of_partial_segments,
            static_cast<void*>(num_of_partial_segments_ptr),
            sizeof(int64_t),
            static_cast<musaMemcpyKind>(musaMemcpyDeviceToHost),
            stream);
        musaStreamSynchronize(stream);

        threads = 128;
        blocks =
            std::min(static_cast<int>(num_of_partial_segments), max_blocks);

        // TODO: check if numerical stability needs to be maintained as in the
        // CUDA implementation
        auto grad_weight_per_segment = at::zeros(
            {num_of_partial_segments, tbl_w}, grad.options().dtype(at::kFloat));
    // auto grad_weight_per_segment =
    //     at::zeros({num_of_partial_segments, tbl_w}, grad.options());

// 1. Compute the grad of each partial-segment
// 2. Sum all the partial-sums and scatter them into `grad_weight`
// For better performance, using float16_t/bfloat16_t rather than Half/BFloat16
#define DISPATCH_SEGMENT_KERNEL(SCALAR_TYPE)                        \
  do {                                                              \
    using scalar_t = SCALAR_TYPE;                                   \
    ComputeDwSegment<<<blocks, threads, 0, stream>>>(               \
        static_cast<scalar_t*>(grad_weight_per_segment.data_ptr()), \
        static_cast<scalar_t*>(grad.data_ptr()),                    \
        orig_indices.data_ptr<index_t>(),                           \
        partial_segment_offset.data_ptr<index_t>(),                 \
        num_of_partial_segments,                                    \
        numel,                                                      \
        tbl_w);                                                     \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                                 \
                                                                    \
    blocks = std::min(num_of_segments, max_blocks);                 \
                                                                    \
    SumAndScater<<<blocks, threads, 0, stream>>>(                   \
        static_cast<scalar_t*>(grad_weight.data_ptr()),             \
        static_cast<scalar_t*>(grad_weight_per_segment.data_ptr()), \
        sorted_indices.data_ptr<index_t>(),                         \
        segment_offsets.data_ptr<index_t>(),                        \
        partials_per_segment_offset.data_ptr<index_t>(),            \
        num_of_segments,                                            \
        num_of_partial_segments,                                    \
        tbl_h,                                                      \
        tbl_w,                                                      \
        padding_idx);                                               \
    C10_MUSA_KERNEL_LAUNCH_CHECK();                                 \
  } while (0)

        const auto& the_type = grad.scalar_type();
        switch (the_type) {
          case at::ScalarType::Double:
            DISPATCH_SEGMENT_KERNEL(
                c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Double>);
            break;
          case at::ScalarType::Float:
            DISPATCH_SEGMENT_KERNEL(
                c10::impl::ScalarTypeToCPPTypeT<at::ScalarType::Float>);
            break;
          case at::ScalarType::Half:
            DISPATCH_SEGMENT_KERNEL(float16_t);
            break;
          case at::ScalarType::BFloat16:
            DISPATCH_SEGMENT_KERNEL(bfloat16_t);
            break;
          default:
            AT_ERROR("EmbeddingDenseBwd not support ", toString(the_type));
        }
      });
#undef DISPATCH_SEGMENT_KERNEL
  return grad_weight;
}

} // namespace native
} // namespace at
