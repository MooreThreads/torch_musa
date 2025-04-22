#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/ReductionType.h>
#include <ATen/native/quantized/AffineQuantizerBase.h>
#include <ATen/native/quantized/IndexKernel.h>
#include <ATen/musa/detail/IndexUtils.muh>
#include <ATen/native/musa/KernelUtils.muh>
#include <ATen/native/musa/Loops.muh>

#include "torch_musa/csrc/aten/musa/MUSAAtomic.muh"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/detail/IndexUtils.muh"
#include "torch_musa/csrc/core/MUSAStream.h"

#ifndef GPU_LAMBDA
#define GPU_LAMBDA __host__ __device__
#endif

namespace at {
namespace musa {
using at::musa::ceil_div;
using at::native::ReductionType;
namespace {
constexpr int MAX_DIMS = 25;
using musa::detail::VariantTensorInfo;

class ReduceAdd {
 public:
  template <typename scalar_t>
  constexpr __device__ void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    (void)numel;
    *(self_data_start + index) += *src_data;
  }
};
class VectorizedReduceAdd {
 public:
  template <typename scalar_t, typename index_t, int vlen = 4, int iobits = 8>
  constexpr __device__ __forceinline__ void operator()(
      scalar_t* self_data,
      index_t self_offset,
      const scalar_t* src_data,
      index_t src_offset,
      const scalar_t& alpha) const {
    using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * iobits>;

    vec_dtype self_reg = vec_dtype::load(self_data, self_offset);
    vec_dtype src_reg = vec_dtype::load(src_data, src_offset);

#pragma unroll
    for (int k = 0; k < vlen; k++) {
      self_reg.val_.elem[k] += src_reg.val_.elem[k] * alpha;
    }
    vec_dtype::store(self_data, self_offset, self_reg);
  }
};

static ReduceAdd reduce_add;
static VectorizedReduceAdd vec_reduce_add;

class ReduceAtomicAdd {
 public:
  template <typename scalar_t>
  constexpr __device__ void operator()(
      scalar_t* self_data_start,
      int64_t index,
      int64_t numel,
      const scalar_t* src_data) const {
    at::native::fastAtomicAdd(self_data_start, index, numel, *src_data, true);
  }
};

class VectorizedReduceAtomicAdd {
 public:
  template <typename scalar_t, typename index_t, int vlen = 4, int iobits = 8>
  constexpr __device__ void operator()(
      scalar_t* self_data,
      index_t self_offset,
      const scalar_t* src_data,
      index_t src_offset,
      index_t numel,
      const scalar_t& alpha) const {
    using vec_dtype = VecType<scalar_t, vlen * sizeof(scalar_t) * iobits>;

    vec_dtype src_reg = vec_dtype::load(src_data, src_offset);
#pragma unroll
    for (index_t k = 0; k < vlen; k++) {
      at::native::fastAtomicAdd<scalar_t, index_t>(
          self_data,
          (index_t)self_offset + k,
          numel,
          src_reg.val_.elem[k] * alpha,
          true);
    }
  }
};

static ReduceAtomicAdd reduce_atomic_add;
static VectorizedReduceAtomicAdd vec_reduce_atomic_add;

// Check tensor dimensions for index operations, and return the slice size.
int64_t GetSliceSize(
    const Tensor& dst,
    int dim,
    const Tensor& index,
    const Tensor& src) {
  const auto dstDims = dst.dim();
  const auto srcDims = src.dim();

  TORCH_CHECK(index.dim() <= 1, "Index must be vector or scalar");

  int64_t dstSliceSize = 1;
  TORCH_CHECK(
      dim >= 0 && dim < dstDims, "Indexing dim ", dim, " is out of bounds");
  for (const auto d : c10::irange(dstDims)) {
    if (d != dim) {
      dstSliceSize *= dst.size(d);
    }
  }
  TORCH_CHECK(dim < srcDims, "Indexing dim ", dim, " is out of bounds");
  TORCH_CHECK(
      index.numel() == src.size(dim),
      "length of src.size[dim] is not equal to length of indices");

  int64_t srcSliceSize = 1;
  bool mismatch = false;
  if (dstDims != srcDims)
    mismatch = true;
  for (const auto d : c10::irange(srcDims)) {
    if (d != dim) {
      srcSliceSize *= src.size(d);
      if (!mismatch && dst.size(d) != src.size(d))
        mismatch = true;
    }
  }
  TORCH_CHECK(
      dstSliceSize == srcSliceSize,
      "Source/destination tensor have different slice sizes (%ld vs %ld)",
      dstSliceSize,
      srcSliceSize);

  TORCH_CHECK(
      !mismatch,
      "source/destination slices have different shape for an index_add operation");

  return dstSliceSize;
}

#define CALCULATE_OFFSET_THREAD_SCAN(x_idx, y_idx, z_idx)                     \
  IndexType idxs[3] = {x_idx, y_idx, z_idx};                                  \
  IndexType dstIndex = indices.data[srcIndex];                                \
  IndexType dstOffset = dstIndex * dst.strides[sliceDim];                     \
  IndexType srcOffset = srcIndex * src.strides[sliceDim];                     \
  if constexpr (IsLEThanThreeDims) {                                          \
    for (int i = 0; i < NDim; i++) {                                          \
      dstOffset += idxs[i] * dst.strides[i];                                  \
      srcOffset += idxs[i] * src.strides[i];                                  \
    }                                                                         \
  } else {                                                                    \
    for (int i = 0; i < 2; i++) {                                             \
      dstOffset += idxs[i] * dst.strides[i];                                  \
      srcOffset += idxs[i] * src.strides[i];                                  \
    }                                                                         \
    IndexType linear_idx = z_idx;                                             \
    for (int i = 2; i < MAX_DIMS; i++) {                                      \
      if (i == src.dims - 1) {                                                \
        break;                                                                \
      }                                                                       \
      DivmodHelper<IndexType> divmod = DivmodHelper<IndexType>(src.sizes[i]); \
      IndexType q, r;                                                         \
      divmod(q, r, linear_idx);                                               \
      dstOffset += r * dst.strides[i];                                        \
      srcOffset += r * src.strides[i];                                        \
      linear_idx = q;                                                         \
    }                                                                         \
  }

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanThreeDims,
    int vlen,
    typename vec_func_t,
    typename ew_func_t>
__global__ void IndexFuncsThreadScanIndicesVec3DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int sliceDim,
    int64_t xNumel,
    int64_t yNumel,
    int64_t zNumel,
    const vec_func_t& vec_op,
    const ew_func_t& ew_op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  IndexType g_z_idx = blockIdx.z * blockDim.z + threadIdx.z;

  IndexType contig_idx = g_x_idx * vlen;
  if ((contig_idx + vlen) <= xNumel && g_y_idx < yNumel && g_z_idx < zNumel) {
    for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
      CALCULATE_OFFSET_THREAD_SCAN(contig_idx, g_y_idx, g_z_idx);
      vec_op.template operator()<T, IndexType, vlen>(
          (T*)dst.data, dstOffset, (T*)src.data, srcOffset, alpha);
    }
  } else {
    while (contig_idx < xNumel && g_y_idx < yNumel && g_z_idx < zNumel) {
      for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
        CALCULATE_OFFSET_THREAD_SCAN(contig_idx, g_y_idx, g_z_idx);
        T val = src.data[srcOffset] * alpha;
        ew_op.template operator()<T>((T*)dst.data, dstOffset, 1, &val);
      }
      contig_idx++;
    }
  }
}

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanThreeDims,
    typename func_t>
__global__ void IndexFuncsThreadScanIndices3DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int sliceDim,
    int64_t xNumel,
    int64_t yNumel,
    int64_t zNumel,
    const func_t& op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  IndexType g_z_idx = blockIdx.z * blockDim.z + threadIdx.z;

  if (g_x_idx >= xNumel || g_y_idx >= yNumel || g_z_idx >= zNumel) {
    return;
  }
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    CALCULATE_OFFSET_THREAD_SCAN(g_x_idx, g_y_idx, g_z_idx);
    T val = src.data[srcOffset] * alpha;
    op.template operator()<T>((T*)dst.data, dstOffset, 0, &val);
  }
}

// This kernel in fact works for all choices of problem size, but if the
// non-indexed dimensions are aligned, IndexFuncsGridScanIndices[Vec3D/3DKernel]
// are better choice.
template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    int IdxDim,
    typename func_t>
__global__ void IndexFuncsThreadScanIndicesKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int dstAddDim,
    int srcAddDim,
    IndexType innerSize,
    int64_t dstNumel,
    const func_t& op,
    T alpha) {
  for (IndexType srcIndex = 0; srcIndex < indices.sizes[0]; ++srcIndex) {
    IndexType dstIndex = indices.data[srcIndex];
    for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
         linearIndex < innerSize;
         linearIndex += gridDim.x * blockDim.x) {
      IndexType dstOffset =
          musa::detail::IndexToOffset<T, IndexType, DstDim>::get(
              linearIndex, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          musa::detail::IndexToOffset<T, IndexType, SrcDim>::get(
              linearIndex, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }
  }
}

#define CALCULATE_OFFSET_BLOCK_SCAN(x_idx, y_idx)                             \
  IndexType idxs[2] = {x_idx, y_idx};                                         \
  IndexType dstIndex = indices.data[srcIndex];                                \
  IndexType srcOffset = srcIndex * src.strides[sliceDim];                     \
  IndexType dstOffset = dstIndex * dst.strides[sliceDim];                     \
  if constexpr (IsLEThanTwoDims) {                                            \
    for (int i = 0; i < NDim; i++) {                                          \
      srcOffset += idxs[i] * src.strides[i];                                  \
      dstOffset += idxs[i] * dst.strides[i];                                  \
    }                                                                         \
  } else {                                                                    \
    srcOffset += idxs[0] * src.strides[0];                                    \
    dstOffset += idxs[0] * dst.strides[0];                                    \
    IndexType y_idx = idxs[1];                                                \
    for (int i = 1; i < MAX_DIMS; i++) {                                      \
      if (i == (src.dims - 1)) {                                              \
        break;                                                                \
      }                                                                       \
      DivmodHelper<IndexType> divmod = DivmodHelper<IndexType>(src.sizes[i]); \
      IndexType q, r;                                                         \
      divmod(q, r, y_idx);                                                    \
      dstOffset += r * dst.strides[i];                                        \
      srcOffset += r * src.strides[i];                                        \
      y_idx = q;                                                              \
    }                                                                         \
  }

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanTwoDims,
    typename func_t>
__global__ void __attribute__((mtgpu_workgroup_atomic))
IndexFuncsBlockScanIndices2DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices, /*vector or scalar*/
    int sliceDim,
    int64_t dstNumel,
    int64_t xNumel,
    int64_t yNumel,
    const func_t& op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (g_x_idx >= xNumel || g_y_idx >= yNumel) {
    return;
  }

  for (IndexType srcIndex = threadIdx.z; srcIndex < indices.sizes[0];
       srcIndex += blockDim.z) {
    CALCULATE_OFFSET_BLOCK_SCAN(g_x_idx, g_y_idx);
    T val = src.data[srcOffset] * alpha;
    op(dst.data, dstOffset, dstNumel, &val);
  }
}

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanTwoDims,
    int vlen,
    typename vec_func_t,
    typename ew_func_t>
__global__ void __attribute__((mtgpu_workgroup_atomic))
IndexFuncsBlockScanIndicesVec2DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int sliceDim,
    int64_t dstNumel,
    int64_t xNumel,
    int64_t yNumel,
    const vec_func_t& vec_op,
    const ew_func_t& ew_op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  IndexType contig_idx = g_x_idx * vlen;
  if ((contig_idx + vlen) <= xNumel && g_y_idx < yNumel) {
    for (IndexType srcIndex = threadIdx.z; srcIndex < indices.sizes[0];
         srcIndex += blockDim.z) {
      CALCULATE_OFFSET_BLOCK_SCAN(contig_idx, g_y_idx);
      vec_op.template operator()<T, IndexType, vlen>(
          (T*)dst.data, dstOffset, (T*)src.data, srcOffset, dstNumel, alpha);
    }
  } else {
    while (contig_idx < xNumel && g_y_idx < yNumel) {
      for (IndexType srcIndex = threadIdx.z; srcIndex < indices.sizes[0];
           srcIndex += blockDim.z) {
        CALCULATE_OFFSET_BLOCK_SCAN(contig_idx, g_y_idx);
        T val = src.data[srcOffset] * alpha;
        ew_op.template operator()<T>((T*)dst.data, dstOffset, -1, &val);
      }
      contig_idx++;
    }
  }
}

// This kernel in fact works for all choices of problem size, but if the
// non-indexed dimensions are aligned,
// IndexFuncsBlockScanIndices[Vec2D/2DKernel] are better choice.
template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int DstDim,
    int SrcDim,
    typename func_t>
__global__ void __attribute__((mtgpu_workgroup_atomic))
IndexFuncsBlockScanIndicesKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int dstAddDim,
    int srcAddDim,
    int64_t dstNumel,
    int64_t sliceNumel,
    const func_t& op,
    T alpha) {
  for (IndexType elementInSlice = blockIdx.x * blockDim.x + threadIdx.x;
       elementInSlice < sliceNumel;
       elementInSlice += blockDim.x * gridDim.x) {
    IndexType srcIndex = threadIdx.y;
    for (; srcIndex < indices.sizes[0]; srcIndex += blockDim.y) {
      IndexType dstIndex = indices.data[srcIndex];
      IndexType dstOffset =
          musa::detail::IndexToOffset<T, IndexType, DstDim>::get(
              elementInSlice, dst);
      dstOffset += dstIndex * dst.strides[dstAddDim];

      IndexType srcOffset =
          musa::detail::IndexToOffset<T, IndexType, SrcDim>::get(
              elementInSlice, src);
      srcOffset += srcIndex * src.strides[srcAddDim];

      T val = src.data[srcOffset] * alpha;
      op(dst.data, dstOffset, dstNumel, &val);
    }
  }
}

#define CALCULATE_OFFSET_GRID_SCAN(x_idx, y_idx)                              \
  IndexType idxs[2] = {x_idx, y_idx};                                         \
  IndexType srcOffset = 0, dstOffset = 0, indexOffset = 0;                    \
  if constexpr (IsLEThanTwoDims) {                                            \
    for (int i = 0; i < NDim; i++) {                                          \
      srcOffset += idxs[i] * src.strides[i];                                  \
      dstOffset += idxs[i] * dst.strides[i];                                  \
      indexOffset += idxs[i] * indices.strides[i];                            \
    }                                                                         \
  } else {                                                                    \
    IndexType dividend_idx = idxs[1];                                         \
    srcOffset += idxs[0] * src.strides[0];                                    \
    dstOffset += idxs[0] * dst.strides[0];                                    \
    indexOffset += idxs[0] * indices.strides[0];                              \
    for (int i = 1; i < MAX_DIMS; i++) {                                      \
      if (i == src.dims) {                                                    \
        break;                                                                \
      }                                                                       \
      DivmodHelper<IndexType> divmod = DivmodHelper<IndexType>(src.sizes[i]); \
      IndexType q, r;                                                         \
      divmod(q, r, dividend_idx);                                             \
      dstOffset += r * dst.strides[i];                                        \
      srcOffset += r * src.strides[i];                                        \
      indexOffset += r * indices.strides[i];                                  \
      dividend_idx = q;                                                       \
    }                                                                         \
  }

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanTwoDims,
    int vlen,
    typename vec_func_t,
    typename ew_func_t>
__global__ void IndexFuncsGridScanIndicesVec2DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int64_t indexed_stride,
    int64_t dstNumel,
    int64_t xNumel,
    int64_t yNumel,
    const vec_func_t& vec_op,
    const ew_func_t& ew_op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  IndexType contig_idx = g_x_idx * vlen;
  if ((contig_idx + vlen) <= xNumel && g_y_idx < yNumel) {
    CALCULATE_OFFSET_GRID_SCAN(contig_idx, g_y_idx);
    dstOffset += indices.data[indexOffset] * indexed_stride;
    vec_op.template operator()<T, IndexType, vlen>(
        (T*)dst.data, dstOffset, (T*)src.data, srcOffset, dstNumel, alpha);
  } else {
    while (contig_idx < xNumel && g_y_idx < yNumel) {
      CALCULATE_OFFSET_GRID_SCAN(contig_idx, g_y_idx);
      dstOffset += indices.data[indexOffset] * indexed_stride;
      T val = src.data[srcOffset] * alpha;
      ew_op.template operator()<T>((T*)dst.data, dstOffset, -1, &val);
      contig_idx++;
    }
  }
}

template <
    typename T,
    typename IndicesType,
    typename IndexType,
    int NDim,
    bool IsLEThanTwoDims,
    typename ew_func_t>
__global__ void IndexFuncsGridScanIndices2DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int64_t indexed_stride,
    int64_t dstNumel,
    int64_t xNumel,
    int64_t yNumel,
    const ew_func_t& ew_op,
    T alpha) {
  IndexType g_x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  IndexType g_y_idx = blockIdx.y * blockDim.y + threadIdx.y;
  if (g_x_idx >= xNumel || g_y_idx >= yNumel) {
    return;
  }
  CALCULATE_OFFSET_GRID_SCAN(g_x_idx, g_y_idx);
  dstOffset += indices.data[indexOffset] * indexed_stride;
  T val = src.data[srcOffset] * alpha;
  ew_op.template operator()<T>((T*)dst.data, dstOffset, -1, &val);
}

// This kernel in fact works for all choices of problem size, but if the
// non-indexed dimensions are aligned, IndexFuncsGridScanIndices[Vec2D/2DKernel]
// are better choice.
template <
    typename T,
    typename IndicesType,
    typename IndexType,
    typename ew_func_t>
__global__ void IndexFuncsGridScanIndices1DKernel(
    VariantTensorInfo<T, IndexType> dst,
    VariantTensorInfo<T, IndexType> src,
    VariantTensorInfo<IndicesType, IndexType> indices,
    int64_t indexed_stride,
    int64_t dstNumel,
    int64_t sourceNumel,
    const ew_func_t& ew_op,
    T alpha) {
  for (IndexType linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
       linearIndex < sourceNumel;
       linearIndex += gridDim.x * blockDim.x) {
    IndexType indexOffset = 0, srcOffset = 0, dstOffset = 0;
    IndexType q, r;
    IndexType dividend_idx = linearIndex;
    for (int i = 0; i < src.dims; i++) {
      DivmodHelper<IndexType> divmod = DivmodHelper<IndexType>(src.sizes[i]);
      divmod(q, r, dividend_idx);
      srcOffset += r * src.strides[i];
      indexOffset += r * indices.strides[i];
      dividend_idx = q;
    }
    dividend_idx = linearIndex;
    for (int i = 0; i < dst.dims; i++) {
      DivmodHelper<IndexType> divmod = DivmodHelper<IndexType>(dst.sizes[i]);
      divmod(q, r, dividend_idx);
      dstOffset += r * dst.strides[i];
      dividend_idx = q;
    }
    dstOffset += indices.data[indexOffset] * indexed_stride;

    T val = src.data[srcOffset] * alpha;
    ew_op.template operator()<T>((T*)dst.data, dstOffset, -1, &val);
  }
}

template <typename T, typename IndexType>
bool IsNonSliceDimsSameAfterCollapse(
    musa::detail::TensorInfo<T, IndexType>& self,
    musa::detail::TensorInfo<T, IndexType>& source,
    int sliceDim) {
  if (self.dims != source.dims) {
    return false;
  }
  for (int i = 0; i < self.dims; i++) {
    if (i != sliceDim && self.sizes[i] != source.sizes[i]) {
      return false;
    }
  }
  return true;
}

} // anonymous namespace

template <
    bool is_index_add,
    typename vec_func_t,
    typename vec_atomic_func_t,
    typename ew_func_t,
    typename ew_atomic_func_t>
void IndexReduceFuncMUSAImpl(
    const Tensor& self,
    int64_t dim,
    const Tensor& index,
    const Tensor& source,
    bool include_self,
    const ReductionType& reduce,
    const vec_func_t& vec_reduce_func,
    const ew_func_t& ew_reduce_func,
    const vec_atomic_func_t& vec_atomic_reduce_func,
    const ew_atomic_func_t& ew_atomic_reduce_func,
    const Scalar& alpha,
    const Tensor& result) {
  if (!result.is_same(self)) {
    result.copy_(self);
  }

  const Tensor self_ = (result.dim() == 0) ? result.view(1) : result;
  const Tensor source_ = (source.dim() == 0) ? source.view(1) : source;
  TORCH_CHECK(
      result.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      source.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");
  TORCH_CHECK(
      index.dim() <= MAX_TENSORINFO_DIMS,
      "tensor has too many (>",
      MAX_TENSORINFO_DIMS,
      ") dims");

  if constexpr (is_index_add) {
    if (globalContext().deterministicAlgorithms()) {
      torch::List<c10::optional<Tensor>> indices;
      indices.reserve(dim + 1);
      for (const auto i : c10::irange(dim)) {
        indices.emplace_back();
      }
      indices.emplace_back(index.to(at::kLong));
      result.index_put_(indices, source * alpha, true);
      return;
    }
  } else {
    // Preprocessing logic reserved for index_reduce
    TORCH_CHECK(false, "index_reduce is not supported");
  }
  const int64_t sliceSize = GetSliceSize(self_, dim, index, source_);
  const int64_t sourceNumel = source.numel();
  const int64_t selfAddDimSize = self_.size(dim);
  const int64_t numIndices = index.numel();
  const int64_t selfNumel = self.numel();

  if (sliceSize == 0) {
    return;
  }

  const musaStream_t stream = at::musa::getCurrentMUSAStream();
  auto* device_prop = at::musa::getCurrentDeviceProperties();
  const int mp_count = device_prop->multiProcessorCount;
  const int device_major_version = device_prop->major;

#define LAUNCH_THREAD_SCAN_INDICES_VEC_3D_KERNEL(                        \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_THREE_DIMS, VLEN) \
  IndexFuncsThreadScanIndicesVec3DKernel<                                \
      TENSOR_TYPE,                                                       \
      INDICES_TYPE,                                                      \
      TYPE,                                                              \
      N_DIM,                                                             \
      IS_LE_THAN_THREE_DIMS,                                             \
      VLEN><<<grid, block, 0, stream>>>(                                 \
      varSelfInfo,                                                       \
      varSourceInfo,                                                     \
      varIndexInfo,                                                      \
      sliceDim,                                                          \
      xNumel,                                                            \
      yNumel,                                                            \
      zNumel,                                                            \
      vec_reduce_func,                                                   \
      ew_reduce_func,                                                    \
      alpha_value);                                                      \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_THREAD_SCAN_INDICES_3D_KERNEL(                      \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_THREE_DIMS) \
  IndexFuncsThreadScanIndices3DKernel<                             \
      TENSOR_TYPE,                                                 \
      INDICES_TYPE,                                                \
      TYPE,                                                        \
      N_DIM,                                                       \
      IS_LE_THAN_THREE_DIMS><<<grid, block, 0, stream>>>(          \
      varSelfInfo,                                                 \
      varSourceInfo,                                               \
      varIndexInfo,                                                \
      sliceDim,                                                    \
      xNumel,                                                      \
      yNumel,                                                      \
      zNumel,                                                      \
      ew_reduce_func,                                              \
      alpha_value);                                                \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_THREAD_SCAN_INDICES_KERNEL(                          \
    TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM, IDX_DIM) \
  IndexFuncsThreadScanIndicesKernel<                                \
      TENSOR_TYPE,                                                  \
      INDICES_TYPE,                                                 \
      TYPE,                                                         \
      SELF_DIM,                                                     \
      SOURCE_DIM,                                                   \
      IDX_DIM><<<grid, block, 0, stream>>>(                         \
      varSelfInfo,                                                  \
      varSourceInfo,                                                \
      varIndexInfo,                                                 \
      selfAddDim,                                                   \
      sourceAddDim,                                                 \
      sliceSize,                                                    \
      selfNumel,                                                    \
      ew_atomic_reduce_func,                                        \
      alpha_value);                                                 \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_BLOCK_SCAN_INDICES_2D_KERNEL(                     \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_TWO_DIMS) \
  IndexFuncsBlockScanIndices2DKernel<                            \
      TENSOR_TYPE,                                               \
      INDICES_TYPE,                                              \
      TYPE,                                                      \
      N_DIM,                                                     \
      IS_LE_THAN_TWO_DIMS><<<grid, block, 0, stream>>>(          \
      varSelfInfo,                                               \
      varSourceInfo,                                             \
      varIndexInfo,                                              \
      sliceDim,                                                  \
      selfNumel,                                                 \
      xNumel,                                                    \
      yNumel,                                                    \
      ew_atomic_reduce_func,                                     \
      alpha_value);                                              \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_BLOCK_SCAN_INDICES_VEC_2D_KERNEL(                       \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_TWO_DIMS, VLEN) \
  IndexFuncsBlockScanIndicesVec2DKernel<                               \
      TENSOR_TYPE,                                                     \
      INDICES_TYPE,                                                    \
      TYPE,                                                            \
      N_DIM,                                                           \
      IS_LE_THAN_TWO_DIMS,                                             \
      VLEN><<<grid, block, 0, stream>>>(                               \
      varSelfInfo,                                                     \
      varSourceInfo,                                                   \
      varIndexInfo,                                                    \
      sliceDim,                                                        \
      selfNumel,                                                       \
      xNumel,                                                          \
      yNumel,                                                          \
      vec_atomic_reduce_func,                                          \
      ew_atomic_reduce_func,                                           \
      alpha_value);                                                    \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_BLOCK_SCAN_INDICES_KERNEL(                  \
    TENSOR_TYPE, INDICES_TYPE, TYPE, SELF_DIM, SOURCE_DIM) \
  IndexFuncsBlockScanIndicesKernel<                        \
      TENSOR_TYPE,                                         \
      INDICES_TYPE,                                        \
      TYPE,                                                \
      SELF_DIM,                                            \
      SOURCE_DIM><<<grid, block, 0, stream>>>(             \
      varSelfInfo,                                         \
      varSourceInfo,                                       \
      varIndexInfo,                                        \
      selfAddDim,                                          \
      sourceAddDim,                                        \
      selfNumel,                                           \
      sliceSize,                                           \
      ew_atomic_reduce_func,                               \
      alpha_value);                                        \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_GRID_SCAN_INDICES_VEC_2D_KERNEL(                        \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_TWO_DIMS, VLEN) \
  IndexFuncsGridScanIndicesVec2DKernel<                                \
      TENSOR_TYPE,                                                     \
      INDICES_TYPE,                                                    \
      TYPE,                                                            \
      N_DIM,                                                           \
      IS_LE_THAN_TWO_DIMS,                                             \
      VLEN><<<grid, block, 0, stream>>>(                               \
      varSelfInfo,                                                     \
      varSourceInfo,                                                   \
      varIndexInfo,                                                    \
      indexedStride,                                                   \
      selfNumel,                                                       \
      xNumel,                                                          \
      yNumel,                                                          \
      vec_atomic_reduce_func,                                          \
      ew_atomic_reduce_func,                                           \
      alpha_value);                                                    \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_GRID_SCAN_INDICES_2D_KERNEL(                      \
    TENSOR_TYPE, INDICES_TYPE, TYPE, N_DIM, IS_LE_THAN_TWO_DIMS) \
  IndexFuncsGridScanIndices2DKernel<                             \
      TENSOR_TYPE,                                               \
      INDICES_TYPE,                                              \
      TYPE,                                                      \
      N_DIM,                                                     \
      IS_LE_THAN_TWO_DIMS><<<grid, block, 0, stream>>>(          \
      varSelfInfo,                                               \
      varSourceInfo,                                             \
      varIndexInfo,                                              \
      indexedStride,                                             \
      selfNumel,                                                 \
      xNumel,                                                    \
      yNumel,                                                    \
      ew_atomic_reduce_func,                                     \
      alpha_value);                                              \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

#define LAUNCH_GRID_SCAN_INDICES_1D_KERNEL(TENSOR_TYPE, INDICES_TYPE, TYPE) \
  IndexFuncsGridScanIndices1DKernel<TENSOR_TYPE, INDICES_TYPE, TYPE>        \
      <<<grid, block, 0, stream>>>(                                         \
          varSelfInfo,                                                      \
          varSourceInfo,                                                    \
          varIndexInfo,                                                     \
          indexedStride,                                                    \
          selfNumel,                                                        \
          sourceNumel,                                                      \
          ew_atomic_reduce_func,                                            \
          alpha_value);                                                     \
  C10_MUSA_KERNEL_LAUNCH_CHECK();

  if (musa::detail::canUse32BitIndexMath(result) &&
      musa::detail::canUse32BitIndexMath(source) &&
      musa::detail::canUse32BitIndexMath(index)) {
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        result.scalar_type(),
        "index_add",
        [&] {
          musa::detail::VariantTensorInfo<scalar_t, uint32_t> varSelfInfo =
              musa::detail::getTensorInfo<scalar_t, uint32_t>(self_);

          int originSelfAddDim = varSelfInfo.collapseDims(dim);
          const auto alpha_value = alpha.to<scalar_t>();

          AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_musa_", [&] {
            musa::detail::VariantTensorInfo<scalar_t, uint32_t> varSourceInfo =
                musa::detail::getTensorInfo<scalar_t, uint32_t>(source_);
            int originSourceAddDim = varSourceInfo.collapseDims(dim);

            musa::detail::VariantTensorInfo<index_t, uint32_t> varIndexInfo =
                musa::detail::getTensorInfo<index_t, uint32_t>(index);
            varIndexInfo.collapseDims();

            // all other dimensions match except dim-th dimension after
            // dimension collapseing
            const bool has_regular_dims =
                (originSelfAddDim == originSourceAddDim &&
                 IsNonSliceDimsSameAfterCollapse(
                     varSelfInfo, varSourceInfo, originSelfAddDim));
            varSelfInfo.Fill();
            varSourceInfo.Fill();
            // Put the indexed dimension to the end
            varSelfInfo.Swap(originSelfAddDim, varSelfInfo.dims - 1);
            varSourceInfo.Swap(originSourceAddDim, varSourceInfo.dims - 1);

            int contig_dim = -1;
            bool vec_load_eligible = false;
            if (has_regular_dims) {
              for (int i = 0; i < varSelfInfo.dims - 1; i++) {
                if (varSelfInfo.strides[i] == 1 &&
                    varSourceInfo.strides[i] == 1) {
                  contig_dim = i;
                  break;
                }
              }
              vec_load_eligible = contig_dim != -1;
              if (vec_load_eligible) {
                varSelfInfo.Swap(contig_dim, 0);
                varSourceInfo.Swap(contig_dim, 0);
              }
            }

            constexpr int64_t vlen = sizeof(scalar_t) <= 4
                ? (sizeof(scalar_t) <= 2 ? (sizeof(scalar_t) <= 1 ? 16 : 8) : 4)
                : 2;

            const bool run_non_atomic =
                (device_major_version >= 3 && numIndices <= 16) ||
                (device_major_version < 3 && numIndices <= 1024 &&
                 sliceSize >= mp_count * /*maxThreadsPerBlock*/ 1024);
            const bool run_global_atomic = !run_non_atomic &&
                (device_major_version >= 3 || numIndices > 8192);
            if (run_non_atomic) {
              if (has_regular_dims && varSourceInfo.dims > 1) {
                int sliceDim = varSourceInfo.dims - 1;
                dim3 block = {1, 1, 1};
                dim3 grid = {1, 1, 1};
                int64_t xNumel = varSourceInfo.sizes[0];
                int64_t yNumel =
                    varSourceInfo.dims <= 2 ? 1 : varSourceInfo.sizes[1];
                int64_t zNumel = sliceSize / (xNumel * yNumel);

                block.x = 128;
                block.y = std::min((int64_t)(1024 / block.x), yNumel);
                grid.x = ceil_div(
                    xNumel, (int64_t)block.x * (vec_load_eligible ? vlen : 1));
                grid.y = ceil_div(yNumel, (int64_t)block.y);
                grid.z = ceil_div(zNumel, (int64_t)block.z);

                if (vec_load_eligible) {
                  if (varSourceInfo.dims == 2) {
                    LAUNCH_THREAD_SCAN_INDICES_VEC_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 1, true, vlen);
                  } else if (varSourceInfo.dims == 3) {
                    LAUNCH_THREAD_SCAN_INDICES_VEC_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true, vlen);
                  } else if (varSourceInfo.dims == 4) {
                    LAUNCH_THREAD_SCAN_INDICES_VEC_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 3, true, vlen);
                  } else {
                    LAUNCH_THREAD_SCAN_INDICES_VEC_3D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false, vlen);
                  }
                } else {
                  if (varSourceInfo.dims == 2) {
                    LAUNCH_THREAD_SCAN_INDICES_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 1, true);
                  } else if (varSourceInfo.dims == 3) {
                    LAUNCH_THREAD_SCAN_INDICES_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true);
                  } else if (varSourceInfo.dims == 4) {
                    LAUNCH_THREAD_SCAN_INDICES_3D_KERNEL(
                        scalar_t, index_t, uint32_t, 3, true);
                  } else {
                    LAUNCH_THREAD_SCAN_INDICES_3D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false);
                  }
                }
              } else {
                // fallback to the linear indexing kernel, including 1-d and
                // other unregular cases
                int64_t selfAddDim = varSelfInfo.dims - 1;
                int64_t sourceAddDim = varSourceInfo.dims - 1;
                // use the way of torch's IndexToOffset, the size of indexed
                // dimension needs to be set to 1
                varSourceInfo.reduceDim(sourceAddDim);
                varSelfInfo.reduceDim(selfAddDim);
                const dim3 grid(std::min(
                    ceil_div(sliceSize, (int64_t)128),
                    (int64_t)(mp_count * 8)));
                const dim3 block(std::min(sliceSize, (int64_t)128));
                LAUNCH_THREAD_SCAN_INDICES_KERNEL(
                    scalar_t, index_t, uint32_t, -1, -1, -1);
              }
            } else if (!run_global_atomic) {
              // workgroup atomic path
              if (has_regular_dims) {
                int sliceDim = varSourceInfo.dims - 1;
                int64_t xNumel =
                    varSourceInfo.dims == 1 ? 1 : varSourceInfo.sizes[0];
                int64_t yNumel = sliceSize / xNumel;

                // the number of block at indexed dimension should be 1 in this
                // path, now always regrad z-dim as indexed dimension
                dim3 block = {1, 1, 8};
                dim3 grid = {1, 1, 1};
                int num_thread_x =
                    ceil_div(xNumel, vec_load_eligible ? vlen : 1);
                if (num_thread_x <= 128) {
                  block.x = 32;
                } else if (num_thread_x <= 256) {
                  block.x = 64;
                } else {
                  block.x = 128;
                }
                block.x = std::min(block.x, 1024 / block.z);
                grid.x = ceil_div(
                    xNumel, (int64_t)block.x * (vec_load_eligible ? vlen : 1));
                grid.y = ceil_div(yNumel, (int64_t)block.y);

                if (vec_load_eligible) {
                  if (varSourceInfo.dims == 2) {
                    LAUNCH_BLOCK_SCAN_INDICES_VEC_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 1, true, vlen);
                  } else if (varSourceInfo.dims == 3) {
                    LAUNCH_BLOCK_SCAN_INDICES_VEC_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true, vlen);
                  } else {
                    LAUNCH_BLOCK_SCAN_INDICES_VEC_2D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false, vlen);
                  }
                } else {
                  if (varSourceInfo.dims == 1) {
                    LAUNCH_BLOCK_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 0, true);
                  } else if (varSourceInfo.dims == 2) {
                    LAUNCH_BLOCK_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 1, true);
                  } else if (varSourceInfo.dims == 3) {
                    LAUNCH_BLOCK_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true);
                  } else {
                    LAUNCH_BLOCK_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false);
                  }
                }
              } else {
                int64_t selfAddDim = varSelfInfo.dims - 1;
                int64_t sourceAddDim = varSourceInfo.dims - 1;
                varSourceInfo.reduceDim(sourceAddDim);
                varSelfInfo.reduceDim(selfAddDim);
                const dim3 block = {128, 8, 1};
                dim3 grid = {1, 1, 1};
                grid.x = ceil_div(sliceSize, (int64_t)128);
                LAUNCH_BLOCK_SCAN_INDICES_KERNEL(
                    scalar_t, index_t, uint32_t, -1, -1);
              }
            } else {
              // global atomic path
              if (has_regular_dims) {
                // use 2D thread block to ensure that vectorization cases could
                // be landed, and also avoid the cases that too many idle
                // threads exists.
                int sliceDim = varSourceInfo.dims - 1;
                varIndexInfo.Fill();
                varIndexInfo.Swap(0, sliceDim);
                varIndexInfo.dims = varSourceInfo.dims;

                int64_t indexedStride = varSelfInfo.strides[sliceDim];
                varSelfInfo.strides[sliceDim] = 0;

                int64_t xNumel = varSourceInfo.dims == 1
                    ? sourceNumel
                    : varSourceInfo.sizes[0];
                int64_t yNumel = sourceNumel / xNumel;

                dim3 block = {1, 1, 1};
                dim3 grid = {1, 1, 1};
                int num_thread_x =
                    ceil_div(xNumel, vec_load_eligible ? vlen : 1);
                if (num_thread_x <= 128) {
                  block.x = 32;
                } else if (num_thread_x <= 256) {
                  block.x = 64;
                } else {
                  block.x = 128;
                }
                block.y = 1024 / block.x;
                grid.x = ceil_div(
                    xNumel, (int64_t)block.x * (vec_load_eligible ? vlen : 1));
                grid.y = ceil_div(yNumel, (int64_t)block.y);

                if (vec_load_eligible) {
                  if (varSourceInfo.dims == 2) {
                    LAUNCH_GRID_SCAN_INDICES_VEC_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true, vlen);
                  } else {
                    LAUNCH_GRID_SCAN_INDICES_VEC_2D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false, vlen);
                  }
                } else {
                  if (varSourceInfo.dims == 1) {
                    LAUNCH_GRID_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 1, true);
                  } else if (varSourceInfo.dims == 2) {
                    LAUNCH_GRID_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, 2, true);
                  } else {
                    LAUNCH_GRID_SCAN_INDICES_2D_KERNEL(
                        scalar_t, index_t, uint32_t, -1, false);
                  }
                }
              } else {
                int dstSliceDim = varSelfInfo.dims - 1;
                int srcSliceDim = varSourceInfo.dims - 1;
                varIndexInfo.Fill();
                varIndexInfo.Swap(0, srcSliceDim);
                varIndexInfo.dims = varSourceInfo.dims;

                int64_t indexedStride = varSelfInfo.strides[dstSliceDim];
                varSelfInfo.strides[dstSliceDim] = 0;

                const dim3 block(128);
                const dim3 grid(std::min(
                    ceil_div(sourceNumel, (int64_t)block.x),
                    (int64_t)(mp_count * 8)));
                LAUNCH_GRID_SCAN_INDICES_1D_KERNEL(scalar_t, index_t, uint32_t);
              }
            }
          });
        });
  } else {
    // 64 bit indexing
    AT_DISPATCH_ALL_TYPES_AND3(
        at::ScalarType::Bool,
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        result.scalar_type(),
        "index_add",
        [&]() {
          musa::detail::VariantTensorInfo<scalar_t, uint64_t> varSelfInfo =
              musa::detail::getTensorInfo<scalar_t, uint64_t>(self_);
          const int selfAddDim = varSelfInfo.collapseDims(dim);
          const auto alpha_value = alpha.to<scalar_t>();

          musa::detail::VariantTensorInfo<scalar_t, uint64_t> varSourceInfo =
              musa::detail::getTensorInfo<scalar_t, uint64_t>(source_);
          const int sourceAddDim = varSourceInfo.collapseDims(dim);

          AT_DISPATCH_INDEX_TYPES(index.scalar_type(), "index_add_musa_", [&] {
            musa::detail::VariantTensorInfo<index_t, uint64_t> varIndexInfo =
                musa::detail::getTensorInfo<index_t, uint64_t>(index);
            varIndexInfo.collapseDims();

            int dstSliceDim = selfAddDim;
            int srcSliceDim = sourceAddDim;
            varIndexInfo.Fill();
            varIndexInfo.Swap(0, srcSliceDim);
            varIndexInfo.dims = varSourceInfo.dims;

            int64_t indexedStride = varSelfInfo.strides[dstSliceDim];
            varSelfInfo.strides[dstSliceDim] = 0;

            const dim3 block(128);
            const dim3 grid(std::min(
                ceil_div(sourceNumel, (int64_t)block.x),
                (int64_t)(mp_count * 8)));
            LAUNCH_GRID_SCAN_INDICES_1D_KERNEL(scalar_t, index_t, uint64_t);
          });
        });
  }
}

TORCH_IMPL_FUNC(index_add_musa_out)
(const Tensor& self,
 int64_t dim,
 const Tensor& index,
 const Tensor& source,
 const Scalar& alpha,
 const Tensor& result) {
  IndexReduceFuncMUSAImpl<true>(
      self,
      dim,
      index,
      source,
      true,
      ReductionType::SUM,
      vec_reduce_add,
      reduce_add,
      vec_reduce_atomic_add,
      reduce_atomic_add,
      alpha,
      result);
}

template <typename scalar_t>
void MusaMaskedFillKernelQuantized(
    TensorIterator& iter,
    scalar_t quantized_val) {
  at::native::gpu_kernel(
      iter, [quantized_val] GPU_LAMBDA(scalar_t self, bool mask) -> scalar_t {
        if (mask) {
          return quantized_val;
        }
        return self;
      });
}

void MaskedFillKernelQuantized(
    TensorIterator& iter,
    const Scalar& value,
    double scale,
    int zero_point) {
  TORCH_CHECK(
      iter.input_dtype(1) == at::ScalarType::Bool,
      "masked_fill only supports boolean masks, ",
      "but got dtype ",
      iter.input_dtype(1));
  AT_DISPATCH_QINT_TYPES(iter.common_dtype(), "masked_fill_", [&]() {
    float float_val = value.to<float>();
    const auto quantized_val =
        at::native::quantize_val<scalar_t>(scale, zero_point, float_val);

    MusaMaskedFillKernelQuantized<scalar_t>(iter, quantized_val);
  });
}

} // namespace musa

// register dispatch stub in native namespace
namespace native {

REGISTER_MUSA_DISPATCH(
    masked_fill_kernel_quantized_stub,
    &at::musa::MaskedFillKernelQuantized);

} // namespace native
} // namespace at
