#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_DYNAMICCAST_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_DYNAMICCAST_H_

#include <c10/macros/Macros.h>
#include <ATen/native/musa/MemoryAccess.muh>

namespace at::musa {

#ifdef C10_HOST_DEVICE
#define ERROR_UNSUPPORTED_CAST CUDA_KERNEL_ASSERT(false);
#else
#define ERROR_UNSUPPORTED_CAST TORCH_CHECK(false, "Unexpected scalar type");
#endif

template <typename scalar_t, int vec_size>
using aligned_vector = native::memory::aligned_vector<scalar_t, vec_size>;

template <typename scalar_t, int vec_size>
__host__ __device__ inline aligned_vector<scalar_t, vec_size>
vectorized_by_scalar(scalar_t val) {
  aligned_vector<scalar_t, vec_size> ret;
#pragma unroll
  for (int i = 0; i < vec_size; ++i) {
    ret.val[i] = val;
  }
  return ret;
}

template <typename src_t, typename dst_t, int vec_size>
__host__ __device__ inline aligned_vector<dst_t, vec_size>
vectorized_fetch_and_cast_case(const void* ptr) {
  using src_vec_t = aligned_vector<src_t, vec_size>;
  using dst_vec_t = aligned_vector<dst_t, vec_size>;

  auto src = *(const src_vec_t*)ptr;

  if constexpr (std::is_same_v<src_t, dst_t>) {
    return src;
  } else {
    dst_vec_t ret;

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      auto* from = &(src.val[i]);
      ret.val[i] = c10::convert<dst_t>(c10::load<src_t>(from));
    }
    return ret;
  }
}

#define VEC_FETCH_CAST_CASE(cpptype, scalartype)                           \
  case ScalarType::scalartype: {                                           \
    return vectorized_fetch_and_cast_case<cpptype, dest_t, vec_size>(ptr); \
  }

template <typename dest_t, int vec_size>
__host__ __device__ inline aligned_vector<dest_t, vec_size>
vectorized_fetch_and_cast(ScalarType src_type, const void* ptr) {
  switch (src_type) {
    VEC_FETCH_CAST_CASE(c10::complex<float>, ComplexFloat)
    VEC_FETCH_CAST_CASE(c10::complex<double>, ComplexDouble)
    default:
      ERROR_UNSUPPORTED_CAST
  }

  return vectorized_by_scalar<dest_t, vec_size>(0);
}

#undef VEC_FETCH_CAST_CASE

template <typename src_t, typename dst_t, int vec_size>
__host__ __device__ inline void vectorized_cast_and_store_case(
    const src_t* src,
    void* dst) {
  using src_vec_t = aligned_vector<src_t, vec_size>;
  using dst_vec_t = aligned_vector<dst_t, vec_size>;

  if constexpr (std::is_same_v<src_t, dst_t>) {
    *(dst_vec_t*)dst = *(const src_vec_t*)src;
  } else {
    dst_vec_t tmp;

#pragma unroll
    for (int i = 0; i < vec_size; ++i) {
      tmp.val[i] = c10::convert<dst_t>(*(src + i));
    }
    *(dst_vec_t*)dst = tmp;
  }
}

#define VEC_CAST_STORE_CASE(cpptype, scalartype)                        \
  case ScalarType::scalartype: {                                        \
    vectorized_cast_and_store_case<src_t, cpptype, vec_size>(src, dst); \
    return;                                                             \
  }

template <typename src_t, int vec_size>
__host__ __device__ inline void vectorized_cast_and_store(
    ScalarType dest_type,
    void* dst,
    const src_t* src) {
  switch (dest_type) {
    VEC_CAST_STORE_CASE(c10::complex<float>, ComplexFloat)
    VEC_CAST_STORE_CASE(c10::complex<double>, ComplexDouble)
    VEC_CAST_STORE_CASE(int, Int)
    default:;
  }
  ERROR_UNSUPPORTED_CAST
}

#undef VEC_CAST_STORE_CASE

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_DYNAMICCAST_H_
