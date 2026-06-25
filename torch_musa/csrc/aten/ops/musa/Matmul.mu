#include <ATen/ATen.h>
#include <stdint.h>
#include "musa_runtime.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/quantized/mudnn/Qlinear.h"

namespace at::native {
__global__ void int8_gemm_naive_kernel(
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* __restrict__ mat1,
    int64_t lda,
    const int8_t* __restrict__ mat2,
    int64_t ldb,
    int32_t* __restrict__ result,
    int64_t ldc) {
  using vec4 = at::musa::Dtype<int8_t>::Vec4;
  int64_t stride_y = gridDim.y * blockDim.y;
  int64_t stride_x = gridDim.x * blockDim.x;

  int64_t start_row = blockIdx.y * blockDim.y + threadIdx.y;
  int64_t start_col = blockIdx.x * blockDim.x + threadIdx.x;

  for (int64_t row = start_row; row < m; row += stride_y) {
    for (int64_t col = start_col; col < n; col += stride_x) {
      int32_t sum = 0;
      int64_t i = 0;
      for (; i <= k - 4; i += 4) {
        vec4 a_vec = *(const vec4*)(&mat1[row * lda + i]);

        int8_t b_val0 = mat2[(i + 0) * ldb + col];
        int8_t b_val1 = mat2[(i + 1) * ldb + col];
        int8_t b_val2 = mat2[(i + 2) * ldb + col];
        int8_t b_val3 = mat2[(i + 3) * ldb + col];

        sum += static_cast<int32_t>(a_vec.x) * static_cast<int32_t>(b_val0);
        sum += static_cast<int32_t>(a_vec.y) * static_cast<int32_t>(b_val1);
        sum += static_cast<int32_t>(a_vec.z) * static_cast<int32_t>(b_val2);
        sum += static_cast<int32_t>(a_vec.w) * static_cast<int32_t>(b_val3);
      }

      for (; i < k; ++i) {
        int8_t a_val = mat1[row * lda + i];
        int8_t b_val = mat2[i * ldb + col];
        sum += static_cast<int32_t>(a_val) * static_cast<int32_t>(b_val);
      }
      result[row * ldc + col] = sum;
    }
  }
}

void int8_gemm(
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* mat1_ptr,
    int64_t mat1_ld,
    const int8_t* mat2_ptr,
    int64_t mat2_ld,
    int32_t* result_ptr,
    int64_t result_ld) {
  dim3 block(16, 16);

  dim3 grid(
      std::min<int>((n + block.x - 1) / block.x, 65535),
      std::min<int>((m + block.y - 1) / block.y, 65535));
  int8_gemm_naive_kernel<<<grid, block>>>(
      m, n, k, mat1_ptr, mat1_ld, mat2_ptr, mat2_ld, result_ptr, result_ld);
}

REGISTER_MUSA_DISPATCH(intmatmul_stub, &int8_gemm);

} // namespace at::native