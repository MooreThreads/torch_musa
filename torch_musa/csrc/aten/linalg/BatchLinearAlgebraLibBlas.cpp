#include "BatchLinearAlgebraLibBlas.h"
#include <ATen/Dispatch.h>
#include <ATen/musa/MUSAContext.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/musa/MiscUtils.h>
#include <ATen/ops/scalar_tensor.h>
#include <internal/mublas_types.h>
#include "torch_musa/csrc/aten/musa/MUSABlas.h"

namespace at::native {

static mublasOperation_t to_cublas(TransposeType trans) {
  switch (trans) {
    case TransposeType::NoTranspose:
      return MUBLAS_OP_N;
    case TransposeType::Transpose:
      return MUBLAS_OP_T;
    case TransposeType::ConjTranspose:
      return MUBLAS_OP_C;
  }
  TORCH_INTERNAL_ASSERT(false, "Invalid transpose type");
}

template <typename scalar_t>
static void apply_triangular_solve(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  mublasFillMode_t uplo =
      upper ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  const auto trans = to_cublas(transpose);
  mublasSideMode_t side = left ? MUBLAS_SIDE_LEFT : MUBLAS_SIDE_RIGHT;
  mublasDiagType_t diag =
      unitriangular ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  auto A_data = A.data_ptr<scalar_t>();
  auto B_data = B.data_ptr<scalar_t>();
  auto A_mat_stride = matrixStride(A);
  auto B_mat_stride = matrixStride(B);
  auto batch_size = batchCount(A);
  // This allows to pass rectangular A and B when left = True
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  auto n = cuda_int_cast(B.size(-1), "n");
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  auto alpha = scalar_t{1};

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* A_working_ptr = &A_data[i * A_mat_stride];
    scalar_t* B_working_ptr = &B_data[i * B_mat_stride];
    auto handle = at::musa::getCurrentMUSABlasHandle();
    at::musa::blas::trsm(
        handle,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        &alpha,
        A_working_ptr,
        lda,
        B_working_ptr,
        ldb);
  }
}

void triangular_solve_mublas(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      A.scalar_type(), "triangular_solve_musa", [&] {
        apply_triangular_solve<scalar_t>(
            A, B, left, upper, transpose, unitriangular);
      });
}

// Some muBLAS and muSOLVER batched routines require input to be a device array
// of pointers to device individual matrices 'input' must be a contiguous tensor
template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  int64_t input_mat_stride = matrixStride(input);

  // mublas/musolver interface requires 'int'
  int batch_size = cuda_int_cast(batchCount(input), "batch_size");

  // if batch_size==0, then start=0 and end=0
  // if input_mat_stride==0, then step=sizeof(scalar_t)
  return at::arange(
      /*start=*/reinterpret_cast<int64_t>(input_data),
      /*end=*/
      reinterpret_cast<int64_t>(input_data + batch_size * input_mat_stride),
      /*step=*/
      static_cast<int64_t>(
          std::max<int64_t>(input_mat_stride, 1) * sizeof(scalar_t)),
      input.options().dtype(at::kLong));
}

template <typename scalar_t>
static void apply_triangular_solve_batched(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  mublasFillMode_t uplo =
      upper ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  const auto trans = to_cublas(transpose);
  mublasSideMode_t side = left ? MUBLAS_SIDE_LEFT : MUBLAS_SIDE_RIGHT;
  mublasDiagType_t diag =
      unitriangular ? MUBLAS_DIAG_UNIT : MUBLAS_DIAG_NON_UNIT;

  auto batch_size = cuda_int_cast(batchCount(A), "batch_size");
  // This allows to pass rectangular A and B when left = True
  auto m = cuda_int_cast(left ? A.size(-1) : B.size(-2), "m");
  auto n = cuda_int_cast(B.size(-1), "n");
  auto lda = std::max<int>(1, cuda_int_cast(A.size(-2), "lda"));
  auto ldb = std::max<int>(1, cuda_int_cast(B.size(-2), "ldb"));

  auto alpha = scalar_t{1};

  // muBLAS batched trsm requires input to be the device array of pointers to
  // device single matrices
  Tensor A_ptr_array = get_device_pointers<scalar_t>(A);
  Tensor B_ptr_array = get_device_pointers<scalar_t>(B);
  auto A_ptr_array_data = reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr());
  auto B_ptr_array_data = reinterpret_cast<scalar_t**>(B_ptr_array.data_ptr());

  auto handle = at::musa::getCurrentMUSABlasHandle();
  at::musa::blas::trsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      &alpha,
      A_ptr_array_data,
      lda,
      B_ptr_array_data,
      ldb,
      batch_size);
}

void triangular_solve_batched_mublas(
    const Tensor& A,
    const Tensor& B,
    bool left,
    bool upper,
    TransposeType transpose,
    bool unitriangular) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      A.scalar_type(), "triangular_solve_musa", [&] {
        apply_triangular_solve_batched<scalar_t>(
            A, B, left, upper, transpose, unitriangular);
      });
}
} // namespace at::native