// See Note [BatchLinearAlgebraLib split implementation files]
#define TORCH_ASSERT_ONLY_METHOD_OPERATORS
#include <ATen/Context.h>
#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <c10/util/irange.h>

#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/native/TransposeType.h>
#include <ATen/native/musa/MiscUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/ops/arange.h>
#include <ATen/ops/nan_to_num.h>
#include <ATen/ops/scalar_tensor.h>
#include <ATen/ops/where.h>
#endif

#include "torch_musa/csrc/aten/linalg/MUSASolver.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at::musa {

using at::native::TransposeType;

template <typename scalar_t>
static Tensor get_device_pointers(const Tensor& input) {
  auto input_data = input.const_data_ptr<scalar_t>();
  int64_t input_mat_stride = at::native::matrixStride(input);

  // cublas/cusolver interface requires 'int'
  int batch_size =
      at::native::cuda_int_cast(at::native::batchCount(input), "batch_size");

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

static mublasOperation_t to_mublas(TransposeType trans) {
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

// Implementation of Cholesky decomposition using batched
// musolver<T>potrfBatched Warning from PyTorch: cusolverDn<T>potrfBatched
// doesn't work quite well when matrix size or batch size is zero. If you write
// your own C++ extension and use this function, make sure you do a zero numel
// check for the input.
template <typename scalar_t>
inline static void apply_cholesky_musolver_potrfBatched(
    const Tensor& self_working_copy,
    bool upper,
    const Tensor& infos) {
  auto handle = at::musa::getCurrentMUSABlasHandle();
  const auto uplo = upper ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  const int n = at::native::cuda_int_cast(self_working_copy.size(-1), "n");
  const int lda = std::max<int>(1, n);

  const int batch_size = at::native::cuda_int_cast(
      at::native::batchCount(self_working_copy), "batch_size");

  // musolver batched kernels require input be "device array of device pointers"
  Tensor self_working_copy_array =
      get_device_pointers<scalar_t>(self_working_copy);

  at::musa::solver::potrfBatched<scalar_t>(
      handle,
      uplo,
      n,
      reinterpret_cast<scalar_t**>(self_working_copy_array.data_ptr()),
      lda,
      infos.data_ptr<int>(),
      batch_size);
}

void cholesky_helper_musolver(
    const Tensor& input,
    bool upper,
    const Tensor& info) {
  if (input.numel() == 0) {
    return;
  }

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "cholesky_musolver", [&] {
        apply_cholesky_musolver_potrfBatched<scalar_t>(input, upper, info);
      });
}

// The 'apply_' word is used for templated by dtype functions that call an API
// routine underneath. Since the cusolver API has a slightly different structure
// we do not prepend apply_ to this function.
void lu_factor_looped_musolver(
    const Tensor& self,
    const Tensor& pivots,
    const Tensor& infos,
    bool get_pivots) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      self.scalar_type(),
      "lu_factor_musolver",
      [&self, &pivots, &infos, get_pivots]() {
        const auto m = at::native::cuda_int_cast(self.size(-2), "m");
        const auto n = at::native::cuda_int_cast(self.size(-1), "n");
        const auto lda = std::max<int>(1, m);
        const auto self_stride = at::native::matrixStride(self);
        const auto batch_size = at::native::batchCount(self);
        const auto self_data = self.data_ptr<scalar_t>();
        const auto infos_data = infos.data_ptr<int>();

        const auto pivots_data = get_pivots ? pivots.data_ptr<int>() : nullptr;
        const auto pivots_stride = get_pivots ? pivots.size(-1) : 0;

        const auto handle = at::musa::getCurrentMUSABlasHandle();
        for (auto batch = decltype(batch_size){0}; batch < batch_size;
             ++batch) {
          at::musa::solver::getrf<scalar_t>(
              handle,
              m,
              n,
              self_data + batch * self_stride,
              lda,
              get_pivots ? pivots_data + batch * pivots_stride : nullptr,
              infos_data + batch);
        }
      });

  // TODO(@ai-infra): This is a workaround for the issue that cuSOLVER uses
  // nan for outputs that correspond to 0 in MAGMA for non-pivoted LU.
  // https://github.com/pytorch/pytorch/issues/53879#issuecomment-830633572
  if (!get_pivots) {
    // nan_to_num does not work for complex inputs
    // https://github.com/pytorch/pytorch/issues/59247
    if (self.is_complex()) {
      self.copy_(at::where(
          self.eq(self), self, at::scalar_tensor(0., self.options())));
    } else {
      at::nan_to_num_(
          const_cast<Tensor&>(self),
          0,
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity());
    }
  }
}

template <typename scalar_t>
inline static void apply_cholesky_musolver_potrs(
    Tensor& self_working_copy,
    const Tensor& A_column_major_copy,
    bool upper,
    Tensor& infos) {
  auto handle = at::musa::getCurrentMUSABlasHandle();
  const auto uplo = upper ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = at::native::batchCount(self_working_copy);
  const int64_t self_matrix_stride =
      at::native::matrixStride(self_working_copy);
  scalar_t* self_working_copy_ptr = self_working_copy.data_ptr<scalar_t>();

  scalar_t* A_ptr = A_column_major_copy.data_ptr<scalar_t>();
  const int64_t A_matrix_stride = at::native::matrixStride(A_column_major_copy);
  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  int* infos_ptr = infos.data_ptr<int>();

  int n_32 = at::native::cuda_int_cast(n, "n");
  int nrhs_32 = at::native::cuda_int_cast(nrhs, "nrhs");
  int lda_32 = at::native::cuda_int_cast(lda, "lda");
  int ldb_32 = at::native::cuda_int_cast(ldb, "ldb");

  for (int64_t i = 0; i < batch_size; i++) {
    at::musa::solver::potrs<scalar_t>(
        handle,
        uplo,
        n_32,
        nrhs_32,
        A_ptr + i * A_matrix_stride,
        lda_32,
        self_working_copy_ptr + i * self_matrix_stride,
        ldb_32,
        infos_ptr);
  }
}

template <typename scalar_t>
static void apply_cholesky_musolver_potrsBatched(
    Tensor& self_working_copy,
    const Tensor& A_column_major_copy,
    bool upper,
    Tensor& infos) {
  auto handle = at::musa::getCurrentMUSABlasHandle();
  const auto uplo = upper ? MUBLAS_FILL_MODE_UPPER : MUBLAS_FILL_MODE_LOWER;
  const int64_t n = self_working_copy.size(-2);
  const int64_t nrhs = self_working_copy.size(-1);
  const int64_t lda = std::max<int64_t>(1, n);
  const int64_t batch_size = at::native::batchCount(self_working_copy);

  const int64_t ldb = std::max<int64_t>(1, A_column_major_copy.size(-1));

  int* infos_ptr = infos.data_ptr<int>();

  auto self_ptr_array = get_device_pointers<scalar_t>(self_working_copy);
  auto A_ptr_array = get_device_pointers<scalar_t>(A_column_major_copy);

  at::musa::solver::potrsBatched(
      handle,
      uplo,
      at::native::cuda_int_cast(n, "n"),
      at::native::cuda_int_cast(nrhs, "nrhs"),
      reinterpret_cast<scalar_t**>(A_ptr_array.data_ptr()),
      at::native::cuda_int_cast(lda, "lda"),
      reinterpret_cast<scalar_t**>(self_ptr_array.data_ptr()),
      at::native::cuda_int_cast(ldb, "ldb"),
      infos_ptr,
      at::native::cuda_int_cast(batch_size, "batch_size"));
}

Tensor& cholesky_inverse_kernel_impl_musolver(
    Tensor& result,
    Tensor& infos,
    bool upper) {
  at::Tensor input_working_copy = at::native::cloneBatchedColumnMajor(result);
  at::Tensor infos_gpu = at::zeros({1}, result.options().dtype(at::kInt));
  result.fill_(0);
  result.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "cholesky_musa_potri", [&] {
        apply_cholesky_musolver_potrs<scalar_t>(
            result, input_working_copy, upper, infos_gpu);
      });

  // Debug only: info of cusolver potrs only check if the i-th parameter is
  // wrong Function argument `infos` is a CPU tensor, the following copy will
  // cause a device-host sync. infos.copy_(infos_gpu);
  return result;
}

void lu_solve_looped_musolver(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    TransposeType transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      LU.scalar_type(), "lu_solve_musolver", [&] {
        const auto trans = to_mublas(transpose);
        int n = at::native::cuda_int_cast(LU.size(-2), "n");
        int nrhs = at::native::cuda_int_cast(B.size(-1), "nrhs");
        auto batch_size = at::native::batchCount(B);
        auto info = at::zeros({1}, LU.options().dtype(kInt));
        auto info_data = info.data_ptr<int>();
        auto b_data = B.data_ptr<scalar_t>();
        auto lu_data = LU.data_ptr<scalar_t>();
        auto pivots_data = pivots.data_ptr<int>();
        auto pivots_stride = pivots.dim() > 1 ? pivots.stride(-2) : 0;
        auto lu_stride = LU.dim() > 2 ? LU.stride(-3) : 0;
        auto b_stride = at::native::matrixStride(B);
        int leading_dimension =
            at::native::cuda_int_cast(std::max<int>(1, n), "leading_dimension");

        // lu and pivots tensors can be broadcast to b
        // here we construct a helper indexing tensor to linearly index into lu
        // and pivots
        IntArrayRef lu_batch_shape(LU.sizes().data(), LU.dim() - 2);
        IntArrayRef b_batch_shape(B.sizes().data(), B.dim() - 2);
        at::native::BroadcastLinearIndices lu_index(
            at::native::batchCount(LU), lu_batch_shape, b_batch_shape);

        auto handle = at::musa::getCurrentMUSABlasHandle();
        for (auto batch = decltype(batch_size){0}; batch < batch_size;
             ++batch) {
          int64_t lu_index_i = lu_index(batch);
          at::musa::solver::getrs<scalar_t>(
              handle,
              n,
              nrhs,
              lu_data + lu_index_i * lu_stride,
              leading_dimension,
              pivots_data + lu_index_i * pivots_stride,
              b_data + batch * b_stride,
              leading_dimension,
              info_data,
              trans);

          TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
        }
      });
}

/*
  The geqrf function computes the QR decomposition of a m x n matrix A.

  Args:
  * `A` - [in] Tensor with matrices for QR decomposition,
          [out] Tensor containing R in the upper triangle of A
          and elementary reflectors below the main diagonal of A
  * `tau` - Tensor containing the magnitudes of the elementary reflectors
  * `m` - The number of rows of `input` to consider
  * `n` - The number of columns of `input` to consider (actual sizes of `input`
  could be larger)

  For further details, please see the cuSOLVER documentation for GEQRF.
*/
template <typename scalar_t>
static void apply_geqrf(const Tensor& A, const Tensor& tau) {
  int64_t m = A.size(-2);
  int64_t n = A.size(-1);
  int64_t lda = std::max<int64_t>(1, m);
  int64_t batch_size = at::native::batchCount(A);

  auto A_stride = at::native::matrixStride(A);
  auto tau_stride = tau.size(-1);

  auto A_data = A.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();

  auto infos = at::zeros({1}, A.options().dtype(at::kInt));
  auto infos_data = infos.data_ptr<int>();

  int m_32 = at::native::cuda_int_cast(m, "m");
  int n_32 = at::native::cuda_int_cast(n, "n");
  int lda_32 = at::native::cuda_int_cast(lda, "lda");
  auto handle = at::musa::getCurrentMUSABlasHandle();
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
  size_t lwork;
  at::musa::solver::geqrf_bufferSize<scalar_t>(m_32, n_32, &lwork);
#else
  int lwork;
  at::musa::solver::geqrf_bufferSize<scalar_t>(
      handle, m_32, n_32, A_data, lda_32, &lwork);
#endif // MUSA_VERSION

  for (decltype(batch_size) i = 0; i < batch_size; i++) {
    scalar_t* A_working_ptr = &A_data[i * A_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::musa::getCurrentMUSABlasHandle();
    // allocate workspace storage on device
    auto& allocator = *::c10::musa::MUSACachingAllocator::get();
    auto work_data =
        allocator.allocate(sizeof(scalar_t) * std::max<int>(1, lwork));
#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
    at::musa::solver::geqrf<scalar_t>(
        handle,
        m_32,
        n_32,
        A_working_ptr,
        lda_32,
        tau_working_ptr,
        static_cast<void*>(work_data.get()));
#else
    at::musa::solver::geqrf<scalar_t>(
        handle,
        m_32,
        n_32,
        A_working_ptr,
        lda_32,
        tau_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        infos_data);
#endif
  }

  // info from geqrf only reports if the i-th parameter is wrong, not about the
  // matrix singularity so we don't need to check it all the time
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);
}

/*
  The orgqr function allows reconstruction of an orthogonal (or unitary) matrix
  Q, from a sequence of elementary reflectors, such as produced by the geqrf
  function.

  Args:
  * `self` - Tensor with the directions of the elementary reflectors below the
  diagonal, it will be overwritten with the result
  * `tau` - Tensor containing the magnitudes of the elementary reflectors

  For further details, please see the cuSOLVER documentation for ORGQR and
  UNGQR.
*/
template <typename scalar_t>
static void apply_orgqr(Tensor& self, const Tensor& tau) {
  auto self_data = self.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto self_matrix_stride = at::native::matrixStride(self);
  auto batchsize =
      at::native::cuda_int_cast(at::native::batchCount(self), "batch size");
  auto m = at::native::cuda_int_cast(self.size(-2), "m");
  auto n = at::native::cuda_int_cast(self.size(-1), "n");
  auto k = at::native::cuda_int_cast(tau.size(-1), "k");
  auto tau_stride = std::max<int>(1, k);
  auto lda = std::max<int>(1, m);

  // LAPACK's requirement
  TORCH_INTERNAL_ASSERT(m >= n);
  TORCH_INTERNAL_ASSERT(n >= k);

  // cuSOLVER doesn't compute anything for this case, which is wrong
  // the result should be a matrix with 1 on the diagonal
  if (k == 0) {
    self.fill_(0);
    self.diagonal(/*offset=*/0, /*dim1=*/-2, /*dim2=*/-1).fill_(1);
    return;
  }

  // get the optimal work size and allocate workspace tensor
  auto handle = at::musa::getCurrentMUSABlasHandle();

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
  size_t lwork;
  at::musa::solver::orgqr_buffersize<scalar_t>(m, n, k, &lwork);
#else
  int lwork;
  at::musa::solver::orgqr_buffersize<scalar_t>(
      handle, m, n, k, self_data, lda, tau_data, &lwork);
#endif

  auto info = at::zeros({1}, self.options().dtype(at::kInt));
  auto info_data = info.data_ptr<int>();

  for (auto i = decltype(batchsize){0}; i < batchsize; i++) {
    scalar_t* self_working_ptr = &self_data[i * self_matrix_stride];

    // allocate workspace storage
    auto& allocator = *::c10::musa::MUSACachingAllocator::get();

    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5010)
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    at::musa::solver::orgqr<scalar_t>(
        handle,
        m,
        n,
        k,
        self_working_ptr,
        lda,
        tau_working_ptr,
        static_cast<void*>(work_data.get()));
#else
    const scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    at::musa::solver::orgqr<scalar_t>(
        handle,
        m,
        n,
        k,
        self_working_ptr,
        lda,
        tau_working_ptr,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_data);
#endif // REAL_MUSA_VERSION

    // info from orgqr only reports if the i-th parameter is wrong
    // so we don't need to check it all the time
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info.item().toInt() == 0);
  }
}

template <typename scalar_t>
void batch_ormqr(
    scalar_t* input_data,
    scalar_t* tau_data,
    scalar_t* other_data,
    int64_t batch_size,
    int64_t input_matrix_stride,
    int64_t other_matrix_stride,
    int64_t tau_stride,
    int m,
    int n,
    int k,
    int lda,
    int ldc,
    mublasSideMode_t side,
    mublasOperation_t trans,
    int lwork = 0,
    int* info_data = nullptr) {
  for (int64_t i = 0; i < batch_size; i++) {
    scalar_t* input_working_ptr = &input_data[i * input_matrix_stride];
    scalar_t* other_working_ptr = &other_data[i * other_matrix_stride];
    scalar_t* tau_working_ptr = &tau_data[i * tau_stride];
    auto handle = at::musa::getCurrentMUSABlasHandle();

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION < 5000)
    at::musa::solver::ormqr<scalar_t>(
        handle,
        side,
        trans,
        m,
        n,
        k,
        input_working_ptr,
        lda,
        tau_working_ptr,
        other_working_ptr,
        ldc);
#else
    auto& allocator = *at::musa::getMUSADeviceAllocator();
    auto work_data = allocator.allocate(sizeof(scalar_t) * lwork);
    at::musa::solver::ormqr<scalar_t>(
        handle,
        side,
        trans,
        m,
        n,
        k,
        input_working_ptr,
        lda,
        tau_working_ptr,
        other_working_ptr,
        ldc,
        static_cast<scalar_t*>(work_data.get()),
        lwork,
        info_data);
#endif // REAL_MUSA_VERSION

    if (info_data) {
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(info_data[i] == 0);
    }
  }
}

template <typename scalar_t>
void apply_ormqr(
    const Tensor& input,
    const Tensor& tau,
    const Tensor& other,
    bool left,
    bool transpose) {
  auto side = left ? MUBLAS_SIDE_LEFT : MUBLAS_SIDE_RIGHT;
  auto trans = transpose ? (input.is_complex() ? MUBLAS_OP_C : MUBLAS_OP_T)
                         : MUBLAS_OP_N;

  auto input_data = input.data_ptr<scalar_t>();
  auto tau_data = tau.data_ptr<scalar_t>();
  auto other_data = other.data_ptr<scalar_t>();

  auto input_matrix_stride = at::native::matrixStride(input);
  auto other_matrix_stride = at::native::matrixStride(other);
  auto tau_stride = tau.size(-1);
  auto batch_size = at::native::batchCount(input);

  auto int_cast = [](int64_t value, const char* varname) -> int {
    int result = static_cast<int>(value);
    TORCH_CHECK(
        static_cast<int64_t>(result) == value,
        "musa_int_cast: The value of ",
        varname,
        " (",
        (long long)value,
        ") is too large to fit into a int (",
        sizeof(int),
        " bytes)");
    return result;
  };

  auto m = int_cast(other.size(-2), "m");
  auto n = int_cast(other.size(-1), "n");
  auto k = int_cast(tau.size(-1), "k");
  auto lda = std::max<int>(1, left ? m : n);
  auto ldc = std::max<int>(1, m);

  bool IS_COMPLEX = at::isComplexType(other.scalar_type());

  auto handle = at::musa::getCurrentMUSABlasHandle();

  int lwork = 0;
  int* info_data = nullptr;
  Tensor info;

#if defined(REAL_MUSA_VERSION) && (REAL_MUSA_VERSION >= 5000)
  if (!IS_COMPLEX) {
    at::musa::solver::ormqr_buffersize<scalar_t>(
        handle,
        side,
        trans,
        m,
        n,
        k,
        input_data,
        lda,
        tau_data,
        other_data,
        ldc,
        &lwork);

    info = at::zeros({1}, input.options().dtype(at::kInt));
    info_data = info.data_ptr<int>();
  }
#endif // REAL_MUSA_VERSION

  at::musa::batch_ormqr<scalar_t>(
      input_data,
      tau_data,
      other_data,
      batch_size,
      input_matrix_stride,
      other_matrix_stride,
      tau_stride,
      m,
      n,
      k,
      lda,
      ldc,
      side,
      trans,
      lwork,
      info_data);
}

Tensor _cholesky_solve_helper_musolver(
    const Tensor& self,
    const Tensor& A,
    bool upper) {
  const int64_t batch_size = at::native::batchCount(self);
  Tensor infos = at::zeros({1}, self.options().dtype(at::kInt));
  Tensor self_working_copy = at::native::cloneBatchedColumnMajor(self);
  Tensor A_column_major_copy = at::native::cloneBatchedColumnMajor(A);

  const int64_t nrhs = self_working_copy.size(-1);

  if (batch_size > 1 && nrhs == 1) {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        self.scalar_type(), "cholesky_musa_potrs_batched", [&] {
          apply_cholesky_musolver_potrsBatched<scalar_t>(
              self_working_copy, A_column_major_copy, upper, infos);
        });
  } else {
    AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        self.scalar_type(), "cholesky_musa_potrs", [&] {
          apply_cholesky_musolver_potrs<scalar_t>(
              self_working_copy, A_column_major_copy, upper, infos);
        });
  }

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(infos.item().toInt() == 0);

  return self_working_copy;
}

// This is a type dispatching helper function for 'apply_geqrf'
void geqrf_musolver(const Tensor& input, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      input.scalar_type(), "geqrf_musa", [&] {
        apply_geqrf<scalar_t>(input, tau);
      });
}

Tensor& orgqr_helper_musolver(Tensor& result, const Tensor& tau) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      result.scalar_type(), "orgqr_cuda", [&] {
        apply_orgqr<scalar_t>(result, tau);
      });
  return result;
}

void ormqr_helper_musolver(
    const Tensor& input,
    const Tensor& tau,
    const Tensor& other,
    bool left,
    bool transpose) {
  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
      other.scalar_type(), "ormqr_musa", [&] {
        apply_ormqr<scalar_t>(input, tau, other, left, transpose);
      });
}

} // namespace at::musa
