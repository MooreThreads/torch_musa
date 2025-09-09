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

} // namespace at::musa