#include <ATen/Dispatch.h>
#include <ATen/ExpandUtils.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorUtils.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/Resize.h>
#include <ATen/ops/baddbmm_native.h>
#include <ATen/ops/copy_native.h>
#include <ATen/ops/empty.h>
#include <c10/util/MaybeOwned.h>

#include "torch_musa/csrc/aten/musa/MUSABlas.h"
namespace at::native {

namespace {

c10::MaybeOwned<Tensor> inline resolve_conj_if_indicated(
    const Tensor& tensor,
    bool resolve_conj) {
  if (resolve_conj && tensor.is_conj()) {
    return c10::MaybeOwned<Tensor>::owned(tensor.resolve_conj());
  } else {
    return c10::MaybeOwned<Tensor>::borrowed(tensor);
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_mublas(
    const Tensor& tensor,
    bool& transpose_tensor,
    bool transpose_result) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = tensor.is_contiguous();
    return resolve_conj_if_indicated(
        tensor, transpose_result ? transpose_tensor : !transpose_tensor);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, !transpose_result);
  } else if (
      (tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, transpose_result);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

c10::MaybeOwned<Tensor> inline prepare_matrix_for_mublas(
    const Tensor& tensor,
    bool& transpose_tensor) {
  if (tensor.is_non_overlapping_and_dense()) { // common case
    transpose_tensor = tensor.is_contiguous();
    return resolve_conj_if_indicated(tensor, true);
  }
  IntArrayRef tensor_strides = tensor.strides();
  IntArrayRef tensor_sizes = tensor.sizes();
  if ((tensor_strides[0] == 1) &&
      (tensor_strides[1] >= std::max<int64_t>(1, tensor_sizes[0]))) {
    transpose_tensor = false;
    return resolve_conj_if_indicated(tensor, true);
  } else if (
      (tensor_strides[1] == 1) &&
      (tensor_strides[0] >= std::max<int64_t>(1, tensor_sizes[1]))) {
    transpose_tensor = true;
    return resolve_conj_if_indicated(tensor, true);
  } else {
    transpose_tensor = true;
    return c10::MaybeOwned<Tensor>::owned(
        tensor.clone(at::MemoryFormat::Contiguous));
  }
}

} // namespace

c10::MaybeOwned<Tensor> prepare_batch_matrix_for_mublas(
    const Tensor& tensor,
    bool& transpose_tensor,
    int64_t& ld_tensor,
    bool transpose_result,
    int64_t m,
    int64_t n) {
  IntArrayRef tensor_strides = tensor.strides();
  c10::MaybeOwned<Tensor> tensor_;
  int fast_dim = transpose_result ? 2 : 1;
  int leading_dim = transpose_result ? 1 : 2;

  if (tensor_strides[fast_dim] == 1 &&
      (tensor_strides[leading_dim] >= std::max<int64_t>(1, m))) {
    transpose_tensor = false;
    tensor_ = resolve_conj_if_indicated(tensor, true);
    ld_tensor = tensor_->strides()[leading_dim];
  } else if (
      (tensor_strides[leading_dim] == 1) &&
      (tensor_strides[fast_dim] >= std::max<int64_t>(1, n))) {
    transpose_tensor = true;
    tensor_ = resolve_conj_if_indicated(tensor, false);
    ld_tensor = tensor_->strides()[fast_dim];
  } else {
    transpose_tensor = !transpose_result;
    // gemm call requires leading dimension and stride parameters to be non-zero
    bool is_stride_non_zero =
        tensor.strides()[1] != 0 && tensor.strides()[2] != 0;
    if (tensor.is_contiguous() && is_stride_non_zero) {
      tensor_ = resolve_conj_if_indicated(tensor, transpose_result);
    } else {
      tensor_ = c10::MaybeOwned<Tensor>::owned(
          tensor.clone(at::MemoryFormat::Contiguous));
    }
    ld_tensor = tensor_->strides()[1];
  }

  return tensor_;
}

namespace {

const Tensor& baddbmm_out_musa_impl(
    const Tensor& result,
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  IntArrayRef batch1_sizes = batch1.sizes();

  // handle pathological cases that blas may not like
  if (result.numel() == 0) {
    return result;
  } else if (batch1_sizes[2] == 0) {
    if (beta.to<c10::complex<double>>() == 0.0) {
      return result.zero_();
    } else {
      return result.mul_(beta);
    }
  }

  bool transpose_result = false;
  c10::MaybeOwned<Tensor> result_;
  IntArrayRef result_strides = result.strides();
  IntArrayRef result_sizes = result.sizes();

  if ((result_strides[1] == 1) &&
      ((result_sizes[2] == 1) ||
       (result_strides[2] >= std::max<int64_t>(1, result_sizes[1])))) {
    result_ = resolve_conj_if_indicated(result, true);
  } else if (
      (result_strides[2] == 1) &&
      (result_sizes[1] == 1 ||
       (result_strides[1] >= std::max<int64_t>(1, result_sizes[2])))) {
    transpose_result = true;
    result_ = resolve_conj_if_indicated(result, true);
  } else {
    result_ =
        c10::MaybeOwned<Tensor>::owned(result.transpose(1, 2)
                                           .clone(at::MemoryFormat::Contiguous)
                                           .transpose(1, 2));
  }

  int leading_dim = transpose_result ? 1 : 2;

  int64_t m = result_sizes[transpose_result ? 2 : 1];
  int64_t n = result_sizes[leading_dim];
  int64_t k = (transpose_result ? batch2 : batch1).sizes()[leading_dim];

  int64_t lda, ldb, ldc;
  bool transpose_batch1, transpose_batch2;
  auto batch1_ = prepare_batch_matrix_for_mublas(
      transpose_result ? batch2 : batch1,
      transpose_batch1,
      lda,
      transpose_result,
      m,
      k);
  auto batch2_ = prepare_batch_matrix_for_mublas(
      transpose_result ? batch1 : batch2,
      transpose_batch2,
      ldb,
      transpose_result,
      k,
      n);

  ldc = result_->strides()[leading_dim];
  int64_t num_batches = result_->sizes()[0];

  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!result_->is_conj());

  AT_DISPATCH_FLOATING_AND_COMPLEX_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      self.scalar_type(),
      "baddbmm_musa",
      [&] {
        using opmath_t = at::opmath_type<scalar_t>;
        opmath_t alpha_val = alpha.to<opmath_t>();
        opmath_t beta_val = beta.to<opmath_t>();
        scalar_t* batch1_ptr = batch1_->data_ptr<scalar_t>();
        scalar_t* batch2_ptr = batch2_->data_ptr<scalar_t>();
        scalar_t* result_ptr = result_->data_ptr<scalar_t>();
        at::musa::blas::bgemm<scalar_t>(
            transpose_batch1 ? batch1_->is_conj() ? 'c' : 't' : 'n',
            transpose_batch2 ? batch2_->is_conj() ? 'c' : 't' : 'n',
            m,
            n,
            k,
            alpha_val,
            batch1_ptr,
            lda,
            batch1_->strides()[0],
            batch2_ptr,
            ldb,
            batch2_->strides()[0],
            beta_val,
            result_ptr,
            ldc,
            result_->strides()[0],
            num_batches);
      });
  if (!result.is_same(*result_)) {
    result.copy_(*result_);
  }
  return result;
}

} // anonymous namespace

TORCH_IMPL_FUNC(baddbmm_out_cuda)
(const Tensor& self,
 const Tensor& batch1,
 const Tensor& batch2,
 const Scalar& beta,
 const Scalar& alpha,
 const Tensor& result) {
  {
    at::NoNamesGuard guard;
    baddbmm_out_musa_impl(result, self, batch1, batch2, beta, alpha);
  }
}

} // namespace at::native
