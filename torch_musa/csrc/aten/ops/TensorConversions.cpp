#include <ATen/ATen.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/count_nonzero.h>
#include <ATen/ops/count_nonzero_native.h>
#endif

#include <ATen/native/TensorConversions.h>
#include "torch_musa/csrc/aten/utils/StateGuard.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {
at::Tensor SparseDimDenseToSparse(const at::Tensor& self, int64_t sparse_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse(self, sparse_dim);
}

// Redefine _to_sparse MUSA dispatch becauseof Nonzero differs from CUDA with
// contiguous output
at::Tensor DenseToSparse(
    const at::Tensor& self,
    ::std::optional<at::Layout> layout,
    at::OptionalIntArrayRef blocksize,
    ::std::optional<int64_t> dense_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse(self, layout, blocksize, dense_dim);
}

// Redefine _to_sparse_csr MUSA dispatch becauseof Nonzero differs from CUDA
// with contiguous output
at::Tensor DenseToSparseCsr(
    const Tensor& self,
    ::std::optional<int64_t> dense_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse_csr(self, dense_dim);
}

// Redefine _to_sparse_csc MUSA dispatch becauseof Nonzero differs from CUDA
// with contiguous output
at::Tensor DenseToSparseCsc(
    const Tensor& self,
    ::std::optional<int64_t> dense_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse_csc(self, dense_dim);
}

// Redefine _to_sparse_bsr MUSA dispatch becauseof Nonzero differs from CUDA
// with contiguous output
at::Tensor DenseToSparseBsr(
    const Tensor& self,
    at::IntArrayRef blocksize,
    ::std::optional<int64_t> dense_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse_bsr(self, blocksize, dense_dim);
}

// Redefine _to_sparse_bsc MUSA dispatch becauseof Nonzero differs from CUDA
// with contiguous output
at::Tensor DenseToSparseBsc(
    const Tensor& self,
    at::IntArrayRef blocksize,
    ::std::optional<int64_t> dense_dim) {
  TORCH_CHECK(
      self.device().type() == kMUSA,
      "Device of input tensor of Nonzero must be MUSA, but now is ",
      self.device());
  MAKE_STATE_GUARD(NNZ_NONE_CONTIGUOUS, true);
  return at::native::dense_to_sparse_bsc(self, blocksize, dense_dim);
}
} // namespace at::musa
