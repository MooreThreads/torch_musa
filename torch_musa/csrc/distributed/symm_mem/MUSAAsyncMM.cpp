#define TORCH_ASSERT_ONLY_METHOD_OPERATORS

#include "torch_musa/csrc/distributed/symm_mem/MUSAAsyncMM.hpp"

#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/core/MUSAGuard.h"

#include <ATen/ops/mm.h>

namespace c10d::musa::detail {

namespace {

void check_async_input_mm_inputs(
    const at::Tensor& a,
    const at::Tensor& b,
    const at::Tensor& a_chunk_signals,
    int64_t a_chunk_pivot,
    const at::Tensor& out) {
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2 && out.dim() == 2,
      "async_input_mm: `a`, `b` and `out` must be matrices");
  TORCH_CHECK(
      a.is_contiguous() && out.is_contiguous(),
      "async_input_mm: `a` and `out` must be in row-major layout");

  if (!b.is_contiguous()) {
    TORCH_CHECK(b.stride(1) == b.size(0));
    TORCH_CHECK(b.stride(0) == 1);
  }

  TORCH_CHECK_EQ(a.scalar_type(), at::kBFloat16);
  TORCH_CHECK_EQ(b.scalar_type(), at::kBFloat16);
  TORCH_CHECK_EQ(out.scalar_type(), at::kBFloat16);

  const int64_t M = a.sizes()[0];
  const int64_t N = b.sizes()[1];
  const int64_t K = a.sizes()[1];
  TORCH_CHECK_EQ(b.sizes()[0], K);
  TORCH_CHECK_EQ(out.sizes()[0], M);
  TORCH_CHECK_EQ(out.sizes()[1], N);

  TORCH_CHECK(
      a_chunk_signals.dim() == 1,
      "async_input_mm: `a_chunk_signals` must be a 1D tensor.");
  TORCH_CHECK_EQ(a_chunk_signals.scalar_type(), c10::ScalarType::UInt32);
  TORCH_CHECK(
      a_chunk_signals.device() == a.device(),
      "async_input_mm: `a_chunk_signals` must be on the same device as `a`.");

  const int64_t num_chunks_M = a_chunk_signals.numel();
  TORCH_CHECK(
      num_chunks_M > 0,
      "async_input_mm: `a_chunk_signals` must contain at least one element.");
  TORCH_CHECK(
      M % num_chunks_M == 0,
      "async_input_mm: `a.shape(0)` must be an integer multiple of `a_chunk_signals.numel()`");
  TORCH_CHECK(
      a_chunk_pivot >= 0 && a_chunk_pivot < num_chunks_M,
      "async_input_mm: `a_chunk_pivot` must be in the range [0, `a_chunk_signals.numel()`).");

  constexpr int64_t kTileSizeM = 128;
  const int64_t chunk_size_M = M / num_chunks_M;
  TORCH_CHECK(chunk_size_M % kTileSizeM == 0);
}

} // namespace

at::Tensor async_input_mm_out(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t a_chunk_pivot,
    at::Tensor out) {
  c10::musa::MUSAGuard guard(a.device());
  check_async_input_mm_inputs(a, b, a_chunk_signals, a_chunk_pivot, out);

  at::mm_out(out, a, b);
  return out;
}

at::Tensor async_input_mm(
    at::Tensor a,
    at::Tensor b,
    at::Tensor a_chunk_signals,
    int64_t a_chunk_pivot) {
  TORCH_CHECK(
      a.dim() == 2 && b.dim() == 2,
      "async_input_mm: `a`, `b` and `out` must all be a matrix");

  const int64_t M = a.sizes()[0];
  const int64_t N = b.sizes()[1];
  auto out = a.new_empty({M, N});
  return async_input_mm_out(a, b, a_chunk_signals, a_chunk_pivot, out);
}

} // namespace c10d::musa::detail
