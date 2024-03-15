/*
  Provides the implementations of MUSA BLAS function templates.
 */
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/aten/musa/MUSABlas.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"

#define MUSABLAS_POSINT_CHECK(FD, X)         \
  TORCH_CHECK(                               \
      (X > 0 && X <= INT_MAX),               \
      "at::musa::blas::" #FD " argument " #X \
      " must be positive and less than ",    \
      INT_MAX,                               \
      " but got ",                           \
      X)

#define MUSABLAS_NONNEGINT_CHECK(FD, X)       \
  TORCH_CHECK(                                \
      (X >= 0 && X <= INT_MAX),               \
      "at::musa::blas::" #FD " argument " #X  \
      " must be non-negative and less than ", \
      INT_MAX,                                \
      " but got ",                            \
      X)

namespace {

static mublasOperation_t _mublasOpFromChar(char op) {
  switch (op) {
    case 'n':
    case 'N':
      return MUBLAS_OP_N;
    case 't':
    case 'T':
      return MUBLAS_OP_T;
    case 'c':
    case 'C':
      return MUBLAS_OP_C;
  }
  AT_ERROR(
      "_mublasOpFromChar input should be 't', 'n' or 'c' but got `", op, "`");
}

static void _mublasAdjustLdLevel3(
    char transa,
    char transb,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t* lda,
    int64_t* ldb,
    int64_t* ldc) {
  bool transa_ = ((transa != 'n') && (transa != 'N'));
  bool transb_ = ((transb != 'n') && (transb != 'N'));

  // Note: leading dimensions generally are checked that they are > 0
  // and at least as big the result requires (even if the value won't
  // be used).
  if (n <= 1)
    *ldc = std::max<int64_t>(m, 1);

  if (transa_) {
    if (m <= 1)
      *lda = std::max<int64_t>(k, 1);
  } else {
    if (k <= 1)
      *lda = std::max<int64_t>(m, 1);
  }

  if (transb_) {
    if (k <= 1)
      *ldb = std::max<int64_t>(n, 1);
  } else {
    if (n <= 1)
      *ldb = std::max<int64_t>(k, 1);
  }
}
} // anonymous namespace

namespace at {
namespace musa {
namespace blas {

#define GEMM_CHECK_ARGVALUES(Dtype)           \
  do {                                        \
    MUSABLAS_NONNEGINT_CHECK(gemm<Dtype>, m); \
    MUSABLAS_NONNEGINT_CHECK(gemm<Dtype>, n); \
    MUSABLAS_NONNEGINT_CHECK(gemm<Dtype>, k); \
    MUSABLAS_POSINT_CHECK(gemm<Dtype>, lda);  \
    MUSABLAS_POSINT_CHECK(gemm<Dtype>, ldb);  \
    MUSABLAS_POSINT_CHECK(gemm<Dtype>, ldc);  \
  } while (0)

#define BGEMM_CHECK_ARGVALUES(Dtype)                     \
  do {                                                   \
    MUSABLAS_NONNEGINT_CHECK(bgemm<Dtype>, m);           \
    MUSABLAS_NONNEGINT_CHECK(bgemm<Dtype>, n);           \
    MUSABLAS_NONNEGINT_CHECK(bgemm<Dtype>, k);           \
    MUSABLAS_POSINT_CHECK(bgemm<Dtype>, lda);            \
    MUSABLAS_POSINT_CHECK(bgemm<Dtype>, ldb);            \
    MUSABLAS_POSINT_CHECK(bgemm<Dtype>, ldc);            \
    MUSABLAS_NONNEGINT_CHECK(bgemm<Dtype>, num_batches); \
  } while (0)

template <>
void bgemm<double>(MUSABLAS_BGEMM_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(double);
  TORCH_MUSABLAS_CHECK(mublasDgemmStridedBatched(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &alpha,
      a,
      lda,
      stridea,
      b,
      ldb,
      strideb,
      &beta,
      c,
      ldc,
      stridec,
      num_batches));
}

template <>
void bgemm<float>(MUSABLAS_BGEMM_ARGTYPES(float)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(float);
  TORCH_MUSABLAS_CHECK(mublasSgemmStridedBatched(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      &alpha,
      a,
      lda,
      stridea,
      b,
      ldb,
      strideb,
      &beta,
      c,
      ldc,
      stridec,
      num_batches));
}

template <>
void bgemm<c10::complex<double>>(
    MUSABLAS_BGEMM_ARGTYPES(c10::complex<double>)) {
  C10_THROW_ERROR(
      NotImplementedError,
      "bgemm for complex<double> in MUBLAS is not supported now!");
}

template <>
void bgemm<c10::complex<float>>(MUSABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  C10_THROW_ERROR(
      NotImplementedError,
      "bgemm for complex<float> in MUBLAS is not supported now!");
}

template <>
void bgemm<at::Half>(MUSABLAS_BGEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(at::Half);
  float falpha = alpha;
  float fbeta = beta;
  musaDeviceProp* prop = at::musa::getCurrentDeviceProperties();
  for (const auto i : c10::irange(num_batches)) {
    at::musa::blas::gemm<at::Half>(
        transa,
        transb,
        m,
        n,
        k,
        alpha,
        (a + i * stridea),
        lda,
        (b + i * strideb),
        ldb,
        beta,
        (c + i * stridec),
        ldc);
  }
}

template <>
void bgemm<at::BFloat16>(MUSABLAS_BGEMM_ARGTYPES(at::BFloat16)) {
  C10_THROW_ERROR(
      NotImplementedError,
      "bgemm for BFloat16 in MUBLAS is not supported now!");
}

template <>
void gemm<double>(MUSABLAS_GEMM_ARGTYPES(double)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(double);
  TORCH_MUSABLAS_CHECK(mublasDgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<float>(MUSABLAS_GEMM_ARGTYPES(float)) {
  // See Note [Writing Nondeterministic Operations]
  globalContext().alertCuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(float);
  TORCH_MUSABLAS_CHECK(mublasSgemm(
      handle, opa, opb, m, n, k, &alpha, a, lda, b, ldb, &beta, c, ldc));
}

template <>
void gemm<c10::complex<double>>(MUSABLAS_GEMM_ARGTYPES(c10::complex<double>)) {
  C10_THROW_ERROR(
      NotImplementedError,
      "gemm for complex<double> in MUBLAS is not supported now!");
}

template <>
void gemm<c10::complex<float>>(MUSABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  C10_THROW_ERROR(
      NotImplementedError,
      "gemm for complex<float> in MUBLAS is not supported now!");
}

template <>
void gemm<at::Half>(MUSABLAS_GEMM_ARGTYPES(at::Half)) {
  C10_THROW_ERROR(
      NotImplementedError, "gemm for Half in MUBLAS is not supported now!");
}

template <>
void gemm<at::BFloat16>(MUSABLAS_GEMM_ARGTYPES(at::BFloat16)) {
  C10_THROW_ERROR(
      NotImplementedError, "gemm for BFloat16 in MUBLAS is not supported now!");
}

} // namespace blas
} // namespace musa
} // namespace at
