/*
  Provides the implementations of MUSA BLAS function templates.
 */
#include <c10/macros/Export.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/aten/musa/MUSABlas.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
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
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
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
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
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
  // See Note [Writing Nondeterministic Operations]
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_MUSABLAS_CHECK(mublasZgemmStridedBatched(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      reinterpret_cast<const muDoubleComplex*>(&alpha),
      reinterpret_cast<const muDoubleComplex*>(a),
      lda,
      stridea,
      reinterpret_cast<const muDoubleComplex*>(b),
      ldb,
      strideb,
      reinterpret_cast<const muDoubleComplex*>(&beta),
      reinterpret_cast<muDoubleComplex*>(c),
      ldc,
      stridec,
      num_batches));
}

template <>
void bgemm<c10::complex<float>>(MUSABLAS_BGEMM_ARGTYPES(c10::complex<float>)) {
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  BGEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_MUSABLAS_CHECK(mublasCgemmStridedBatched(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      reinterpret_cast<const muComplex*>(&alpha),
      reinterpret_cast<const muComplex*>(a),
      lda,
      stridea,
      reinterpret_cast<const muComplex*>(b),
      ldb,
      strideb,
      reinterpret_cast<const muComplex*>(&beta),
      reinterpret_cast<muComplex*>(c),
      ldc,
      stridec,
      num_batches));
}

template <>
void bgemm<at::Half>(MUSABLAS_BGEMM_ARGTYPES(at::Half)) {
  // See Note [Writing Nondeterministic Operations]
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
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
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
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
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
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
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<double>);
  TORCH_MUSABLAS_CHECK(mublasZgemm(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      reinterpret_cast<const muDoubleComplex*>(&alpha),
      reinterpret_cast<const muDoubleComplex*>(a),
      lda,
      reinterpret_cast<const muDoubleComplex*>(b),
      ldb,
      reinterpret_cast<const muDoubleComplex*>(&beta),
      reinterpret_cast<muDoubleComplex*>(c),
      ldc));
}

template <>
void gemm<c10::complex<float>>(MUSABLAS_GEMM_ARGTYPES(c10::complex<float>)) {
  at::musa::GlobalContext().alertMuBLASConfigNotDeterministic();
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  mublasOperation_t opa = _mublasOpFromChar(transa);
  mublasOperation_t opb = _mublasOpFromChar(transb);
  _mublasAdjustLdLevel3(transa, transb, m, n, k, &lda, &ldb, &ldc);
  GEMM_CHECK_ARGVALUES(c10::complex<float>);
  TORCH_MUSABLAS_CHECK(mublasCgemm(
      handle,
      opa,
      opb,
      m,
      n,
      k,
      reinterpret_cast<const muComplex*>(&alpha),
      reinterpret_cast<const muComplex*>(a),
      lda,
      reinterpret_cast<const muComplex*>(b),
      ldb,
      reinterpret_cast<const muComplex*>(&beta),
      reinterpret_cast<muComplex*>(c),
      ldc));
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

template <>
void vdot<c10::complex<float>>(MUSABLAS_VDOT_ARGTYPES(c10::complex<float>)) {
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  TORCH_MUSABLAS_CHECK(mublasCdotc(
      handle,
      n,
      reinterpret_cast<const muComplex*>(x),
      incx,
      reinterpret_cast<const muComplex*>(y),
      incy,
      reinterpret_cast<muComplex*>(result)));
}

template <>
void vdot<c10::complex<double>>(MUSABLAS_VDOT_ARGTYPES(c10::complex<double>)) {
  mublasHandle_t handle = at::musa::getCurrentMUSABlasHandle();
  TORCH_MUSABLAS_CHECK(mublasZdotc(
      handle,
      n,
      reinterpret_cast<const muDoubleComplex*>(x),
      incx,
      reinterpret_cast<const muDoubleComplex*>(y),
      incy,
      reinterpret_cast<muDoubleComplex*>(result)));
}

template <>
void trsm<float>(MUSABLAS_TRSM_ARGTYPES(float)) {
  TORCH_MUSABLAS_CHECK(mublasStrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<double>(MUSABLAS_TRSM_ARGTYPES(double)) {
  TORCH_MUSABLAS_CHECK(mublasDtrsm(
      handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb));
}

template <>
void trsm<c10::complex<float>>(MUSABLAS_TRSM_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSABLAS_CHECK(mublasCtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const muComplex*>(alpha),
      reinterpret_cast<const muComplex*>(A),
      lda,
      reinterpret_cast<muComplex*>(B),
      ldb));
}

template <>
void trsm<c10::complex<double>>(MUSABLAS_TRSM_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSABLAS_CHECK(mublasZtrsm(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      reinterpret_cast<const muDoubleComplex*>(A),
      lda,
      reinterpret_cast<muDoubleComplex*>(B),
      ldb));
}

template <>
// NOLINTNEXTLINE(*array*)
void trsmBatched<float>(MUSABLAS_TRSM_BATCHED_ARGTYPES(float)) {
  TORCH_MUSABLAS_CHECK(mublasStrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
// NOLINTNEXTLINE(*array*)
void trsmBatched<double>(MUSABLAS_TRSM_BATCHED_ARGTYPES(double)) {
  TORCH_MUSABLAS_CHECK(mublasDtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      alpha,
      A,
      lda,
      B,
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<float>>(
    // NOLINTNEXTLINE(*array*)
    MUSABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<float>)) {
  TORCH_MUSABLAS_CHECK(mublasCtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const muComplex*>(alpha),
      reinterpret_cast<muComplex**>(A),
      lda,
      reinterpret_cast<muComplex**>(B),
      ldb,
      batchCount));
}

template <>
void trsmBatched<c10::complex<double>>(
    // NOLINTNEXTLINE(*array*)
    MUSABLAS_TRSM_BATCHED_ARGTYPES(c10::complex<double>)) {
  TORCH_MUSABLAS_CHECK(mublasZtrsmBatched(
      handle,
      side,
      uplo,
      trans,
      diag,
      m,
      n,
      reinterpret_cast<const muDoubleComplex*>(alpha),
      reinterpret_cast<muDoubleComplex**>(A),
      lda,
      reinterpret_cast<muDoubleComplex**>(B),
      ldb,
      batchCount));
}

template <typename T, mublasStatus_t (*destructor)(T*)>
struct MuBlasLtDeleter {
  void operator()(T* x) {
    if (x != nullptr) {
      TORCH_MUSABLAS_CHECK(destructor(x));
    }
  }
};

template <typename T, mublasStatus_t (*destructor)(T*)>
class MuBlasLtDescriptor {
 public:
  T* descriptor() const {
    return descriptor_.get();
  }
  T* descriptor() {
    return descriptor_.get();
  }

 protected:
  std::unique_ptr<T, MuBlasLtDeleter<T, destructor>> descriptor_;
};

class MuBlasLtMatmulDescriptor : public MuBlasLtDescriptor<
                                     mublasLtMatmulDescOpaque_t,
                                     &mublasLtMatmulDescDestroy> {
 public:
  MuBlasLtMatmulDescriptor(
      mublasComputeType_t compute_type,
      musaDataType_t scale_type) {
    mublasLtMatmulDesc_t raw_descriptor = nullptr;
    TORCH_MUSABLAS_CHECK(
        mublasLtMatmulDescCreate(&raw_descriptor, compute_type, scale_type));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(mublasLtMatmulDescAttributes_t attr, const T value) {
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    TORCH_MUSABLAS_CHECK(::mublasLtMatmulDescSetAttribute(
        descriptor(), attr, &value, sizeof(value)));
  }
};

class MuBlasLtMatmulPreference : public MuBlasLtDescriptor<
                                     mublasLtMatmulPreferenceOpaque_t,
                                     &mublasLtMatmulPreferenceDestroy> {
 public:
  MuBlasLtMatmulPreference() {
    mublasLtMatmulPreference_t raw_descriptor = nullptr;
    TORCH_MUSABLAS_CHECK(mublasLtMatmulPreferenceCreate(&raw_descriptor));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(
      mublasLtMatmulPreferenceAttributes_t attr,
      const T value) {
    TORCH_MUSABLAS_CHECK(::mublasLtMatmulPreferenceSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

class MuBlasLtMatrixLayout : public MuBlasLtDescriptor<
                                 mublasLtMatrixLayoutOpaque_t,
                                 &mublasLtMatrixLayoutDestroy> {
 public:
  MuBlasLtMatrixLayout(
      musaDataType_t type,
      uint64_t rows,
      uint64_t cols,
      int64_t ld,
      bool t = false) {
    mublasLtMatrixLayout_t raw_descriptor = nullptr;
    TORCH_MUSABLAS_CHECK(mublasLtMatrixLayoutCreate(
        &raw_descriptor, type, t ? cols : rows, t ? rows : cols, ld));
    descriptor_.reset(raw_descriptor);
  }
  template <typename T>
  inline void setAttribute(
      mublasLtMatrixLayoutAttribute_t attr,
      const T value) {
    TORCH_MUSABLAS_CHECK(::mublasLtMatrixLayoutSetAttribute(
        descriptor(), attr, &value, sizeof(T)));
  }
};

void int8_gemm(
    bool transpose_mat1,
    bool transpose_mat2,
    int64_t m,
    int64_t n,
    int64_t k,
    const int8_t* mat1_ptr,
    int64_t mat1_ld,
    const int8_t* mat2_ptr,
    int64_t mat2_ld,
    int32_t* result_ptr,
    int64_t result_ld) {
  mublasComputeType_t computeType = MUBLAS_COMPUTE_32I;
  musaDataType_t scaleType = MUSA_R_32I;

  musaDataType_t abType = MUSA_R_8I;
  musaDataType_t cType = MUSA_R_32I;

  mublasLtMatmulHeuristicResult_t heuristicResult = {};

  MuBlasLtMatmulDescriptor computeDesc(computeType, scaleType);
  mublasOperation_t transa = transpose_mat1 ? MUBLAS_OP_T : MUBLAS_OP_N;
  computeDesc.setAttribute(MUBLASLT_MATMUL_DESC_TRANSA, transa);
  mublasOperation_t transb = transpose_mat2 ? MUBLAS_OP_T : MUBLAS_OP_N;
  computeDesc.setAttribute(MUBLASLT_MATMUL_DESC_TRANSB, transb);
  auto stream = at::musa::getCurrentMUSAStream();

  MuBlasLtMatrixLayout Adesc(abType, m, k, mat1_ld, transpose_mat1);
  MuBlasLtMatrixLayout Bdesc(abType, k, n, mat2_ld, transpose_mat2);
  MuBlasLtMatrixLayout Cdesc(cType, m, n, result_ld);

  // mublas team: alpha and beta need to be the same dtype as of scaleType
  at::opmath_type<int32_t> alpha_val = 1;
  int32_t beta_val = 0;
  mublasLtHandle_t ltHandle = at::musa::getCurrentMUSABlasLtHandle();

  size_t workspaceSize = 1024 * 1024 * 4;
  auto& allocator = *::c10::musa::MUSACachingAllocator::get();
  void* workspace = allocator.allocate(workspaceSize).get();

  MuBlasLtMatmulPreference preference;
  preference.setAttribute(
      MUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, workspaceSize);
  int returnedResult = 0;

  TORCH_MUSABLAS_CHECK(mublasLtMatmulAlgoGetHeuristic(
      ltHandle,
      computeDesc.descriptor(),
      Adesc.descriptor(),
      Bdesc.descriptor(),
      Cdesc.descriptor(),
      Cdesc.descriptor(),
      preference.descriptor(),
      1,
      &heuristicResult,
      &returnedResult));
  if (returnedResult == 0) {
    TORCH_MUSABLAS_CHECK(MUBLAS_STATUS_NOT_IMPLEMENTED);
  }

  mublasStatus_t mublasStatus = mublasLtMatmul(
      ltHandle,
      computeDesc.descriptor(),
      &alpha_val,
      mat1_ptr,
      Adesc.descriptor(),
      mat2_ptr,
      Bdesc.descriptor(),
      &beta_val,
      result_ptr,
      Cdesc.descriptor(),
      result_ptr,
      Cdesc.descriptor(),
      &heuristicResult.algo, // Heuristics don't seem to work for int8
      workspace, // Non-zero workspace doesn't seem to work.
      workspaceSize,
      stream);
  TORCH_CHECK(
      mublasStatus == MUBLAS_STATUS_SUCCESS,
      "MUSA error: ",
      at::musa::blas::_mublasGetErrorEnum(mublasStatus),
      " when calling mublasLtMatmul with transpose_mat1 ",
      transpose_mat1,
      " transpose_mat2 ",
      transpose_mat2,
      " m ",
      m,
      " n ",
      n,
      " k ",
      k,
      " mat1_ld ",
      mat1_ld,
      " mat2_ld ",
      mat2_ld,
      " result_ld ",
      result_ld,
      " abType ",
      abType,
      " cType ",
      cType,
      " computeType ",
      computeType,
      " scaleType ",
      scaleType);
}

} // namespace blas
} // namespace musa
} // namespace at
