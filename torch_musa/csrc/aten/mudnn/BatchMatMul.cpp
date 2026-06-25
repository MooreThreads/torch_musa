#include <torch_musa/csrc/aten/mudnn/BatchMatMul.h>

#include <optional>

#include <torch_musa/csrc/aten/mudnn/Exception.h>

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetBatchMatMulImpl(
    BatchMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    std::optional<double> alpha,
    std::optional<double> beta,
    std::optional<double> gamma) {
  const auto a = alpha.value_or(1.0f);
  const auto b = beta.value_or(0.0f);
  const auto g = gamma.value_or(1.0f);
  CHECK_MUDNN_STATUS(
      mudnnSetBatchMatMulDescriptorEx(
          op.Desc(),
          mode,
          transa,
          transb,
          false,
          MUDNN_MATMULLT_EPILOGUE_DEFAULT,
          0,
          a,
          b,
          g),
      "mudnnSetBatchMatMulDescriptorEx");
}

#else

template <typename MM>
void SetBatchMatMulImpl(
    MM& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    std::optional<double> alpha,
    std::optional<double> beta,
    std::optional<double> gamma) {
  CHECK_MUDNN_STATUS(op.SetComputeMode(mode), "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetTranspose(transa, transb), "SetTranspose");
  if (alpha) {
    CHECK_MUDNN_STATUS(op.SetAlpha(alpha.value()), "SetAlpha");
  }
  if (beta) {
    CHECK_MUDNN_STATUS(op.SetBeta(beta.value()), "SetBeta");
  }
  if (gamma) {
    CHECK_MUDNN_STATUS(op.SetGamma(gamma.value()), "SetGamma");
  }
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t BatchMatMul::Run(
    muHandle& h,
    muTensor& c,
    const muTensor& a,
    const muTensor& b,
    const MemoryMaintainer& maintainer) const {
  c.Build();
  a.Build();
  b.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnBatchMatmulGetWorkspaceSize(
          h, Desc(), a.Desc(), b.Desc(), c.Desc(), c.Desc(), &ws_size),
      "mudnnBatchMatmulGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnBatchMatMulBias(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(a),
      MUDNN_UNPACK_TENSOR(b),
      MUDNN_UNPACK_TENSOR(c),
      MUDNN_UNPACK_TENSOR(c),
      nullptr,
      nullptr,
      mem.get(),
      ws_size);
}

mudnnStatus_t BatchMatMul::RunWithBiasAdd(
    muHandle& h,
    muTensor& d,
    const muTensor& a,
    const muTensor& b,
    const muTensor& c,
    const muTensor& bias,
    const MemoryMaintainer& maintainer) const {
  d.Build();
  c.Build();
  a.Build();
  b.Build();
  bias.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnBatchMatmulGetWorkspaceSize(
          h, Desc(), a.Desc(), b.Desc(), c.Desc(), d.Desc(), &ws_size),
      "mudnnBatchMatmulGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnBatchMatMulBias(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(a),
      MUDNN_UNPACK_TENSOR(b),
      MUDNN_UNPACK_TENSOR(c),
      MUDNN_UNPACK_TENSOR(d),
      MUDNN_UNPACK_TENSOR(bias),
      mem.get(),
      ws_size);
}

mudnnStatus_t BatchMatMul::RunWithBiasAdd(
    muHandle& h,
    muTensor& c,
    const muTensor& a,
    const muTensor& b,
    const muTensor& bias,
    const MemoryMaintainer& maintainer) const {
  c.Build();
  a.Build();
  b.Build();
  bias.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnBatchMatmulGetWorkspaceSize(
          h, Desc(), a.Desc(), b.Desc(), c.Desc(), c.Desc(), &ws_size),
      "mudnnBatchMatmulGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnBatchMatMulBias(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(a),
      MUDNN_UNPACK_TENSOR(b),
      MUDNN_UNPACK_TENSOR(c),
      MUDNN_UNPACK_TENSOR(c),
      MUDNN_UNPACK_TENSOR(bias),
      mem.get(),
      ws_size);
}

mudnnStatus_t BatchMatMul::RunLt(
    muHandle& h,
    muTensor& d,
    const muTensor& a,
    const muTensor& b,
    const muTensor& c,
    const muTensor& bias,
    const MatMulLtParam& param,
    const MemoryMaintainer& maintainer) const {
  d.Build();
  c.Build();
  a.Build();
  b.Build();
  bias.Build();
  param.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnBatchMatmulGetWorkspaceSize(
          h, Desc(), a.Desc(), b.Desc(), c.Desc(), d.Desc(), &ws_size),
      "mudnnBatchMatmulGetWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnBatchMatMulBiasTensorScaleEpilogue(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(a),
      MUDNN_UNPACK_TENSOR(b),
      MUDNN_UNPACK_TENSOR(c),
      MUDNN_UNPACK_TENSOR(d),
      MUDNN_UNPACK_TENSOR(bias),
      MUDNN_UNPACK_TENSOR(param.scale_a),
      MUDNN_UNPACK_TENSOR(param.scale_b),
      MUDNN_UNPACK_TENSOR(param.scale_c),
      MUDNN_UNPACK_TENSOR(param.scale_d),
      0,
      false,
      MUDNN_UNPACK_TENSOR(param.amax_d),
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      mem.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetBatchMatMul(
    BatchMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb) {
  SetBatchMatMulImpl(
      op, mode, transa, transb, std::nullopt, std::nullopt, std::nullopt);
}

void SetBatchMatMul(
    BatchMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  SetBatchMatMulImpl(op, mode, transa, transb, alpha, beta, gamma);
}

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetMatMul(MatMul& op, ComputeMode mode, bool transa, bool transb) {
  SetBatchMatMul(op, mode, transa, transb);
}

void SetMatMul(
    MatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  SetBatchMatMul(op, mode, transa, transb, alpha, beta, gamma);
}

#else

void SetMatMul(MatMul& op, ComputeMode mode, bool transa, bool transb) {
  SetBatchMatMulImpl(
      op, mode, transa, transb, std::nullopt, std::nullopt, std::nullopt);
}

void SetMatMul(
    MatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  SetBatchMatMulImpl(op, mode, transa, transb, alpha, beta, gamma);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
