#include <torch_musa/csrc/aten/mudnn/Binary.h>

#include <optional>

#include <torch_musa/csrc/aten/mudnn/Exception.h>
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnBinaryMode_t GetBinaryModeFromEnum(Binary::Mode mode) {
#define CASE(MODE1, MODE2)    \
  case Binary::Mode::MODE1: { \
    return MODE2;             \
  }

  switch (mode) {
    MUDNN_FORALL_BINARY_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

template <typename T>
void SetBinaryImpl(
    Binary& op,
    Binary::Mode mode,
    std::optional<T> alpha,
    std::optional<T> beta) {
  const auto m = GetBinaryModeFromEnum(mode);
  MudnnParam a, b;
  if (alpha) {
    a.Set(alpha.value());
  }
  if (beta) {
    b.Set(beta.value());
  }
  CHECK_MUDNN_STATUS(
      mudnnSetBinaryDescriptor(
          op.Desc(), m, a.Type(), a.Ptr(), b.Type(), b.Ptr()),
      "mudnnSetBinaryDescriptor");
}

#else

template <typename T>
void SetBinaryImpl(
    Binary& op,
    Binary::Mode mode,
    std::optional<T> alpha,
    std::optional<T> beta) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  if (alpha) {
    CHECK_MUDNN_STATUS(op.SetAlpha(alpha.value()), "SetAlpha");
  }
  if (beta) {
    CHECK_MUDNN_STATUS(op.SetBeta(beta.value()), "SetBeta");
  }
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Binary::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& left,
    const muTensor& right) const {
  out.Build();
  left.Build();
  right.Build();
  return mudnnBinary(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(left),
      MUDNN_UNPACK_TENSOR(right));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetBinary(Binary& op, Binary::Mode mode) {
  SetBinaryImpl<double>(op, mode, std::nullopt, std::nullopt);
}

void SetBinary(Binary& op, Binary::Mode mode, int64_t alpha) {
  SetBinaryImpl<int64_t>(op, mode, alpha, std::nullopt);
}

void SetBinary(Binary& op, Binary::Mode mode, double alpha) {
  SetBinaryImpl<double>(op, mode, alpha, std::nullopt);
}

void SetBinary(Binary& op, Binary::Mode mode, int64_t alpha, int64_t beta) {
  SetBinaryImpl<int64_t>(op, mode, alpha, beta);
}

void SetBinary(Binary& op, Binary::Mode mode, double alpha, double beta) {
  SetBinaryImpl<double>(op, mode, alpha, beta);
}

} // namespace at::musa
