#include <torch_musa/csrc/aten/mudnn/Unary.h>

#include <optional>

#include <torch_musa/csrc/aten/mudnn/Exception.h>
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnUnaryMode_t GetUnaryModeFromEnum(Unary::Mode mode) {
#define CASE(MODE1, MODE2)   \
  case Unary::Mode::MODE1: { \
    return MODE2;            \
  }

  switch (mode) {
    MUDNN_FORALL_UNARY_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

template <typename T>
void SetUnaryImpl(
    Unary& op,
    Unary::Mode mode,
    std::optional<T> alpha,
    std::optional<T> beta) {
  const auto m = GetUnaryModeFromEnum(mode);
  MudnnParam a, b;
  if (alpha) {
    a.Set(alpha.value());
  }
  if (beta) {
    b.Set(beta.value());
  }
  CHECK_MUDNN_STATUS(
      mudnnSetUnaryDescriptor(
          op.Desc(), m, a.Type(), a.Ptr(), b.Type(), b.Ptr()),
      "mudnnSetUnaryDescriptor");
}

#else

template <typename T>
void SetUnaryImpl(
    Unary& op,
    Unary::Mode mode,
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

mudnnStatus_t Unary::Run(muHandle& h, muTensor& out, const muTensor& in) const {
  out.Build();
  in.Build();
  return mudnnUnary(
      h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetUnary(Unary& op, Unary::Mode mode) {
  SetUnaryImpl<double>(op, mode, std::nullopt, std::nullopt);
}

void SetUnary(Unary& op, Unary::Mode mode, int64_t alpha) {
  SetUnaryImpl<int64_t>(op, mode, alpha, std::nullopt);
}

void SetUnary(Unary& op, Unary::Mode mode, double alpha) {
  SetUnaryImpl<double>(op, mode, alpha, std::nullopt);
}

void SetUnary(Unary& op, Unary::Mode mode, int64_t alpha, int64_t beta) {
  SetUnaryImpl<int64_t>(op, mode, alpha, beta);
}

void SetUnary(Unary& op, Unary::Mode mode, double alpha, double beta) {
  SetUnaryImpl<double>(op, mode, alpha, beta);
}

} // namespace at::musa
