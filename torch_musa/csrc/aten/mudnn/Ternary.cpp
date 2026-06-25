#include <torch_musa/csrc/aten/mudnn/Ternary.h>

#include <optional>

#include <torch_musa/csrc/aten/mudnn/Exception.h>
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnTernaryMode_t GetTernaryModeFromEnum(Ternary::Mode mode) {
#define CASE(MODE1, MODE2)     \
  case Ternary::Mode::MODE1: { \
    return MODE2;              \
  }

  switch (mode) {
    MUDNN_FORALL_TERNARY_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

template <typename T>
void SetTernaryImpl(
    Ternary& op,
    Ternary::Mode mode,
    std::optional<T> alpha,
    std::optional<T> beta,
    std::optional<T> gamma) {
  const auto m = GetTernaryModeFromEnum(mode);
  MudnnParam a, b, c;
  if (alpha) {
    a.Set(alpha.value());
  }
  if (beta) {
    b.Set(beta.value());
  }
  if (gamma) {
    c.Set(gamma.value());
  }
  CHECK_MUDNN_STATUS(
      mudnnSetTernaryDescriptor(
          op.Desc(),
          m,
          a.Type(),
          a.Ptr(),
          b.Type(),
          b.Ptr(),
          c.Type(),
          c.Ptr()),
      "mudnnSetTernaryDescriptor");
}

#else

template <typename T>
void SetTernaryImpl(
    Ternary& op,
    Ternary::Mode mode,
    std::optional<T> alpha,
    std::optional<T> beta,
    std::optional<T> gamma) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  if (alpha) {
    CHECK_MUDNN_STATUS(op.SetAlpha(alpha.value()), "SetAlpha");
  }
  if (beta) {
    CHECK_MUDNN_STATUS(op.SetBeta(beta.value()), "SetBeta");
  }
  if (gamma) {
    CHECK_MUDNN_STATUS(op.SetGamma(gamma.value()), "SetBeta");
  }
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Ternary::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& input1,
    const muTensor& input2,
    const muTensor& input3) const {
  out.Build();
  input1.Build();
  input2.Build();
  input3.Build();
  return mudnnTernary(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(input1),
      MUDNN_UNPACK_TENSOR(input2),
      MUDNN_UNPACK_TENSOR(input3));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetTernary(Ternary& op, Ternary::Mode mode) {
  SetTernaryImpl<double>(op, mode, std::nullopt, std::nullopt, std::nullopt);
}

void SetTernary(Ternary& op, Ternary::Mode mode, int64_t alpha) {
  SetTernaryImpl<int64_t>(op, mode, alpha, std::nullopt, std::nullopt);
}

void SetTernary(Ternary& op, Ternary::Mode mode, double alpha) {
  SetTernaryImpl<double>(op, mode, alpha, std::nullopt, std::nullopt);
}

} // namespace at::musa
