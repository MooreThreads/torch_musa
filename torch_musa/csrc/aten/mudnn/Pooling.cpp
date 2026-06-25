#include "torch_musa/csrc/aten/mudnn/Pooling.h"

#include <optional>

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnPoolingMode_t GetPoolingModeFromEnum(Pooling::Mode mode) {
#define CASE(MODE1, MODE2)     \
  case Pooling::Mode::MODE1: { \
    return MODE2;              \
  }

  switch (mode) {
    MUDNN_FORALL_POOLING_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

void SetPoolingImpl(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation,
    std::optional<int> divisor) {
  const auto nd = kernel.size();
  TORCH_INTERNAL_ASSERT(
      nd == pad.size() && nd == stride.size() && nd == dilation.size());
  const auto m = GetPoolingModeFromEnum(mode);
  const int d = divisor.value_or(0);
  CHECK_MUDNN_STATUS(
      mudnnSetPoolingNdDescriptorEx(
          op.Desc(),
          m,
          d,
          static_cast<int>(nd),
          std::data(kernel),
          std::data(pad),
          std::data(stride),
          std::data(dilation)),
      "mudnnSetPoolingNdDescriptorEx");
}

#else

void SetPoolingImpl(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation,
    std::optional<int> divisor) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(
      op.SetNdInfo(
          std::move(kernel),
          std::move(pad),
          std::move(stride),
          std::move(dilation)),
      "SetNdInfo");
  if (divisor) {
    CHECK_MUDNN_STATUS(op.SetDivisor(divisor.value()), "SetDivisor");
  }
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Pooling::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    muTensor& indices) const {
  out.Build();
  in.Build();
  indices.Build();
  return mudnnPoolingForwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(indices));
}

mudnnStatus_t Pooling::RunBwd(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    muTensor& indices) const {
  out.Build();
  in.Build();
  indices.Build();
  return mudnnPoolingBackwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(indices));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetPooling(Pooling& op, Pooling::Mode mode) {
  SetPoolingImpl(op, mode, {}, {}, {}, {}, std::nullopt);
}

void SetPooling(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation) {
  SetPoolingImpl(
      op,
      mode,
      std::move(kernel),
      std::move(pad),
      std::move(stride),
      std::move(dilation),
      std::nullopt);
}

void SetPooling(
    Pooling& op,
    Pooling::Mode mode,
    std::initializer_list<int> kernel,
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation,
    int divisor) {
  SetPoolingImpl(
      op,
      mode,
      std::move(kernel),
      std::move(pad),
      std::move(stride),
      std::move(dilation),
      divisor);
}

} // namespace at::musa
