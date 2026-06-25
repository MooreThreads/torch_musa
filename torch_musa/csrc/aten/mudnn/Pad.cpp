#include <torch_musa/csrc/aten/mudnn/Pad.h>

#include <optional>

#include <torch_musa/csrc/aten/mudnn/Exception.h>
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnPadMode_t GetPadModeFromEnum(Pad::Mode mode) {
#define CASE(MODE1, MODE2) \
  case Pad::Mode::MODE1: { \
    return MODE2;          \
  }

  switch (mode) {
    MUDNN_FORALL_PAD_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

template <typename T>
void SetPadImpl(
    Pad& op,
    Pad::Mode mode,
    std::optional<T> v,
    int len,
    const int* pad) {
  const auto m = GetPadModeFromEnum(mode);

  if constexpr (std::is_same_v<T, double>) {
    double value = v.value_or(0.0);
    CHECK_MUDNN_STATUS(
        mudnnSetPadDescriptor(
            op.Desc(),
            m,
            MUDNN_PARAM_DATA_DOUBLE,
            &value,
            const_cast<int*>(pad),
            static_cast<size_t>(len)),
        "mudnnSetPadDescriptor");
    return;
  } else {
    MudnnParam value;
    value.Set(v.value_or(0.0));

    CHECK_MUDNN_STATUS(
        mudnnSetPadDescriptor(
            op.Desc(),
            m,
            value.Type(),
            value.Ptr(),
            const_cast<int*>(pad),
            static_cast<size_t>(len)),
        "mudnnSetPadDescriptor");
  }
}

#else

template <typename T>
void SetPadImpl(
    Pad& op,
    Pad::Mode mode,
    std::optional<T> v,
    int len,
    const int* pad) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  if (v) {
    CHECK_MUDNN_STATUS(op.SetValue(v.value()), "SetValue");
  }
  CHECK_MUDNN_STATUS(op.SetPaddingInfo(len, pad), "SetPaddingInfo");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Pad::Run(muHandle& h, muTensor& out, const muTensor& in) const {
  out.Build();
  in.Build();
  return mudnnPad(h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetPad(Pad& op, Pad::Mode mode, int len, const int* pad) {
  SetPadImpl<double>(op, mode, std::nullopt, len, pad);
}

void SetPad(Pad& op, Pad::Mode mode, double v, int len, const int* pad) {
  SetPadImpl<double>(op, mode, v, len, pad);
}

} // namespace at::musa
