#include "torch_musa/csrc/aten/mudnn/Interpolate.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnInterpolateMode_t GetInterpolateModeFromEnum(Interpolate::Mode mode) {
#define CASE(MODE1, MODE2)         \
  case Interpolate::Mode::MODE1: { \
    return MODE2;                  \
  }

  switch (mode) {
    MUDNN_FORALL_INTERPOLATE_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

void SetInterpolateImpl(
    Interpolate& op,
    Interpolate::Mode mode,
    bool antialias,
    bool align_corners,
    const std::vector<float>& scales) {
  std::vector<float> scales_copy(scales);
  const auto m = GetInterpolateModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetInterpolateDescriptor(
          op.Desc(),
          m,
          static_cast<int>(scales_copy.size()),
          scales_copy.data(),
          align_corners,
          antialias),
      "mudnnSetInterpolateDescriptor");
}

#else

void SetInterpolateImpl(
    Interpolate& op,
    Interpolate::Mode mode,
    bool antialias,
    bool align_corners,
    const std::vector<float>& scales) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(op.SetAntialias(antialias), "SetAntialias");
  CHECK_MUDNN_STATUS(op.SetAlignCorners(align_corners), "SetAlignCorners");
  CHECK_MUDNN_STATUS(
      op.SetScaleInfo(
          static_cast<int>(scales.size()), const_cast<float*>(scales.data())),
      "SetScaleInfo");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Interpolate::Run(muHandle& h, muTensor& out, const muTensor& in)
    const {
  out.Build();
  in.Build();
  return mudnnInterpolateForward(
      h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
}

mudnnStatus_t Interpolate::RunBackward(
    muHandle& h,
    muTensor& grad_in,
    const muTensor& grad_out) const {
  grad_in.Build();
  grad_out.Build();
  return mudnnInterpolateBackward(
      h, Desc(), MUDNN_UNPACK_TENSOR(grad_in), MUDNN_UNPACK_TENSOR(grad_out));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetInterpolate(
    Interpolate& op,
    Interpolate::Mode mode,
    bool antialias,
    bool align_corners,
    const std::vector<float>& scales) {
  return SetInterpolateImpl(op, mode, antialias, align_corners, scales);
}

} // namespace at::musa
