#include "torch_musa/csrc/aten/mudnn/GroupNorm.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetGroupNormImpl(GroupNorm& gn, double eps, int axis, int group) {
  CHECK_MUDNN_STATUS(
      mudnnSetGroupNormDescriptor(gn.Desc(), eps, axis, group),
      "mudnnSetGroupNormDescriptor");
}

#else

void SetGroupNormImpl(GroupNorm& gn, double eps, int axis, int group) {
  CHECK_MUDNN_STATUS(gn.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(gn.SetAxis(axis), "SetAxis");
  CHECK_MUDNN_STATUS(gn.SetGroup(group), "SetGroup");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t GroupNorm::Run(
    muHandle& h,
    muTensor& out,
    muTensor& mean,
    muTensor& rstd,
    const muTensor& in,
    const muTensor& gamma,
    const muTensor& beta) {
  out.Build();
  mean.Build();
  rstd.Build();
  in.Build();
  gamma.Build();
  beta.Build();

  return mudnnGroupNorm(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(mean),
      MUDNN_UNPACK_TENSOR(rstd),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(gamma),
      MUDNN_UNPACK_TENSOR(beta));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetGroupNorm(GroupNorm& gn, double eps, int axis, int group) {
  return SetGroupNormImpl(gn, eps, axis, group);
}

} // namespace at::musa
