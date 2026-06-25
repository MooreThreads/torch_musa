#include "torch_musa/csrc/aten/mudnn/RMSNorm.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetRMSNormImpl(
    RMSNorm& op,
    size_t axis_count,
    const int* axes,
    double eps) {
  CHECK_MUDNN_STATUS(
      mudnnSetRMSNormDescriptor(
          op.Desc(),
          MUDNN_RMS_NORM_MODE_DIRECT,
          eps,
          const_cast<int*>(axes),
          static_cast<int>(axis_count)),
      "mudnnSetRMSNormDescriptor");
}

#else

void SetRMSNormImpl(
    ::musa::dnn::RMSNorm& op,
    size_t axis_count,
    const int* axes,
    double eps) {
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(op.SetAxis(axis_count, axes), "SetAxis");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t RMSNorm::Run(
    muHandle& h,
    muTensor& out,
    muTensor& invvar,
    const muTensor& in,
    const muTensor& gamma) {
  out.Build();
  invvar.Build();
  in.Build();
  gamma.Build();

  return mudnnRMSNormForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(invvar),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(gamma));
}

mudnnStatus_t RMSNorm::RunBwd(
    muHandle& h,
    muTensor& dx,
    muTensor& dgamma,
    const muTensor& dy,
    const muTensor& in,
    const muTensor& rstd,
    const muTensor& gamma,
    const MemoryMaintainer& maintainer) {
  dx.Build();
  dgamma.Build();
  dy.Build();
  in.Build();
  rstd.Build();
  gamma.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetRMSNormBackwardWorkspaceSize(
          h,
          Desc(),
          MUDNN_UNPACK_TENSOR(dx),
          MUDNN_UNPACK_TENSOR(dgamma),
          dy.Desc(),
          in.Desc(),
          rstd.Desc(),
          gamma.Desc(),
          &ws_size),
      "mudnnGetRMSNormBackwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  return mudnnRMSNormBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(dx),
      MUDNN_UNPACK_TENSOR(dgamma),
      MUDNN_UNPACK_TENSOR(dy),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(rstd),
      MUDNN_UNPACK_TENSOR(gamma),
      ws.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetRMSNorm(RMSNorm& op, size_t axis_count, const int* axes, double eps) {
  return SetRMSNormImpl(op, axis_count, axes, eps);
}

} // namespace at::musa
