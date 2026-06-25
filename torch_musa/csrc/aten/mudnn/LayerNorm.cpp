#include "torch_musa/csrc/aten/mudnn/LayerNorm.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

namespace {

mudnnLayerNormMode_t ToMudnnVarMode(LayerNorm::VarMode mode) {
  switch (mode) {
    case LayerNorm::VarMode::DIRECT:
      return MUDNN_LAYER_NORM_MODE_DIRECT;
    case LayerNorm::VarMode::WELFORD:
      return MUDNN_LAYER_NORM_MODE_WELFORD;
    default:
      throw std::runtime_error("Unknown LayerNorm VarMode");
  }
}

} // anonymous namespace

mudnnStatus_t LayerNorm::Run(
    muHandle& h,
    muTensor& out,
    muTensor& mean,
    muTensor& rstd,
    const muTensor& in,
    const muTensor& gamma,
    const muTensor& beta,
    const MemoryMaintainer& maintainer) {
  out.Build();
  mean.Build();
  rstd.Build();
  in.Build();
  gamma.Build();
  beta.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetLayerNormForwardWorkspaceSize(
          h,
          Desc(),
          out.Desc(),
          mean.Desc(),
          rstd.Desc(),
          in.Desc(),
          gamma.Desc(),
          beta.Desc(),
          &ws_size),
      "mudnnGetLayerNormForwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  return mudnnLayerNormForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(mean),
      MUDNN_UNPACK_TENSOR(rstd),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(gamma),
      MUDNN_UNPACK_TENSOR(beta),
      ws.get(),
      ws_size);
}

mudnnStatus_t LayerNorm::RunBwd(
    muHandle& h,
    muTensor& dX,
    muTensor& dgamma,
    muTensor& dbeta,
    const muTensor& grad_out,
    const muTensor& X,
    const muTensor& mean,
    const muTensor& rstd,
    const muTensor& weight,
    const MemoryMaintainer& maintainer) {
  dX.Build();
  dgamma.Build();
  dbeta.Build();
  grad_out.Build();
  X.Build();
  mean.Build();
  rstd.Build();
  weight.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetLayerNormBackwardWorkspaceSize(
          h,
          Desc(),
          MUDNN_UNPACK_TENSOR(dX),
          MUDNN_UNPACK_TENSOR(dgamma),
          MUDNN_UNPACK_TENSOR(dbeta),
          grad_out.Desc(),
          X.Desc(),
          mean.Desc(),
          rstd.Desc(),
          weight.Desc(),
          &ws_size),
      "mudnnGetLayerNormBackwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  return mudnnLayerNormBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(dX),
      MUDNN_UNPACK_TENSOR(dgamma),
      MUDNN_UNPACK_TENSOR(dbeta),
      MUDNN_UNPACK_TENSOR(grad_out),
      MUDNN_UNPACK_TENSOR(X),
      MUDNN_UNPACK_TENSOR(mean),
      MUDNN_UNPACK_TENSOR(rstd),
      MUDNN_UNPACK_TENSOR(weight),
      ws.get(),
      ws_size);
}

void SetLayerNorm(
    LayerNorm& op,
    size_t axis_count,
    const int32_t* axes,
    double eps,
    LayerNorm::VarMode var_mode) {
  const auto vm = ToMudnnVarMode(var_mode);
  CHECK_MUDNN_STATUS(
      mudnnSetLayerNormDescriptor(
          op.Desc(), eps, vm, const_cast<int*>(axes), axis_count),
      "mudnnSetLayerNormDescriptor");
}

#else

void SetLayerNorm(
    ::musa::dnn::LayerNorm& op,
    size_t axis_count,
    const int32_t* axes,
    double eps,
    ::musa::dnn::LayerNorm::VarMode var_mode) {
  CHECK_MUDNN_STATUS(op.SetAxis(axis_count, axes), "SetAxis");
  CHECK_MUDNN_STATUS(op.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(op.SetVarMode(var_mode), "SetVarMode");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
