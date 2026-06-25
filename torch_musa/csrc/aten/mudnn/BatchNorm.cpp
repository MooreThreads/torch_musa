#include "torch_musa/csrc/aten/mudnn/BatchNorm.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnBatchNormMode_t ToMudnnMode(BatchNorm::Mode mode) {
  switch (mode) {
    case BatchNorm::Mode::PER_ACTIVATION:
      return MUDNN_BATCHNORM_PER_ACTIVATION;
    case BatchNorm::Mode::PER_CHANNEL:
      return MUDNN_BATCHNORM_SPATIAL;
    default:
      throw std::runtime_error("Unknown BatchNorm mode");
  }
}

void SetBatchNormImpl(
    BatchNorm& bn,
    BatchNorm::Mode mode,
    double eps,
    bool training) {
  const auto m = ToMudnnMode(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetBatchNormDescriptor(bn.Desc(), m, eps, training),
      "mudnnSetBatchNormDescriptor");
}

#else

void SetBatchNormImpl(
    BatchNorm& bn,
    BatchNorm::Mode mode,
    double eps,
    bool training) {
  CHECK_MUDNN_STATUS(bn.SetEpsilon(eps), "SetEpsilon");
  CHECK_MUDNN_STATUS(bn.SetTraining(training), "SetTraining");
  CHECK_MUDNN_STATUS(bn.SetMode(mode), "SetMode");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t BatchNorm::RunPure(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    muTensor& rm,
    muTensor& rv,
    const muTensor& w,
    const muTensor& b) {
  out.Build();
  in.Build();
  rm.Build();
  rv.Build();
  w.Build();
  b.Build();

  return mudnnBatchNormForwardInference(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(rm),
      MUDNN_UNPACK_TENSOR(rv),
      MUDNN_UNPACK_TENSOR(w),
      MUDNN_UNPACK_TENSOR(b));
}

mudnnStatus_t BatchNorm::RunComposite(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    muTensor& rm,
    muTensor& rv,
    muTensor& m,
    muTensor& v,
    const muTensor& w,
    const muTensor& b,
    double momentum,
    const MemoryMaintainer& maintainer) {
  out.Build();
  in.Build();
  rm.Build();
  rv.Build();
  m.Build();
  v.Build();
  w.Build();
  b.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetBatchNormForwardTrainingWorkspaceSize(
          h,
          Desc(),
          out.Desc(),
          in.Desc(),
          rm.Desc(),
          rv.Desc(),
          m.Desc(),
          v.Desc(),
          w.Desc(),
          b.Desc(),
          &ws_size),
      "mudnnGetBatchNormForwardTrainingWorkspaceSize");

  auto ws = maintainer(ws_size);
  return mudnnBatchNormForwardTraining(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(rm),
      MUDNN_UNPACK_TENSOR(rv),
      MUDNN_UNPACK_TENSOR(m),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(w),
      MUDNN_UNPACK_TENSOR(b),
      momentum,
      ws.get(),
      ws_size);
}

mudnnStatus_t BatchNorm::RunBwd(
    muHandle& h,
    muTensor& dx,
    muTensor& dm,
    muTensor& dv,
    muTensor& dg,
    muTensor& db,
    muTensor& x,
    muTensor& dy,
    muTensor& m,
    muTensor& v,
    muTensor& g,
    const MemoryMaintainer& maintainer) {
  dx.Build();
  dm.Build();
  dv.Build();
  dg.Build();
  db.Build();
  x.Build();
  dy.Build();
  m.Build();
  v.Build();
  g.Build();

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetBatchNormBackwardWorkspaceSize(
          h,
          Desc(),
          dx.Desc(),
          dm.Desc(),
          dv.Desc(),
          dg.Desc(),
          db.Desc(),
          x.Desc(),
          dy.Desc(),
          m.Desc(),
          v.Desc(),
          g.Desc(),
          &ws_size),
      "mudnnGetBatchNormBackwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  return mudnnBatchNormBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(dx),
      MUDNN_UNPACK_TENSOR(dm),
      MUDNN_UNPACK_TENSOR(dv),
      MUDNN_UNPACK_TENSOR(dg),
      MUDNN_UNPACK_TENSOR(db),
      MUDNN_UNPACK_TENSOR(x),
      MUDNN_UNPACK_TENSOR(dy),
      MUDNN_UNPACK_TENSOR(m),
      MUDNN_UNPACK_TENSOR(v),
      MUDNN_UNPACK_TENSOR(g),
      ws.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetBatchNorm(
    BatchNorm& bn,
    BatchNorm::Mode mode,
    double eps,
    bool training) {
  return SetBatchNormImpl(bn, mode, eps, training);
}

} // namespace at::musa
