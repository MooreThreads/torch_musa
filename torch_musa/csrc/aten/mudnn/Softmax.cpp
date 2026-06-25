#include "torch_musa/csrc/aten/mudnn/Softmax.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnSoftmaxMode_t GetSoftmaxModeFromEnum(::musa::dnn::Softmax::Mode mode) {
  switch (mode) {
    case ::musa::dnn::Softmax::Mode::SOFTMAX:
      return MUDNN_SOFTMAX_MODE_SOFTMAX;
    case ::musa::dnn::Softmax::Mode::LOGSOFTMAX:
      return MUDNN_SOFTMAX_MODE_LOG_SOFTMAX;
    case ::musa::dnn::Softmax::Mode::LOGSUMEXP:
      return MUDNN_SOFTMAX_MODE_LOG_SUM_EXP;
    default:
      throw std::runtime_error("unreachable");
  }
}

mudnnSoftmaxAlgorithm_t GetSoftmaxAlgoFromEnum(
    ::musa::dnn::Softmax::Algorithm algo) {
  switch (algo) {
    case ::musa::dnn::Softmax::Algorithm::ACCURATE:
      return MUDNN_SOFTMAX_ACCURATE;
    default:
      throw std::runtime_error("unreachable");
  }
}

void SetSoftmaxImpl(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim,
    ::musa::dnn::Softmax::Algorithm algo) {
  const auto m = GetSoftmaxModeFromEnum(mode);
  const auto a = GetSoftmaxAlgoFromEnum(algo);
  CHECK_MUDNN_STATUS(
      mudnnSetSoftmaxDescriptor(op.Desc(), dim, a, m),
      "mudnnSetSoftmaxDescriptor");
}

#else

void SetSoftmaxImpl(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim,
    ::musa::dnn::Softmax::Algorithm algo) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
  CHECK_MUDNN_STATUS(op.SetDim(dim), "SetDim");
  CHECK_MUDNN_STATUS(op.SetAlgorithm(algo), "SetAlgorithm");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnSoftmax::Run(muHandle& h, muTensor& out, const muTensor& in)
    const {
  out.Build();
  in.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetSoftmaxForwardWorkspaceSize(
          h, Desc(), in.Desc(), out.Desc(), &ws_size),
      "mudnnGetSoftmaxForwardWorkspaceSize");
  auto mem = InternalMemAlloc(ws_size);
  return mudnnSoftmaxForwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t MudnnSoftmax::RunBwd(
    muHandle& h,
    muTensor& grad_in,
    const muTensor& out,
    const muTensor& grad_out) const {
  grad_in.Build();
  out.Build();
  grad_out.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetSoftmaxBackwardWorkspaceSize(
          h, Desc(), out.Desc(), grad_out.Desc(), grad_in.Desc(), &ws_size),
      "mudnnGetSoftmaxBackwardWorkspaceSize");
  auto mem = InternalMemAlloc(ws_size);
  return mudnnSoftmaxBackwardEx(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(grad_out),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(grad_in));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetSoftmax(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim) {
  return SetSoftmaxImpl(
      op, mode, dim, ::musa::dnn::Softmax::Algorithm::ACCURATE);
}

void SetSoftmax(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim,
    ::musa::dnn::Softmax::Algorithm algo) {
  return SetSoftmaxImpl(op, mode, dim, algo);
}

} // namespace at::musa
