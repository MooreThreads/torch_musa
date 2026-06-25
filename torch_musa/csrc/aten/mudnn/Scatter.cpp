#include "torch_musa/csrc/aten/mudnn/Scatter.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnScatterMode_t GetScatterModeFromEnum(::musa::dnn::Scatter::Mode mode) {
#define CASE(MODE1, MODE2)                  \
  case ::musa::dnn::Scatter::Mode::MODE1: { \
    return MODE2;                           \
  }

  switch (mode) {
    MUDNN_FORALL_SCATTER_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

void SetScatterImpl(::musa::dnn::Scatter& op, ::musa::dnn::Scatter::Mode mode) {
  const auto m = GetScatterModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetScatterDescriptor(op.Desc(), m), "mudnnSetScatterDescriptor");
}

#else

void SetScatterImpl(::musa::dnn::Scatter& op, ::musa::dnn::Scatter::Mode mode) {
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnScatter::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& idx,
    const muTensor& src,
    int dim,
    const MemoryMaintainer& /*maintainer*/) const {
  out.Build();
  idx.Build();
  src.Build();
  return mudnnScatterInplace(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(idx),
      MUDNN_UNPACK_TENSOR(src),
      dim);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetScatter(::musa::dnn::Scatter& op, ::musa::dnn::Scatter::Mode mode) {
  return SetScatterImpl(op, mode);
}

} // namespace at::musa
