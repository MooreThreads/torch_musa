#include "torch_musa/csrc/aten/mudnn/Cum.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnCumMode_t GetCumModeFromEnum(Cum::Mode mode) {
  switch (mode) {
    case Cum::Mode::ADD:
      return MUDNN_CUM_ADD;
    case Cum::Mode::MUL:
      return MUDNN_CUM_MUL;
  }
  // Fallback, should not reach here.
  return MUDNN_CUM_ADD;
}

void SetCumImpl(Cum& op, int64_t dim, Cum::Mode mode) {
  const auto m = GetCumModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetCumDescriptor(op.Desc(), static_cast<int>(dim), m),
      "mudnnSetCumDescriptor");
}

#else

void SetCumImpl(Cum& op, int64_t dim, Cum::Mode mode) {
  CHECK_MUDNN_STATUS(op.SetDim(static_cast<int>(dim)), "SetDim");
  CHECK_MUDNN_STATUS(op.SetMode(mode), "SetMode");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t Cum::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& input,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  input.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetCumWorkspaceSize(h, Desc(), out.Desc(), input.Desc(), &ws_size),
      "mudnnGetCumWorkspaceSize");
  auto ws = maintainer(ws_size);
  return mudnnCum(
      h,
      Desc(),
      out.Desc(),
      const_cast<void*>(out.DataPtr()),
      input.Desc(),
      input.DataPtr(),
      ws.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetCum(Cum& op, int64_t dim, Cum::Mode mode) {
  return SetCumImpl(op, dim, mode);
}

} // namespace at::musa
