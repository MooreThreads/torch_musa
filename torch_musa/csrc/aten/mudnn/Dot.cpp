#include "torch_musa/csrc/aten/mudnn/Dot.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnDot::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& l,
    const muTensor& r,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  l.Build();
  r.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetDotWorkspaceSize(
          h, Desc(), l.Desc(), r.Desc(), out.Desc(), &ws_size),
      "mudnnGetDotWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnDot(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(l),
      MUDNN_UNPACK_TENSOR(r),
      out.Desc(),
      const_cast<void*>(out.DataPtr()),
      mem.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
