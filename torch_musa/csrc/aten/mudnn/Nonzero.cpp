#include "torch_musa/csrc/aten/mudnn/Nonzero.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnNonzero::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& input,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  input.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetNonzeroWorkspaceSize(
          h, Desc(), input.Desc(), out.Desc(), &ws_size),
      "mudnnGetNonzeroWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnNonzero(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(input),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
