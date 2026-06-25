#include "torch_musa/csrc/aten/mudnn/MaskedScatter.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnMaskedScatter::Run(
    muHandle& h,
    muTensor& output,
    const muTensor& mask,
    const muTensor& source,
    const MemoryMaintainer& maintainer) const {
  output.Build();
  mask.Build();
  source.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetMaskedScatterWorkspaceSize(
          h, Desc(), output.Desc(), mask.Desc(), source.Desc(), &ws_size),
      "mudnnGetMaskedScatterWorkspaceSize");
  auto ws = maintainer(ws_size);
  return mudnnMaskedScatter(
      h,
      Desc(),
      output.Desc(),
      const_cast<void*>(output.DataPtr()),
      MUDNN_UNPACK_TENSOR(mask),
      MUDNN_UNPACK_TENSOR(source),
      ws.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
