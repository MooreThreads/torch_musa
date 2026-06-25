#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_MASKEDSELECT_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_MASKEDSELECT_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Exception.h"
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnMaskedSelect : public Descriptor<
                              mudnnMaskedSelectStruct,
                              mudnnCreateMaskedSelectDescriptor,
                              mudnnDestroyMaskedSelectDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& input,
      const muTensor& mask,
      const MemoryMaintainer& maintainer) const {
    out.Build();
    input.Build();
    mask.Build();
    size_t ws_size = 0;
    CHECK_MUDNN_STATUS(
        mudnnGetMaskedSelectWorkspaceSize(
            h, Desc(), input.Desc(), mask.Desc(), out.Desc(), &ws_size),
        "mudnnGetMaskedSelectWorkspaceSize");
    auto mem = maintainer(ws_size);
    return mudnnMaskedSelect(
        h,
        Desc(),
        MUDNN_UNPACK_TENSOR(input),
        MUDNN_UNPACK_TENSOR(mask),
        MUDNN_UNPACK_TENSOR(out),
        mem.get(),
        ws_size);
  }
};

} // namespace at::musa

namespace musa::dnn {
using MaskedSelect = at::musa::MudnnMaskedSelect;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_MASKEDSELECT_H_
