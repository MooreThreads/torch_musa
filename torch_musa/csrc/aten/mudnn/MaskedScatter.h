#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_MASKED_SCATTER_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_MASKED_SCATTER_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnMaskedScatter : public Descriptor<
                               mudnnMaskedScatterStruct,
                               mudnnCreateMaskedScatterDescriptor,
                               mudnnDestroyMaskedScatterDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& output,
      const muTensor& mask,
      const muTensor& source,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnMaskedScatter MaskedScatter;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_MASKED_SCATTER_H_
