#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_NONZERO_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_NONZERO_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnNonzero : public Descriptor<
                         mudnnNonzeroStruct,
                         mudnnCreateNonzeroDescriptor,
                         mudnnDestroyDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& input,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnNonzero Nonzero;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_NONZERO_H_
