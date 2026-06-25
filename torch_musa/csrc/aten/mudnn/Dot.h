#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_DOT_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_DOT_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnDot : public Descriptor<
                     mudnnDotStruct,
                     mudnnCreateDotDescriptor,
                     mudnnDestroyDotDescriptor> {
 public:
  mudnnStatus_t SetComputeMode(ComputeMode mode) {
    return mudnnSetDotMathType(Desc(), mode);
  }

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& l,
      const muTensor& r,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnDot Dot;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_DOT_H_
