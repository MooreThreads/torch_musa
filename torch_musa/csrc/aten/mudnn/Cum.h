#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_CUM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_CUM_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class Cum : public Descriptor<
                mudnnCumStruct,
                mudnnCreateCumDescriptor,
                mudnnDestroyCumDescriptor> {
 public:
  enum class Mode {
    ADD,
    MUL,
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& input,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::Cum Cum;
} // namespace musa::dnn

#else

namespace at::musa {
typedef class ::musa::dnn::Cum Cum;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetCum(Cum& op, int64_t dim, Cum::Mode mode);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_CUM_H_
