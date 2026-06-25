#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_FILL_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_FILL_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnFill : public Descriptor<
                      mudnnFillStruct,
                      mudnnCreateFillDescriptor,
                      mudnnDestroyFillDescriptor> {
 public:
  mudnnStatus_t Run(muHandle& h, muTensor& out) const;

  mudnnStatus_t Run(muHandle& h, muTensor& out, muTensor& mask) const;
};

} // namespace at::musa

namespace musa::dnn {
using Fill = at::musa::MudnnFill;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetFill(::musa::dnn::Fill& op, int64_t value);
void SetFill(::musa::dnn::Fill& op, double value);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_FILL_H_
