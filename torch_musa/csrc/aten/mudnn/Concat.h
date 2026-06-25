#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_CONCAT_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_CONCAT_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class Concat : public Descriptor<
                   mudnnConcatStruct,
                   mudnnCreateConcatDescriptor,
                   mudnnDestroyConcatDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      int num_input,
      const muTensor* ins) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Concat;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Concat;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetConcat(Concat& op, int axis);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_CONCAT_H_
