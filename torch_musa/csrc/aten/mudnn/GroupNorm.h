#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPNORM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPNORM_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class GroupNorm : public Descriptor<
                      mudnnGroupNormStruct,
                      mudnnCreateGroupNormDescriptor,
                      mudnnDestroyGroupNormDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& mean,
      muTensor& rstd,
      const muTensor& in,
      const muTensor& gamma,
      const muTensor& beta);
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::GroupNorm;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::GroupNorm;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetGroupNorm(GroupNorm& gn, double eps, int axis, int group);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_GROUPNORM_H_
