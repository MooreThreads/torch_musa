#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_RMSNORM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_RMSNORM_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

class RMSNorm : public Descriptor<
                    mudnnRMSNormStruct,
                    mudnnCreateRMSNormDescriptor,
                    mudnnDestroyRMSNormDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& invvar,
      const muTensor& in,
      const muTensor& gamma);

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& dx,
      muTensor& dgamma,
      const muTensor& dy,
      const muTensor& in,
      const muTensor& rstd,
      const muTensor& gamma,
      const MemoryMaintainer& maintainer);
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::RMSNorm;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::RMSNorm;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {
void SetRMSNorm(
    RMSNorm& op,
    size_t axis_count,
    const int* axes,
    double eps = 1e-5);
} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_RMSNORM_H_
