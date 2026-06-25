#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHNORM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHNORM_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

class BatchNorm : public Descriptor<
                      mudnnBatchNormStruct,
                      mudnnCreateBatchNormDescriptor,
                      mudnnDestroyBatchNormDescriptor> {
 public:
  enum class Mode {
    PER_ACTIVATION,
    PER_CHANNEL,
  };

  mudnnStatus_t RunPure(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      muTensor& rm,
      muTensor& rv,
      const muTensor& w,
      const muTensor& b);

  mudnnStatus_t RunComposite(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      muTensor& rm,
      muTensor& rv,
      muTensor& m,
      muTensor& v,
      const muTensor& w,
      const muTensor& b,
      double momentum,
      const MemoryMaintainer& maintainer);

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& dx,
      muTensor& dm,
      muTensor& dv,
      muTensor& dg,
      muTensor& db,
      muTensor& x,
      muTensor& dy,
      muTensor& m,
      muTensor& v,
      muTensor& g,
      const MemoryMaintainer& maintainer);
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::BatchNorm;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::BatchNorm;
}
// namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetBatchNorm(
    BatchNorm& bn,
    BatchNorm::Mode mode,
    double eps,
    bool training);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHNORM_H_
