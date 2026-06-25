#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_LAYERNORM_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_LAYERNORM_H_

#include <vector>

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

class LayerNorm : public Descriptor<
                      mudnnLayerNormStruct,
                      mudnnCreateLayerNormDescriptor,
                      mudnnDestroyLayerNormDescriptor> {
 public:
  enum class VarMode {
    DIRECT,
    WELFORD,
  };

  mudnnStatus_t SetAxis(size_t count, const int32_t* axes);
  mudnnStatus_t SetEpsilon(double eps);
  mudnnStatus_t SetVarMode(VarMode mode);

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& mean,
      muTensor& rstd,
      const muTensor& in,
      const muTensor& gamma,
      const muTensor& beta,
      const MemoryMaintainer& maintainer);

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& dX,
      muTensor& dgamma,
      muTensor& dbeta,
      const muTensor& grad_out,
      const muTensor& X,
      const muTensor& mean,
      const muTensor& rstd,
      const muTensor& weight,
      const MemoryMaintainer& maintainer);
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::LayerNorm;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::LayerNorm;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {
void SetLayerNorm(
    LayerNorm& op,
    size_t axis_count,
    const int32_t* axes,
    double eps,
    LayerNorm::VarMode var_mode);
} // namespace at::musa
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_LAYERNORM_H_
