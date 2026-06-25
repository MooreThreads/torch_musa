#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SOFTMAX_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SOFTMAX_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnSoftmax : public Descriptor<
                         mudnnSoftmaxStruct,
                         mudnnCreateSoftmaxDescriptor,
                         mudnnDestroySoftmaxDescriptor> {
 public:
  enum class Mode { SOFTMAX, LOGSOFTMAX, LOGSUMEXP };
  enum class Algorithm { ACCURATE };

  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const;
  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_in,
      const muTensor& out,
      const muTensor& grad_out) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnSoftmax Softmax;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetSoftmax(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim);
void SetSoftmax(
    ::musa::dnn::Softmax& op,
    ::musa::dnn::Softmax::Mode mode,
    int dim,
    ::musa::dnn::Softmax::Algorithm algo);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SOFTMAX_H_
