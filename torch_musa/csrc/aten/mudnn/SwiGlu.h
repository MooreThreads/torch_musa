#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_SWIGLU_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_SWIGLU_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class SwiGlu : public Descriptor<
                   mudnnSwiGluStruct,
                   mudnnCreateSwiGluDescriptor,
                   mudnnDestroySwiGluDescriptor> {
 public:
  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const {
    out.Build();
    in.Build();
    return mudnnSwiGluForward(
        h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
  }

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      const muTensor& grad) const {
    out.Build();
    in.Build();
    grad.Build();
    return mudnnSwiGluBackward(
        h,
        Desc(),
        MUDNN_UNPACK_TENSOR(out),
        MUDNN_UNPACK_TENSOR(in),
        MUDNN_UNPACK_TENSOR(grad));
  }
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::SwiGlu;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::SwiGlu;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_SWIGLU_H_
