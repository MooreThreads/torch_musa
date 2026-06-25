#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_GLU_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_GLU_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnGlu : public Descriptor<
                     mudnnGluStruct,
                     mudnnCreateGluDescriptor,
                     mudnnDestroyGluDescriptor> {
 public:
  mudnnStatus_t SetAxis(int dim) {
    return mudnnSetGluDescriptor(Desc(), dim);
  }

  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const {
    out.Build();
    in.Build();
    return mudnnGlu(
        h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
  }
};

} // namespace at::musa

namespace musa::dnn {
using Glu = at::musa::MudnnGlu;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_GLU_H_
