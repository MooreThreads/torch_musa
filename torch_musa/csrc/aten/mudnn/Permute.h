#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_PERMUTE_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_PERMUTE_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class Permute : public Descriptor<
                    mudnnPermuteStruct,
                    mudnnCreatePermuteDescriptor,
                    mudnnDestroyPermuteDescriptor> {
 public:
  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const {
    out.Build();
    in.Build();
    return mudnnPermute(
        h, Desc(), MUDNN_UNPACK_TENSOR(out), MUDNN_UNPACK_TENSOR(in));
  }
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Permute;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Permute;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_PERMUTE_H_
