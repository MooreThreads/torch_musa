#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_ROPE_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_ROPE_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class MudnnRope : public Descriptor<
                      mudnnRopeStruct,
                      mudnnCreateRopeDescriptor,
                      mudnnDestroyRopeDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& output,
      const muTensor& input,
      const muTensor& freqs_cis) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_input,
      const muTensor& grad_output,
      const muTensor& freqs_cis) const;
};

} // namespace at::musa

namespace musa::dnn {
typedef class ::at::musa::MudnnRope Rope;
} // namespace musa::dnn

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetRope(
    ::musa::dnn::Rope& op,
    bool rotary_interleaved,
    bool batch_first,
    bool multi_latent_attention);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_ROPE_H_
