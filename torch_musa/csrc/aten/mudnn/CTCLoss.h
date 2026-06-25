#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_CTCLOSS_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_CTCLOSS_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

class CTCLoss : public Descriptor<
                    mudnnCTCLossStruct,
                    mudnnCreateCTCLossDescriptor,
                    mudnnDestroyCTCLossDescriptor> {
 public:
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& alpha,
      const muTensor& input,
      const muTensor& target,
      const muTensor& input_lengths,
      const muTensor& target_lengths) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_input,
      const muTensor& grad_out,
      const muTensor& input,
      const muTensor& target,
      const muTensor& input_lengths,
      const muTensor& target_lengths,
      const muTensor& out,
      const muTensor& alpha,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::CTCLoss;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::CTCLoss;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetCTCLoss(
    CTCLoss& op,
    int64_t blank,
    bool zero_infinity,
    int64_t max_target_length);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_CTCLOSS_H_
