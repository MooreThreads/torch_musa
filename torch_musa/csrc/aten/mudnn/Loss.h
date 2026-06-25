#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_LOSS_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_LOSS_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_LOSS_APPLY(_, MODE) _(MODE, MUDNN_LOSS_REDUCE_MODE_##MODE)

#define MUDNN_FORALL_LOSS_OPS(_) \
  MUDNN_LOSS_APPLY(_, NONE)      \
  MUDNN_LOSS_APPLY(_, MEAN)      \
  MUDNN_LOSS_APPLY(_, SUM)       \
  MUDNN_LOSS_APPLY(_, BATCHMEAN)

class NLLLoss : public Descriptor<
                    mudnnNLLLossStruct,
                    mudnnCreateNLLLossDescriptor,
                    mudnnDestroyNLLLossDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_LOSS_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& total_weight,
      const muTensor& in,
      const muTensor& target,
      const muTensor& weight,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_input,
      const muTensor& grad_output,
      const muTensor& target,
      const muTensor& weight,
      const muTensor& total_weight,
      const MemoryMaintainer& maintainer) const;
};

class KLDivLoss : public Descriptor<
                      mudnnKLDivLossStruct,
                      mudnnCreateKLDivLossDescriptor,
                      mudnnDestroyKLDivLossDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_LOSS_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      muTensor& in,
      const muTensor& target,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_input,
      const muTensor& grad_output,
      const muTensor& input,
      const muTensor& target) const;
};

class CrossEntropyLoss : public Descriptor<
                             mudnnCrossEntropyLossStruct,
                             mudnnCreateCrossEntropyLossDescriptor,
                             mudnnDestroyCrossEntropyLossDescriptor> {
 public:
  enum class ReductionMode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_LOSS_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      const muTensor& target,
      const muTensor& weight,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunBwd(
      muHandle& h,
      muTensor& grad_input,
      const muTensor& input,
      const muTensor& grad_output,
      const muTensor& target,
      const muTensor& weight,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::CrossEntropyLoss;
using at::musa::KLDivLoss;
using at::musa::NLLLoss;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::CrossEntropyLoss;
using ::musa::dnn::KLDivLoss;
using ::musa::dnn::NLLLoss;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetNLLLoss(NLLLoss& op, NLLLoss::Mode mode, int ignore_index);

void SetKLDivLoss(KLDivLoss& op, KLDivLoss::Mode mode, bool log_target);

void SetCrossEntropyLoss(
    CrossEntropyLoss& op,
    CrossEntropyLoss::ReductionMode mode,
    float label_smoothing,
    int64_t ignore_index);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_LOSS_H_
