#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_TERNARY_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_TERNARY_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_TERNARY_APPLY(_, MODE) _(MODE, MUDNN_TERNARY_##MODE)

#define MUDNN_FORALL_TERNARY_OPS(_)     \
  MUDNN_TERNARY_APPLY(_, SELECT)        \
  MUDNN_TERNARY_APPLY(_, SELECTV2)      \
  MUDNN_TERNARY_APPLY(_, ADDCMUL)       \
  MUDNN_TERNARY_APPLY(_, ADDCMUL_ALPHA) \
  MUDNN_TERNARY_APPLY(_, ADDCDIV)       \
  MUDNN_TERNARY_APPLY(_, ADDCDIV_ALPHA) \
  MUDNN_TERNARY_APPLY(_, CLAMP)

class Ternary : public Descriptor<
                    mudnnTernaryStruct,
                    mudnnCreateTernaryDescriptor,
                    mudnnDestroyTernaryDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_TERNARY_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& input1,
      const muTensor& input2,
      const muTensor& input3) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Ternary;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Ternary;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetTernary(Ternary& op, Ternary::Mode mode);

void SetTernary(Ternary& op, Ternary::Mode mode, int64_t alpha);
void SetTernary(Ternary& op, Ternary::Mode mode, double alpha);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_TERNARY_H_
