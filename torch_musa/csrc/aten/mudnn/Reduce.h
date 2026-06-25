#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_REDUCE_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_REDUCE_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_REDUCE_APPLY(_, MODE) _(MODE, MUDNN_REDUCE_##MODE)

#define MUDNN_FORALL_REDUCE_OPS(_)    \
  MUDNN_REDUCE_APPLY(_, MAX)          \
  MUDNN_REDUCE_APPLY(_, ADD)          \
  MUDNN_REDUCE_APPLY(_, MUL)          \
  MUDNN_REDUCE_APPLY(_, MEAN)         \
  MUDNN_REDUCE_APPLY(_, MIN)          \
  MUDNN_REDUCE_APPLY(_, PROD)         \
  MUDNN_REDUCE_APPLY(_, AND)          \
  MUDNN_REDUCE_APPLY(_, NORM)         \
  MUDNN_REDUCE_APPLY(_, OR)           \
  MUDNN_REDUCE_APPLY(_, MUL_NO_ZEROS) \
  MUDNN_REDUCE_APPLY(_, L2LOSS)       \
  MUDNN_REDUCE_APPLY(_, MAX_UNSTABLE) \
  MUDNN_REDUCE_APPLY(_, MIN_UNSTABLE) \
  MUDNN_REDUCE_APPLY(_, VARIANCE)     \
  MUDNN_REDUCE_APPLY(_, STD)

class Reduce : public Descriptor<
                   mudnnReduceStruct,
                   mudnnCreateReduceDescriptor,
                   mudnnDestroyReduceDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_REDUCE_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunIndices(
      muHandle& h,
      muTensor& out,
      const muTensor& in,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunWithIndices(
      muHandle& h,
      muTensor& out,
      muTensor& indices,
      const muTensor& in,
      const MemoryMaintainer& maintainer) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Reduce;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Reduce;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetReduce(Reduce& op, Reduce::Mode mode, int ndim, const int* dim);

void SetReduceCorrection(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    int correction);

void SetReduceNorm(
    Reduce& op,
    Reduce::Mode mode,
    int ndim,
    const int* dim,
    float ord);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_REDUCE_H_
