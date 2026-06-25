#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_BINARY_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_BINARY_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_BINARY_APPLY(_, MODE) _(MODE, MUDNN_BINARY_##MODE)

#define MUDNN_FORALL_BINARY_OPS(_)     \
  MUDNN_BINARY_APPLY(_, ADD)           \
  MUDNN_BINARY_APPLY(_, SUB)           \
  MUDNN_BINARY_APPLY(_, MUL)           \
  MUDNN_BINARY_APPLY(_, DIV)           \
  MUDNN_BINARY_APPLY(_, POW)           \
  MUDNN_BINARY_APPLY(_, ADD_ALPHA)     \
  MUDNN_BINARY_APPLY(_, SUB_ALPHA)     \
  MUDNN_BINARY_APPLY(_, LOGICAL_AND)   \
  MUDNN_BINARY_APPLY(_, LOGICAL_OR)    \
  MUDNN_BINARY_APPLY(_, LOGICAL_XOR)   \
  MUDNN_BINARY_APPLY(_, LT)            \
  MUDNN_BINARY_APPLY(_, LE)            \
  MUDNN_BINARY_APPLY(_, GT)            \
  MUDNN_BINARY_APPLY(_, GE)            \
  MUDNN_BINARY_APPLY(_, EQ)            \
  MUDNN_BINARY_APPLY(_, NE)            \
  MUDNN_BINARY_APPLY(_, MAX)           \
  MUDNN_BINARY_APPLY(_, MIN)           \
  MUDNN_BINARY_APPLY(_, PRELU)         \
  MUDNN_BINARY_APPLY(_, LEAKY_RELU_BW) \
  MUDNN_BINARY_APPLY(_, RELU6_BW)      \
  MUDNN_BINARY_APPLY(_, THRESHOLD_BW)  \
  MUDNN_BINARY_APPLY(_, SIGMOID_BW)    \
  MUDNN_BINARY_APPLY(_, SILU_BW)       \
  MUDNN_BINARY_APPLY(_, TANH_BW)       \
  MUDNN_BINARY_APPLY(_, GELU_NONE_BW)  \
  MUDNN_BINARY_APPLY(_, GELU_TANH_BW)  \
  MUDNN_BINARY_APPLY(_, RSQRT_BW)      \
  MUDNN_BINARY_APPLY(_, TRUEDIV)       \
  MUDNN_BINARY_APPLY(_, FLOORDIV)      \
  MUDNN_BINARY_APPLY(_, TRUNCATEDIV)   \
  MUDNN_BINARY_APPLY(_, FLOORMOD)      \
  MUDNN_BINARY_APPLY(_, TRUNCATEMOD)   \
  MUDNN_BINARY_APPLY(_, DIVNONAN)      \
  MUDNN_BINARY_APPLY(_, SQUARED_DIFF)

class Binary : public Descriptor<
                   mudnnBinaryStruct,
                   mudnnCreateBinaryDescriptor,
                   mudnnDestroyBinaryDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_BINARY_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& left,
      const muTensor& right) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Binary;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Binary;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetBinary(Binary& op, Binary::Mode mode);

void SetBinary(Binary& op, Binary::Mode mode, int64_t alpha);
void SetBinary(Binary& op, Binary::Mode mode, double alpha);

void SetBinary(Binary& op, Binary::Mode mode, int64_t alpha, int64_t beta);
void SetBinary(Binary& op, Binary::Mode mode, double alpha, double beta);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_BINARY_H_
