#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_UNARY_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_UNARY_H_

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"

namespace at::musa {

#define MUDNN_UNARY_APPLY(_, MODE) _(MODE, MUDNN_UNARY_##MODE)

#define MUDNN_FORALL_UNARY_OPS(_)            \
  MUDNN_UNARY_APPLY(_, ADD)                  \
  MUDNN_UNARY_APPLY(_, SUB)                  \
  MUDNN_UNARY_APPLY(_, MUL)                  \
  MUDNN_UNARY_APPLY(_, DIV)                  \
  MUDNN_UNARY_APPLY(_, POW)                  \
  MUDNN_UNARY_APPLY(_, SQRT)                 \
  MUDNN_UNARY_APPLY(_, ROUND)                \
  MUDNN_UNARY_APPLY(_, RSQRT)                \
  MUDNN_UNARY_APPLY(_, RECIPROCAL)           \
  MUDNN_UNARY_APPLY(_, SQUARE)               \
  MUDNN_UNARY_APPLY(_, IS_FINITE)            \
  MUDNN_UNARY_APPLY(_, IS_INF)               \
  MUDNN_UNARY_APPLY(_, IS_NAN)               \
  MUDNN_UNARY_APPLY(_, IS_NONZERO)           \
  MUDNN_UNARY_APPLY(_, LOGICAL_AND)          \
  MUDNN_UNARY_APPLY(_, LOGICAL_OR)           \
  MUDNN_UNARY_APPLY(_, LOGICAL_XOR)          \
  MUDNN_UNARY_APPLY(_, EXP)                  \
  MUDNN_UNARY_APPLY(_, LOG)                  \
  MUDNN_UNARY_APPLY(_, LOG10)                \
  MUDNN_UNARY_APPLY(_, LOG2)                 \
  MUDNN_UNARY_APPLY(_, LOG1P)                \
  MUDNN_UNARY_APPLY(_, SIN)                  \
  MUDNN_UNARY_APPLY(_, COS)                  \
  MUDNN_UNARY_APPLY(_, ACOS)                 \
  MUDNN_UNARY_APPLY(_, TAN)                  \
  MUDNN_UNARY_APPLY(_, ATAN)                 \
  MUDNN_UNARY_APPLY(_, LT)                   \
  MUDNN_UNARY_APPLY(_, LE)                   \
  MUDNN_UNARY_APPLY(_, GT)                   \
  MUDNN_UNARY_APPLY(_, GE)                   \
  MUDNN_UNARY_APPLY(_, EQ)                   \
  MUDNN_UNARY_APPLY(_, NE)                   \
  MUDNN_UNARY_APPLY(_, MAX)                  \
  MUDNN_UNARY_APPLY(_, MIN)                  \
  MUDNN_UNARY_APPLY(_, CEIL)                 \
  MUDNN_UNARY_APPLY(_, FLOOR)                \
  MUDNN_UNARY_APPLY(_, SIGMOID)              \
  MUDNN_UNARY_APPLY(_, HARDSIGMOID)          \
  MUDNN_UNARY_APPLY(_, RELU)                 \
  MUDNN_UNARY_APPLY(_, LEAKY_RELU)           \
  MUDNN_UNARY_APPLY(_, RELU6)                \
  MUDNN_UNARY_APPLY(_, TANH)                 \
  MUDNN_UNARY_APPLY(_, CLIP)                 \
  MUDNN_UNARY_APPLY(_, ELU)                  \
  MUDNN_UNARY_APPLY(_, SWISH)                \
  MUDNN_UNARY_APPLY(_, HARDSWISH)            \
  MUDNN_UNARY_APPLY(_, MISH)                 \
  MUDNN_UNARY_APPLY(_, SILU)                 \
  MUDNN_UNARY_APPLY(_, SOFTPLUS)             \
  MUDNN_UNARY_APPLY(_, GELU)                 \
  MUDNN_UNARY_APPLY(_, GELU_TANH)            \
  MUDNN_UNARY_APPLY(_, ABS)                  \
  MUDNN_UNARY_APPLY(_, ERF)                  \
  MUDNN_UNARY_APPLY(_, IDENTITY)             \
  MUDNN_UNARY_APPLY(_, TRUEDIV)              \
  MUDNN_UNARY_APPLY(_, FLOORDIV)             \
  MUDNN_UNARY_APPLY(_, TRUNCATEDIV)          \
  MUDNN_UNARY_APPLY(_, FLOORMOD)             \
  MUDNN_UNARY_APPLY(_, TRUNCATEMOD)          \
  MUDNN_UNARY_APPLY(_, SUB_BY_ALPHA)         \
  MUDNN_UNARY_APPLY(_, DIV_BY_ALPHA)         \
  MUDNN_UNARY_APPLY(_, TRUNCATEDIV_BY_ALPHA) \
  MUDNN_UNARY_APPLY(_, TRUEDIV_BY_ALPHA)     \
  MUDNN_UNARY_APPLY(_, FLOORDIV_BY_ALPHA)    \
  MUDNN_UNARY_APPLY(_, CAST)                 \
  MUDNN_UNARY_APPLY(_, SIGN)

class Unary : public Descriptor<
                  mudnnUnaryStruct,
                  mudnnCreateUnaryDescriptor,
                  mudnnDestroyUnaryDescriptor> {
 public:
  enum class Mode {
#define CASE(MODE, _2) MODE,
    MUDNN_FORALL_UNARY_OPS(CASE)
#undef CASE
  };

  mudnnStatus_t Run(muHandle& h, muTensor& out, const muTensor& in) const;
};

} // namespace at::musa

namespace musa::dnn {
using at::musa::Unary;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Unary;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetUnary(Unary& op, Unary::Mode mode);

void SetUnary(Unary& op, Unary::Mode mode, int64_t alpha);
void SetUnary(Unary& op, Unary::Mode mode, double alpha);

void SetUnary(Unary& op, Unary::Mode mode, int64_t alpha, int64_t beta);
void SetUnary(Unary& op, Unary::Mode mode, double alpha, double beta);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_UNARY_H_
