#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHMATMUL_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHMATMUL_H_

#include <c10/core/ScalarType.h>

#include "torch_musa/csrc/aten/mudnn/Tensor.h"

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"

namespace at::musa {

struct MatMulLtParam final {
  muTensor scale_a;
  muTensor scale_b;
  muTensor scale_c;
  muTensor scale_d;
  muTensor amax_d;

  mudnnStatus_t SetScale(
      muTensor&& s_a,
      muTensor&& s_b,
      muTensor&& s_c,
      muTensor&& s_d) {
    scale_a = std::move(s_a);
    scale_b = std::move(s_b);
    scale_c = std::move(s_c);
    scale_d = std::move(s_d);
    return MUDNN_STATUS_SUCCESS;
  }

  mudnnStatus_t SetAmaxD(muTensor& a_d) {
    amax_d = std::move(a_d);
    return MUDNN_STATUS_SUCCESS;
  }

  void Build() const {
    scale_a.Build();
    scale_b.Build();
    scale_c.Build();
    scale_d.Build();
    amax_d.Build();
  }
};

class BatchMatMul : public Descriptor<
                        mudnnBatchMatMulStruct,
                        mudnnCreateBatchMatMulDescriptor,
                        mudnnDestroyBatchMatMulDescriptor> {
 public:
  // inplace: c = alpha * a * b + beta * c
  mudnnStatus_t Run(
      muHandle& h,
      muTensor& c,
      const muTensor& a,
      const muTensor& b,
      const MemoryMaintainer& maintainer = nullptr) const;

  // outplace: d = alpha * a * b + beta * c + gamma * bias
  mudnnStatus_t RunWithBiasAdd(
      muHandle& h,
      muTensor& d,
      const muTensor& a,
      const muTensor& b,
      const muTensor& c,
      const muTensor& bias,
      const MemoryMaintainer& maintainer = nullptr) const;

  // inplace: c = alpha * a * b + beta * c + gamma * bias
  mudnnStatus_t RunWithBiasAdd(
      muHandle& h,
      muTensor& c,
      const muTensor& a,
      const muTensor& b,
      const muTensor& bias,
      const MemoryMaintainer& maintainer = nullptr) const;

  mudnnStatus_t RunLt(
      muHandle& h,
      muTensor& d,
      const muTensor& a,
      const muTensor& b,
      const muTensor& c,
      const muTensor& bias,
      const MatMulLtParam& param,
      const MemoryMaintainer& maintainer = nullptr) const;
};

typedef BatchMatMul MatMul;

} // namespace at::musa

namespace musa::dnn {
typedef at::musa::BatchMatMul BatchMatMul;
typedef at::musa::MatMulLtParam MatMulLtParam;
typedef at::musa::MatMul MatMul;
} // namespace musa::dnn

#else

namespace at::musa {
typedef ::musa::dnn::BatchMatMul BatchMatMul;
typedef ::musa::dnn::MatMul MatMul;
typedef ::musa::dnn::MatMulLtParam MatMulLtParam;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {

void SetBatchMatMul(
    BatchMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb);

void SetBatchMatMul(
    BatchMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma);

void SetMatMul(MatMul& op, ComputeMode mode, bool transa, bool transb);

void SetMatMul(
    MatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_BATCHMATMUL_H_
