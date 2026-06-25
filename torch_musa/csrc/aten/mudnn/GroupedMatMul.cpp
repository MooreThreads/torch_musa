#include "torch_musa/csrc/aten/mudnn/GroupedMatMul.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetGroupedMatMulImpl(
    GroupedMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  CHECK_MUDNN_STATUS(
      mudnnSetGroupedMatMulDescriptorEx(
          op.Desc(), mode, transa, transb, false, 0, alpha, beta, gamma),
      "mudnnSetGroupedMatMulDescriptorEx");
}

#else

void SetGroupedMatMulImpl(
    GroupedMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  CHECK_MUDNN_STATUS(op.SetComputeMode(mode), "SetComputeMode");
  CHECK_MUDNN_STATUS(op.SetTranspose(transa, transb), "SetTranspose");
  CHECK_MUDNN_STATUS(op.SetAlpha(alpha), "SetAlpha");
  CHECK_MUDNN_STATUS(op.SetBeta(beta), "SetBeta");
  CHECK_MUDNN_STATUS(op.SetGamma(gamma), "SetGamma");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t GroupedMatMul::RunLt(
    muHandle& h,
    muTensor* d,
    const muTensor* a,
    const muTensor* b,
    const muTensor* c,
    const muTensor* bias,
    const MatMulLtParam* params,
    int groups,
    const MemoryMaintainer& maintainer) const {
  std::vector<mudnnTensorDescriptor_t> d_descs(groups);
  std::vector<void*> d_ptrs(groups);
  std::vector<mudnnTensorDescriptor_t> a_descs(groups);
  std::vector<const void*> a_ptrs(groups);
  std::vector<mudnnTensorDescriptor_t> b_descs(groups);
  std::vector<const void*> b_ptrs(groups);
  std::vector<mudnnTensorDescriptor_t> c_descs(groups);
  std::vector<const void*> c_ptrs(groups);
  std::vector<mudnnTensorDescriptor_t> bias_descs(groups);
  std::vector<const void*> bias_ptrs(groups);

  // Create per-group MatMulLtParam descriptors
  std::vector<mudnnMatMulLtParamDescriptor_t> lt_descs(groups);
  for (int i = 0; i < groups; ++i) {
    mudnnCreateMatMulLtParamDescriptor(&lt_descs[i]);
  }

  for (int i = 0; i < groups; ++i) {
    d[i].Build();
    a[i].Build();
    b[i].Build();
    c[i].Build();
    bias[i].Build();
    params[i].Build();

    d_descs[i] = d[i].Desc();
    d_ptrs[i] = const_cast<void*>(d[i].DataPtr());
    a_descs[i] = a[i].Desc();
    a_ptrs[i] = a[i].DataPtr();
    b_descs[i] = b[i].Desc();
    b_ptrs[i] = b[i].DataPtr();
    c_descs[i] = c[i].Desc();
    c_ptrs[i] = c[i].DataPtr();
    bias_descs[i] = bias[i].Desc();
    bias_ptrs[i] = bias[i].DataPtr();

    // Populate lt descriptor with scale/amax info
    mudnnSetMatMulLtParamScale(
        lt_descs[i],
        MUDNN_UNPACK_TENSOR(params[i].scale_a),
        MUDNN_UNPACK_TENSOR(params[i].scale_b),
        MUDNN_UNPACK_TENSOR(params[i].scale_c),
        MUDNN_UNPACK_TENSOR(params[i].scale_d),
        0,
        false);
    if (params[i].amax_d.DataPtr()) {
      mudnnSetMatMulLtParamAmaxD(
          lt_descs[i],
          params[i].amax_d.Desc(),
          const_cast<void*>(params[i].amax_d.DataPtr()));
    }
  }

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGroupedMatmulGetWorkspaceSizeLt(
          h,
          Desc(),
          d_descs.data(),
          a_descs.data(),
          b_descs.data(),
          c_descs.data(),
          lt_descs.data(),
          groups,
          &ws_size),
      "mudnnGroupedMatmulGetWorkspaceSizeLt");

  auto ws = maintainer(ws_size);
  auto status = mudnnGroupedMatMulRunLt(
      h,
      Desc(),
      d_descs.data(),
      d_ptrs.data(),
      a_descs.data(),
      a_ptrs.data(),
      b_descs.data(),
      b_ptrs.data(),
      c_descs.data(),
      c_ptrs.data(),
      bias_descs.data(),
      bias_ptrs.data(),
      lt_descs.data(),
      groups,
      ws.get(),
      ws_size);

  for (int i = 0; i < groups; ++i) {
    mudnnDestroyMatMulLtParamDescriptor(lt_descs[i]);
  }
  return status;
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetGroupedMatMul(
    GroupedMatMul& op,
    ComputeMode mode,
    bool transa,
    bool transb,
    double alpha,
    double beta,
    double gamma) {
  return SetGroupedMatMulImpl(op, mode, transa, transb, alpha, beta, gamma);
}

} // namespace at::musa
