#include <torch_musa/csrc/aten/mudnn/CTCLoss.h>

#include <torch_musa/csrc/aten/mudnn/Exception.h>

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetCTCLossImpl(
    CTCLoss& op,
    int64_t blank,
    bool zero_infinity,
    int64_t max_target_length) {
  CHECK_MUDNN_STATUS(
      mudnnSetCTCLossDescriptor_v10(
          op.Desc(), blank, zero_infinity, max_target_length),
      "mudnnSetCTCLossDescriptor_v10");
}

#else

void SetCTCLossImpl(
    CTCLoss& op,
    int64_t blank,
    bool zero_infinity,
    int64_t max_target_length) {
  CHECK_MUDNN_STATUS(op.SetBlank(blank), "SetBlank");
  CHECK_MUDNN_STATUS(op.SetZeroInfinity(zero_infinity), "SetZeroInfinity");
  CHECK_MUDNN_STATUS(
      op.SetMaxTargetLength(max_target_length), "SetMaxTargetLength");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t CTCLoss::Run(
    muHandle& h,
    muTensor& out,
    muTensor& alpha,
    const muTensor& input,
    const muTensor& target,
    const muTensor& input_lengths,
    const muTensor& target_lengths) const {
  out.Build();
  alpha.Build();
  input.Build();
  target.Build();
  input_lengths.Build();
  target_lengths.Build();
  return mudnnCTCLoss_v10(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(alpha),
      MUDNN_UNPACK_TENSOR(input),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(input_lengths),
      MUDNN_UNPACK_TENSOR(target_lengths));
}

mudnnStatus_t CTCLoss::RunBwd(
    muHandle& h,
    muTensor& grad_input,
    const muTensor& grad_out,
    const muTensor& input,
    const muTensor& target,
    const muTensor& input_lengths,
    const muTensor& target_lengths,
    const muTensor& out,
    const muTensor& alpha,
    const MemoryMaintainer& maintainer) const {
  grad_input.Build();
  grad_out.Build();
  out.Build();
  alpha.Build();
  input.Build();
  target.Build();
  input_lengths.Build();
  target_lengths.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnCTCLossGetWorkspaceSize_v10(
          h,
          Desc(),
          grad_input.Desc(),
          grad_out.Desc(),
          input.Desc(),
          target.Desc(),
          input_lengths.Desc(),
          target_lengths.Desc(),
          out.Desc(),
          alpha.Desc(),
          &ws_size),
      "mudnnCTCLossGetWorkspaceSize_v10");
  auto mem = maintainer(ws_size);
  return mudnnCTCLossBackward_v10(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(grad_input),
      MUDNN_UNPACK_TENSOR(grad_out),
      MUDNN_UNPACK_TENSOR(input),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(input_lengths),
      MUDNN_UNPACK_TENSOR(target_lengths),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(alpha),
      mem.get(),
      ws_size);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetCTCLoss(
    CTCLoss& op,
    int64_t blank,
    bool zero_infinity,
    int64_t max_target_length) {
  SetCTCLossImpl(op, blank, zero_infinity, max_target_length);
}

} // namespace at::musa
