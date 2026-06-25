#include "torch_musa/csrc/aten/mudnn/Loss.h"

#include <torch_musa/csrc/aten/mudnn/Exception.h>
#include "torch_musa/csrc/aten/mudnn/Param.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

template <typename MODE>
mudnnLossReduceMode_t GetLossModeFromEnum(MODE mode) {
#define CASE(MODE1, MODE2) \
  case MODE::MODE1: {      \
    return MODE2;          \
  }

  switch (mode) {
    MUDNN_FORALL_LOSS_OPS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

void SetNLLLossImpl(NLLLoss& op, NLLLoss::Mode mode, int ignore_index) {
  const auto m = GetLossModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetNLLLossDescriptor(op.Desc(), m, ignore_index),
      "mudnnSetNLLLossDescriptor");
}

void SetKLDivLossImpl(KLDivLoss& op, KLDivLoss::Mode mode, bool log_target) {
  const auto m = GetLossModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetKLDivLossDescriptor(op.Desc(), m, log_target),
      "mudnnSetKLDivLossDescriptor");
}

void SetCrossEntropyLossImpl(
    CrossEntropyLoss& op,
    CrossEntropyLoss::ReductionMode mode,
    float label_smoothing,
    int64_t ignore_index) {
  const auto m = GetLossModeFromEnum(mode);
  CHECK_MUDNN_STATUS(
      mudnnSetCrossEntropyLossDescriptor(
          op.Desc(), m, ignore_index, label_smoothing),
      "mudnnSetCrossEntropyLossDescriptor");
}

#else

void SetNLLLossImpl(NLLLoss& op, NLLLoss::Mode mode, int ignore_index) {
  CHECK_MUDNN_STATUS(op.SetReductionMode(mode), "SetReductionMode");
  CHECK_MUDNN_STATUS(op.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
}

void SetKLDivLossImpl(KLDivLoss& op, KLDivLoss::Mode mode, bool log_target) {
  CHECK_MUDNN_STATUS(op.SetReductionMode(mode), "SetReductionMode");
  CHECK_MUDNN_STATUS(op.SetLogTarget(log_target), "SetReductionMode");
}

void SetCrossEntropyLossImpl(
    CrossEntropyLoss& op,
    CrossEntropyLoss::ReductionMode mode,
    float label_smoothing,
    int64_t ignore_index) {
  CHECK_MUDNN_STATUS(op.SetReductionMode(mode), "SetReductionMode");
  CHECK_MUDNN_STATUS(
      op.SetLabelSmoothing(label_smoothing), "SetLabelSmoothing");
  CHECK_MUDNN_STATUS(op.SetIgnoreIndex(ignore_index), "SetIgnoreIndex");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t NLLLoss::Run(
    muHandle& h,
    muTensor& out,
    muTensor& total_weight,
    const muTensor& in,
    const muTensor& target,
    const muTensor& weight,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  total_weight.Build();
  in.Build();
  target.Build();
  weight.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetNLLLossForwardWorkspaceSize(
          h,
          Desc(),
          in.Desc(),
          target.Desc(),
          weight.Desc(),
          out.Desc(),
          total_weight.Desc(),
          &ws_size),
      "mudnnGetNLLLossForwardWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnNLLLossForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(weight),
      MUDNN_UNPACK_TENSOR(out),
      MUDNN_UNPACK_TENSOR(total_weight),
      mem.get(),
      ws_size);
}

mudnnStatus_t NLLLoss::RunBwd(
    muHandle& h,
    muTensor& grad_input,
    const muTensor& grad_output,
    const muTensor& target,
    const muTensor& weight,
    const muTensor& total_weight,
    const MemoryMaintainer& maintainer) const {
  grad_input.Build();
  grad_output.Build();
  target.Build();
  weight.Build();
  total_weight.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetNLLLossBackwardWorkspaceSize(
          h,
          Desc(),
          grad_output.Desc(),
          target.Desc(),
          weight.Desc(),
          total_weight.Desc(),
          grad_input.Desc(),
          &ws_size),
      "mudnnGetNLLLossBackwardWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnNLLLossBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(grad_output),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(weight),
      MUDNN_UNPACK_TENSOR(total_weight),
      MUDNN_UNPACK_TENSOR(grad_input),
      mem.get(),
      ws_size);
}

mudnnStatus_t KLDivLoss::Run(
    muHandle& h,
    muTensor& out,
    muTensor& in,
    const muTensor& target,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  in.Build();
  target.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetKLDivLossForwardWorkspaceSize(
          h, Desc(), in.Desc(), target.Desc(), out.Desc(), &ws_size),
      "mudnnGetKLDivLossForwardWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnKLDivLossForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(target),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t KLDivLoss::RunBwd(
    muHandle& h,
    muTensor& grad_input,
    const muTensor& grad_output,
    const muTensor& input,
    const muTensor& target) const {
  grad_input.Build();
  grad_output.Build();
  input.Build();
  target.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetKLDivLossBackwardWorkspaceSize(
          h,
          Desc(),
          grad_output.Desc(),
          input.Desc(),
          target.Desc(),
          grad_input.Desc(),
          &ws_size),
      "mudnnGetKLDivLossBackwardWorkspaceSize");
  auto mem = InternalMemAlloc(ws_size);
  return mudnnKLDivLossBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(grad_output),
      MUDNN_UNPACK_TENSOR(input),
      MUDNN_UNPACK_TENSOR(target),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(grad_input));
}

mudnnStatus_t CrossEntropyLoss::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& in,
    const muTensor& target,
    const muTensor& weight,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  in.Build();
  target.Build();
  weight.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetCrossEntropyLossForwardWorkspaceSize(
          h,
          Desc(),
          in.Desc(),
          target.Desc(),
          weight.Desc(),
          out.Desc(),
          &ws_size),
      "mudnnGetCrossEntropyLossForwardWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnCrossEntropyLossForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(weight),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t CrossEntropyLoss::RunBwd(
    muHandle& h,
    muTensor& grad_input,
    const muTensor& input,
    const muTensor& grad_output,
    const muTensor& target,
    const muTensor& weight,
    const MemoryMaintainer& maintainer) const {
  grad_input.Build();
  input.Build();
  grad_output.Build();
  target.Build();
  weight.Build();
  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetCrossEntropyLossBackwardWorkspaceSize(
          h,
          Desc(),
          input.Desc(),
          grad_output.Desc(),
          target.Desc(),
          weight.Desc(),
          grad_input.Desc(),
          &ws_size),
      "mudnnGetCrossEntropyLossBackwardWorkspaceSize");
  auto mem = maintainer(ws_size);
  return mudnnCrossEntropyLossBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(input),
      MUDNN_UNPACK_TENSOR(grad_output),
      MUDNN_UNPACK_TENSOR(target),
      MUDNN_UNPACK_TENSOR(weight),
      mem.get(),
      ws_size,
      MUDNN_UNPACK_TENSOR(grad_input));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetNLLLoss(NLLLoss& op, NLLLoss::Mode mode, int ignore_index) {
  return SetNLLLossImpl(op, mode, ignore_index);
}

void SetKLDivLoss(KLDivLoss& op, KLDivLoss::Mode mode, bool log_target) {
  return SetKLDivLossImpl(op, mode, log_target);
}

void SetCrossEntropyLoss(
    CrossEntropyLoss& op,
    CrossEntropyLoss::ReductionMode mode,
    float label_smoothing,
    int64_t ignore_index) {
  return SetCrossEntropyLossImpl(op, mode, label_smoothing, ignore_index);
}

} // namespace at::musa
