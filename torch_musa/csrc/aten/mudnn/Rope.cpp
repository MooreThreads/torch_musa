#include "torch_musa/csrc/aten/mudnn/Rope.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

namespace at::musa {

namespace {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

void SetRopeImpl(
    ::musa::dnn::Rope& op,
    bool rotary_interleaved,
    bool batch_first,
    bool multi_latent_attention) {
  CHECK_MUDNN_STATUS(
      mudnnSetRopeDescriptor(
          op.Desc(), rotary_interleaved, batch_first, multi_latent_attention),
      "mudnnSetRopeDescriptor");
}

#else

void SetRopeImpl(
    ::musa::dnn::Rope& op,
    bool rotary_interleaved,
    bool batch_first,
    bool multi_latent_attention) {
  CHECK_MUDNN_STATUS(
      op.SetRotaryInterleaved(rotary_interleaved), "SetRotaryInterleaved");
  CHECK_MUDNN_STATUS(op.SetBatchFirst(batch_first), "SetBatchFirst");
  CHECK_MUDNN_STATUS(
      op.SetMultiLatentAttention(multi_latent_attention),
      "SetMultiLatentAttention");
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

mudnnStatus_t MudnnRope::Run(
    muHandle& h,
    muTensor& output,
    const muTensor& input,
    const muTensor& freqs_cis) const {
  output.Build();
  input.Build();
  freqs_cis.Build();
  return mudnnRopeForward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(output),
      MUDNN_UNPACK_TENSOR(input),
      MUDNN_UNPACK_TENSOR(freqs_cis));
}

mudnnStatus_t MudnnRope::RunBwd(
    muHandle& h,
    muTensor& grad_input,
    const muTensor& grad_output,
    const muTensor& freqs_cis) const {
  grad_input.Build();
  grad_output.Build();
  freqs_cis.Build();
  return mudnnRopeBackward(
      h,
      Desc(),
      MUDNN_UNPACK_TENSOR(grad_input),
      MUDNN_UNPACK_TENSOR(grad_output),
      MUDNN_UNPACK_TENSOR(freqs_cis));
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

void SetRope(
    ::musa::dnn::Rope& op,
    bool rotary_interleaved,
    bool batch_first,
    bool multi_latent_attention) {
  return SetRopeImpl(
      op, rotary_interleaved, batch_first, multi_latent_attention);
}

} // namespace at::musa
