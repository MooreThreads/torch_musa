#include <ATen/Config.h>
#include <ATen/native/ReduceOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/dot_native.h>
#endif

#include "torch_musa/csrc/aten/mudnn/Dot.h"
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

at::Tensor& DotOut(const at::Tensor& l, const at::Tensor& r, at::Tensor& out) {
  TORCH_CHECK(l.sizes() == r.sizes(), "dot tensors' shape don't match");
  TORCH_CHECK(
      l.dim() == r.dim() && l.dim() == 1, "dot inputs must be 1-D tensors");
  const c10::musa::MUSAGuard device_guard(l.device());

  if (l.numel() == 0 || r.numel() == 0) {
    out.zero_().squeeze_();
    return out;
  }

  muHandle& h = GetMudnnHandle();

  Tensor contiguous_l = l.contiguous();
  Tensor contiguous_r = r.contiguous();

  auto rst = CreateMUTensor(out);
  auto lmt = CreateMUTensor(contiguous_l);
  auto rmt = CreateMUTensor(contiguous_r);

  ::musa::dnn::Dot op;
  CHECK_MUDNN_STATUS(
      op.SetComputeMode(GetMatmulComputeModeFromCtx(l.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(op.Run(h, rst, lmt, rmt, InternalMemAlloc), "Run");

  out.squeeze_();
  return out;
}

} // namespace at::musa
