#include <ATen/Dispatch.h>
#include <ATen/OpMathType.h>
#include <ATen/TensorMeta.h>
#include <ATen/ops/empty.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

#include <mudnn.h>

namespace at {
namespace musa {

Tensor& BaddbmmOutImpl(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out,
    bool inplace = false) {
  // for pytorch operator:
  //   out = alpha * (batch1 @ batch2) + beta * self
  // for muDNN BatchMatMul operator:
  //   out = alpha * (batch1 @ batch2) + beta * out + gamma * self
  TORCH_CHECK(
      self.scalar_type() == batch1.scalar_type() &&
          batch1.scalar_type() == batch2.scalar_type(),
      "self, batch1 and batch2 should have the same dtype");
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half,
      "BaaddBmm only supports fp16/fp32 dtype");
  TORCH_CHECK(
      batch1.dim() == batch2.dim() && batch1.dim() == out.dim() &&
          out.dim() == 3,
      "baddbmm requires batch1 and batch2 and out to be 3-d tensor");
  TORCH_CHECK(batch1.size(2) == batch2.size(1), "reduce dim shapes unmatched");
  TORCH_CHECK(
      out.size(1) == batch1.size(1) && out.size(2) == batch2.size(2),
      "Wrong inputs shape for baddbmm");

  c10::musa::MUSAGuard device_guard(self.device());
  if (out.numel() == 0) {
    return out; // just return
  }
  if (batch1.numel() == 0 || batch2.numel() == 0) {
    if (inplace) {
      out.mul_(beta);
    } else {
      out.zero_();
      out.add_(self, beta);
    }
    return out;
  }

  muHandle& h = GetMudnnHandle();

  bool trans_b1 = IsTranspose(batch1);
  bool trans_b2 = IsTranspose(batch2);
  double alpha_ = alpha.to<double>();
  double beta_ = beta.to<double>();

  ::musa::dnn::BatchMatMul bmm;
  CHECK_MUDNN_STATUS(
      bmm.SetComputeMode(at::musa::GetComputeModeFromCtx(out.scalar_type())),
      "SetComputeMode");
  CHECK_MUDNN_STATUS(bmm.SetTranspose(trans_b1, trans_b2), "SetTranspose");
  CHECK_MUDNN_STATUS(bmm.SetAlpha(alpha_), "SetAlpha");
  CHECK_MUDNN_STATUS(bmm.SetGamma(beta_), "SetGamma");

  Tensor batch1_contig;
  Tensor batch2_contig;
  auto batch1_m = trans_b1
      ? CreateMUTensor(batch1.transpose(-2, -1))
      : CreateMUTensor(ContiguousRef(batch1, batch1_contig));
  auto batch2_m = trans_b2
      ? CreateMUTensor(batch2.transpose(-2, -1))
      : CreateMUTensor(ContiguousRef(batch2, batch2_contig));
  auto out_m = CreateMUTensor(out);

  if (self.dim() == 1) {
    TORCH_INTERNAL_ASSERT(
        self.size(0) == batch2.size(2), "self cannot be broadcasted");

    Tensor self_contig = self.contiguous();
    auto self_m = CreateMUTensor(self_contig);
    CHECK_MUDNN_STATUS(
        bmm.RunWithBiasAdd(
            h, out_m, batch1_m, batch2_m, self_m, InternalMemAlloc),
        "RunWithBiasAdd");
  } else {
    TORCH_INTERNAL_ASSERT(
        self.size(-1) == batch2.size(2), "self cannot be broadcasted");
    // FIXME:(lms/fan.mo) Mudnn may have fixed the issue when bias has
    // multi-dimensions we should check it later and avoid self.clone.
    auto safe_self = self;
    if (inplace) {
      safe_self = self.clone(
          at::MemoryFormat::Contiguous); // Clone the self to make result right
                                         // at the `out.add_`.
    }
    CHECK_MUDNN_STATUS(
        bmm.Run(h, out_m, batch1_m, batch2_m, InternalMemAlloc), "Run");
    out.add_(safe_self, beta);
  }

  return out;
}

Tensor& BaddbmmOut(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha,
    Tensor& out) {
  return BaddbmmOutImpl(self, batch1, batch2, beta, alpha, out);
}

Tensor Baddbmm(
    const Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(
      self.scalar_type() == batch1.scalar_type() &&
          batch1.scalar_type() == batch2.scalar_type(),
      "self, batch1 and batch2 should have the same dtype");
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half,
      "BaaddBmm only supports fp16/fp32 dtype");
  TORCH_CHECK(
      batch1.dim() == batch2.dim() && batch1.dim() == 3,
      "baddbmm requires batch1 and batch2 to be 3-d tensor");
  TORCH_CHECK(batch1.size(2) == batch2.size(1), "reduce dim shapes unmatched");

  int batch = batch1.size(0);
  int m = batch1.size(1);
  int n = batch2.size(2);
  Tensor output =
      at::empty({batch, m, n}, batch1.options(), at::MemoryFormat::Contiguous);
  if (self.dim() == 1) {
    TORCH_INTERNAL_ASSERT(self.size(0) == n, "self cannot be broadcasted");
  }
  return BaddbmmOutImpl(self, batch1, batch2, beta, alpha, output);
}

Tensor& Baddbmm_(
    Tensor& self,
    const Tensor& batch1,
    const Tensor& batch2,
    const Scalar& beta,
    const Scalar& alpha) {
  TORCH_CHECK(
      self.scalar_type() == batch1.scalar_type() &&
          batch1.scalar_type() == batch2.scalar_type(),
      "self, batch1 and batch2 should have the same dtype");
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Float ||
          self.scalar_type() == at::ScalarType::Half,
      "BaaddBmm only supports fp16/fp32 dtype");
  TORCH_CHECK(
      batch1.dim() == batch2.dim() && batch1.dim() == 3,
      "baddbmm requires batch1 and batch2 to be 3-d tensor");
  TORCH_CHECK(batch1.size(2) == batch2.size(1), "reduce dim shapes unmatched");
  if (self.dim() == 1) {
    TORCH_INTERNAL_ASSERT(
        self.size(0) == batch2.size(2), "self cannot be broadcasted");
  }

  BaddbmmOutImpl(self, batch1, batch2, beta, alpha, self, true);
  return self;
}

} // namespace musa
} // namespace at
