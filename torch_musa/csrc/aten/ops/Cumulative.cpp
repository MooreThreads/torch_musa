#include <ATen/Config.h>
#include <ATen/Dispatch.h>
#include <ATen/core/NamedTensor.h>
#include <ATen/native/ReduceOpsUtils.h>

#include <optional>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/cumprod_meta.h>
#include <ATen/ops/cumprod_native.h>
#include <ATen/ops/cumsum_meta.h>
#include <ATen/ops/cumsum_native.h>
#endif

#include "torch_musa/csrc/aten/mudnn/Cum.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

namespace {

using CUM_MODE = ::musa::dnn::Cum::Mode;

void CumulativeKernel(
    const Tensor& out,
    const Tensor& in,
    int64_t dim,
    CUM_MODE mode) {
  const c10::musa::MUSAGuard device_guard(in.device());
  auto in_ctype = in.scalar_type();
  auto out_ctype = out.scalar_type();

  AT_DISPATCH_ALL_TYPES_AND_COMPLEX_AND2(
      ScalarType::Half,
      ScalarType::BFloat16,
      out_ctype,
      "cumulative_musa",
      [&] {
        if (isFloatingType(in_ctype) || isFloatingType(out_ctype)) {
          out_ctype = promoteTypes(in_ctype, out_ctype);
          in_ctype = out_ctype;
        } else if (
            out_ctype == ScalarType::Int || out_ctype == ScalarType::Long) {
          if (in_ctype == ScalarType::Byte || in_ctype == ScalarType::Short) {
            in_ctype = ScalarType::Int;
          } else if (in_ctype == ScalarType::Long) {
            out_ctype = in_ctype;
          }
        } else {
          // Special cases for muDNN workarounds.
          if (in_ctype == ScalarType::Long) {
            out_ctype = in_ctype;
          } else {
            out_ctype = ScalarType::Int;
            if (in_ctype == ScalarType::Byte || in_ctype == ScalarType::Short) {
              in_ctype = out_ctype;
            }
          }
        }

        auto contig_c_in =
            FormatContiguous(in.to(in_ctype), MemoryFormat::Contiguous);
        auto contig_c_out =
            FormatContiguous(out.to(out_ctype), MemoryFormat::Contiguous);

        auto mudnn_out = CreateMUTensor(contig_c_out);
        auto mudnn_in = CreateMUTensor(contig_c_in);
        muHandle& h = GetMudnnHandle();
        ::musa::dnn::Cum cum;
        SetCum(cum, static_cast<int64_t>(dim), mode);
        CHECK_MUDNN_STATUS(
            cum.Run(h, mudnn_out, mudnn_in, InternalMemAlloc), "CumRun");

        if (!out.is_same(contig_c_out)) {
          out.copy_(contig_c_out);
        }
      });
}

template <CUM_MODE mode>
void CumulativeImpl(const Tensor& self, int64_t dim, const Tensor& result) {
  NoNamesGuard guard;
  if (self.dim() == 0) {
    result.fill_(self);
  } else if (self.numel() == 0) {
    result.zero_();
  } else {
    dim = maybe_wrap_dim(dim, self.dim());
    CumulativeKernel(result, self, dim, mode);
  }
}

} // anonymous namespace

TORCH_IMPL_FUNC(CumSumOut)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  CumulativeImpl<CUM_MODE::ADD>(self, dim, result);
}

TORCH_IMPL_FUNC(CumProdOut)
(const Tensor& self,
 int64_t dim,
 std::optional<ScalarType> dtype,
 const Tensor& result) {
  CumulativeImpl<CUM_MODE::MUL>(self, dim, result);
}

} // namespace at::musa
