#include <ATen/Config.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Pool.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include <mudnn.h>

namespace at {

namespace musa {

Tensor Stft(
    const Tensor& self,
    const int64_t n_fft,
    const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt,
    const c10::optional<Tensor>& window_opt,
    const bool normalized,
    const optional<bool> onesidedOpt,
    const optional<bool> return_complexOpt) {
  auto self_cpu = self.cpu();
  Tensor window_opt_cpu;
  if (window_opt.has_value()) {
    window_opt_cpu = window_opt.value().cpu();
  }
  auto res = at::native::stft(
                 self_cpu,
                 n_fft,
                 hop_lengthOpt,
                 win_lengthOpt,
                 window_opt_cpu,
                 /*center=*/false,
                 /*mode=*/"constant",
                 normalized,
                 onesidedOpt,
                 return_complexOpt)
                 .to("musa");

  return res;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("stft", &Stft);
}

} // namespace musa
} // namespace at