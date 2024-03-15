#include <ATen/Config.h>
#include <ATen/Tensor.h>
#include <ATen/native/SpectralOpsUtils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_fft_c2c_cpu_dispatch.h>
#include <ATen/ops/_fft_r2c_cpu_dispatch.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/pad.h>
#include <ATen/ops/view_as_complex.h>
#include <ATen/ops/view_as_real.h>
#include <ATen/ops/zeros.h>
#endif

#include <torch/library.h>

#include "ComplexHelper.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {
namespace musa {

namespace {

template <typename Stream, typename T>
static Stream& WriteOpt(Stream& s, const optional<T>& value) {
  if (value) {
    s << *value;
  } else {
    s << "None";
  }
  return s;
}

Tensor CpuAwareFFTReal2Complex(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  const OptionalDeviceGuard device_guard(self.device());

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  cpu_self.copy_(self);

  Tensor cpu_out = at::cpu::_fft_r2c(cpu_self, dim, normalization, onesided);

  return cpu_out;
}

Tensor CpuAwareFFTComplex2Complex(
    const Tensor& self,
    IntArrayRef dim,
    int64_t normalization,
    bool onesided) {
  const OptionalDeviceGuard device_guard(self.device());

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  cpu_self.copy_(self);

  Tensor cpu_out = at::cpu::_fft_c2c(cpu_self, dim, normalization, onesided);

  return cpu_out;
}

} // anonymous namespace

Tensor StftCenter(
    const Tensor& self,
    const int64_t n_fft,
    const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt,
    const c10::optional<Tensor>& window_opt,
    const bool center,
    c10::string_view mode,
    const bool normalized,
    const optional<bool> onesidedOpt,
    const optional<bool> return_complexOpt) {
  c10::MaybeOwned<Tensor> window_maybe_owned =
      at::borrow_from_optional_tensor(window_opt);
  const Tensor& window = *window_maybe_owned;

#define REPR(SS)                                                          \
  SS << "stft(" << self.toString() << self.sizes() << ", n_fft=" << n_fft \
     << ", hop_length=" << hop_length << ", win_length=" << win_length    \
     << ", window=";                                                      \
  if (window.defined()) {                                                 \
    SS << window.toString() << "{" << window.sizes() << "}";              \
  } else {                                                                \
    SS << "None";                                                         \
  }                                                                       \
  SS << ", normalized=" << normalized << ", onesided=";                   \
  WriteOpt(SS, onesidedOpt) << ", return_complex=";                       \
  WriteOpt(SS, return_complexOpt) << ") "

  TORCH_CHECK(
      !window.defined() || window.device() == self.device(),
      "stft input and window must be on the same device but got self on ",
      self.device(),
      " and window on ",
      window.device())

  // default_init hop_length and win_length
  auto hop_length = hop_lengthOpt.value_or(n_fft >> 2);
  auto win_length = win_lengthOpt.value_or(n_fft);
  const bool return_complex = return_complexOpt.value_or(
      self.is_complex() || (window.defined() && window.is_complex()));
  if (!return_complex) {
    TORCH_CHECK(
        return_complexOpt.has_value(),
        "stft requires the return_complex parameter be given for real inputs, "
        "and will further require that return_complex=True in a future "
        "PyTorch release.");

    TORCH_WARN_ONCE(
        "stft with return_complex=False is deprecated. In a future pytorch "
        "release, stft will return complex tensors for all inputs, and "
        "return_complex=False will raise an error.\n"
        "Note: you can still call torch.view_as_real on the complex output to "
        "recover the old return format.");
  }

  if (!at::isFloatingType(self.scalar_type()) &&
      !at::isComplexType(self.scalar_type())) {
    std::ostringstream ss;
    REPR(ss) << ": expected a tensor of floating point or complex values";
    AT_ERROR(ss.str());
  }
  if (self.dim() > 2 || self.dim() < 1) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D or 2D tensor";
    AT_ERROR(ss.str());
  }
  Tensor input = self;
  if (self.dim() == 1) {
    input = input.unsqueeze(0);
  }
  if (center) {
    const auto input_shape = input.sizes();
    const auto input_dim = input_shape.size();
    const auto extra_dims = std::max(size_t{3}, input_dim) - input_dim;
    const auto pad_amount = n_fft / 2;

    DimVector extended_shape(extra_dims, 1);
    extended_shape.append(input_shape.begin(), input_shape.end());
    input = at::pad(input.view(extended_shape), {pad_amount, pad_amount}, mode);
    input = input.view(IntArrayRef(input.sizes()).slice(extra_dims));
  }
  int64_t batch = input.size(0);
  int64_t len = input.size(1);
  if (n_fft <= 0 || n_fft > len) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < n_fft < " << len
             << ", but got n_fft=" << win_length;
    AT_ERROR(ss.str());
  }
  if (hop_length <= 0) {
    std::ostringstream ss;
    REPR(ss) << ": expected hop_length > 0, but got hop_length=" << hop_length;
    AT_ERROR(ss.str());
  }
  if (win_length <= 0 || win_length > n_fft) {
    std::ostringstream ss;
    REPR(ss) << ": expected 0 < win_length <= n_fft, but got win_length="
             << win_length;
    AT_ERROR(ss.str());
  }
  if (window.defined() && (window.dim() != 1 || window.size(0) != win_length)) {
    std::ostringstream ss;
    REPR(ss) << ": expected a 1D window tensor of size equal to win_length="
             << win_length << ", but got window with size " << window.sizes();
    AT_ERROR(ss.str());
  }
#undef REPR
  auto window_ = window;
  if (win_length < n_fft) {
    // pad center
    auto left = (n_fft - win_length) / 2;
    if (window.defined()) {
      window_ = at::zeros({n_fft}, window.options());
      window_.narrow(0, left, win_length).copy_(window);
    } else {
      window_ = at::zeros({n_fft}, self.options());
      window_.narrow(0, left, win_length).fill_(1);
    }
  }
  int64_t n_frames = 1 + (len - n_fft) / hop_length;

  // time2col
  input = input.as_strided(
      {batch, n_frames, n_fft},
      {input.stride(0), hop_length * input.stride(1), input.stride(1)});
  if (window_.defined()) {
    input = input.mul(window_);
  }

  // FFT and transpose to get (batch x fft_size x num_frames)
  const bool complex_fft = input.is_complex();
  const auto onesided = onesidedOpt.value_or(!complex_fft);

  const at::native::fft_norm_mode norm = normalized
      ? at::native::fft_norm_mode::by_root_n
      : at::native::fft_norm_mode::none;
  Tensor out_cpu;

  if (complex_fft) {
    TORCH_CHECK(
        !onesided, "Cannot have onesided output if window or input is complex");
    out_cpu = CpuAwareFFTComplex2Complex(
        input,
        input.dim() - 1,
        static_cast<int64_t>(norm),
        /*forward=*/true);
  } else {
    out_cpu = CpuAwareFFTReal2Complex(
        input, input.dim() - 1, static_cast<int64_t>(norm), onesided);
  }

  out_cpu.transpose_(1, 2);
  Tensor out = at::view_as_real(out_cpu).to(self.device()).contiguous();

  if (self.dim() == 1) {
    out.squeeze_(0);
  }

  if (return_complex) {
    return at::view_as_complex(out);
  } else {
    return out;
  }
}

Tensor Stft(
    const Tensor& self,
    const int64_t n_fft,
    const optional<int64_t> hop_lengthOpt,
    const optional<int64_t> win_lengthOpt,
    const c10::optional<Tensor>& window_opt,
    const bool normalized,
    const optional<bool> onesidedOpt,
    const optional<bool> return_complexOpt) {
  return StftCenter(
      self,
      n_fft,
      hop_lengthOpt,
      win_lengthOpt,
      window_opt,
      /*center=*/false,
      /*mode=*/"constant",
      normalized,
      onesidedOpt,
      return_complexOpt);
}

ADVANCED_REGISTER(aten, PrivateUse1, "stft", Stft)
ADVANCED_REGISTER(aten, PrivateUse1, "stft.center", StftCenter)

} // namespace musa
} // namespace at