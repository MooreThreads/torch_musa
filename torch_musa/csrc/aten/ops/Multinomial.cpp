#include <ATen/TensorOperators.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/UnaryOps.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/_dirichlet_grad_native.h>
#include <ATen/ops/_sample_dirichlet_native.h>
#include <ATen/ops/_standard_gamma_grad_native.h>
#include <ATen/ops/_standard_gamma_native.h>
#include <ATen/ops/argmax.h>
#include <ATen/ops/bernoulli_native.h>
#include <ATen/ops/binomial_native.h>
#include <ATen/ops/cauchy_native.h>
#include <ATen/ops/div.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <ATen/ops/exponential_native.h>
#include <ATen/ops/geometric_native.h>
#include <ATen/ops/log_normal_native.h>
#include <ATen/ops/multinomial_native.h>
#include <ATen/ops/normal_native.h>
#include <ATen/ops/poisson_native.h>
#include <ATen/ops/random_native.h>
#include <ATen/ops/topk.h>
#include <ATen/ops/uniform_native.h>
#include <ATen/ops/zeros.h>
#endif

#include <float.h>
#include <math.h>
#include <torch/library.h>
#include <torch_musa/csrc/aten/ops/TensorFactory.h>

namespace at {

namespace native {

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

DEFINE_DISPATCH(multinomial_with_replacement_stub);

at::Tensor multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return at::native::multinomial_out(
      self, num_samples, replacement, generator, result);
}

at::Tensor& multinomial_out(
    const Tensor& self,
    int64_t n_sample,
    bool with_replacement,
    c10::optional<Generator> gen,
    Tensor& result) {
  TORCH_CHECK(
      result.device() == self.device(),
      "multinomial arguments must have the same device");
  TORCH_CHECK(
      self.dim() > 0 && self.dim() <= 2, "prob_dist must be 1 or 2 dim");
  TORCH_CHECK(
      at::isFloatingType(self.scalar_type()),
      "multinomial only supports floating-point dtypes for input, got: ",
      self.scalar_type());
  TORCH_CHECK(
      result.scalar_type() == ScalarType::Long,
      "multinomial expects Long tensor out, got: ",
      result.scalar_type());
  TORCH_CHECK(n_sample > 0, "cannot sample n_sample <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(
      with_replacement || (n_sample <= n_categories),
      "cannot sample n_sample > prob_dist.size(-1) samples without replacement");

  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  if (self.dim() == 1) {
    result.resize_({n_sample});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, n_sample});
  }
  if (result.numel() == 0) {
    return result;
  }

  if (!with_replacement || n_sample == 1) {
    // Sanity checks on `self`.
    auto is_valid = ((self.max() < INFINITY) & (self.min() >= 0)).item();
    TORCH_CHECK(
        is_valid.to<bool>(),
        "probability tensor contains either `inf`, `nan` or element < 0");
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    bool zero_prob_condition;
    if (self.dim() == 1) {
      zero_prob_condition = (self.sum() == 0).item().to<bool>();
    } else {
      zero_prob_condition = (self.sum(1) == 0).sum().item().to<bool>();
    }
    TORCH_CHECK(
        !zero_prob_condition,
        "invalid multinomial distribution (sum of probabilities <= 0)");

    // The algorithm is from gumbel softmax.
    // s = argmax( logp - log(-log(eps)) ) where eps ~ U(0, 1)
    // Here we can apply exp to the formula which will not affect result of
    // argmax or topk. Then we have
    // s = argmax( p / (-log(eps)) ) where eps ~ U(0, 1).
    // We can also simplify the formula above by
    // s = argmax( p / q ) where q ~ Exp(1)
    Tensor q = at::empty_like(self).exponential_(1, std::move(gen));
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on CPU
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG). We just
    // ignore it here, but there may be some risk to get invalid output on CPU.
    at::div_out(q, self, q);
    if (n_sample == 1) {
      at::argmax_out(result, q, /*dim=*/-1, /*keepdim=*/true);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, n_sample);
    }
    return result;
  }

  multinomial_with_replacement_stub(
      result.device().type(), result, self, n_sample, gen);
  return result;
}

} // namespace native
namespace musa {

at::Tensor Multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "Multinomial", "self");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at::native::multinomial(self, num_samples, replacement, generator);
}

at::Tensor& MultinomialOut(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator,
    Tensor& out) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "MultinomialOut", "self");
  c10::impl::check_and_update_common_device(
      common_device, out, "MultinomialOut", "out");
  const c10::OptionalDeviceGuard device_guard(device_of(self));
  return at::native::multinomial_out(
      self, num_samples, replacement, generator, out);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("multinomial", Multinomial);
  m.impl("multinomial.out", MultinomialOut);
}

} // namespace musa
} // namespace at