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
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace at {

namespace native {

DEFINE_DISPATCH(multinomial_with_replacement_stub);
} // namespace native

namespace musa {

/* The largest consecutive integer representable in float32 (2^24) */
constexpr int64_t FLOAT32_MAX_CONSECUTIVE_INT = 1 << (FLT_MANT_DIG);

at::Tensor& MultinomialOut(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator,
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
      "multinomial expects Long tensor result, got: ",
      result.scalar_type());
  TORCH_CHECK(num_samples > 0, "cannot sample num_samples <= 0 samples");
  int64_t n_categories = self.size(-1);
  TORCH_CHECK(
      replacement || (num_samples <= n_categories),
      "cannot sample num_samples > prob_dist.size(-1) samples without replacement");
  // Since the index tensor is float, numCategories cannot exceed max
  // float integer precision
  TORCH_CHECK(
      n_categories <= FLOAT32_MAX_CONSECUTIVE_INT,
      "number of categories cannot exceed 2^24");

  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, self, "MultinomialOut", "self");
  c10::impl::check_and_update_common_device(
      common_device, result, "MultinomialOut", "result");
  const c10::OptionalDeviceGuard device_guard(device_of(self));

  if (self.dim() == 1) {
    result.resize_({num_samples});
  } else {
    const int64_t n_dist = self.size(0);
    result.resize_({n_dist, num_samples});
  }
  if (result.numel() == 0) {
    return result;
  }

  if (!replacement) {
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
    Tensor q = at::empty_like(self).exponential_(1, std::move(generator));
    // In theory the probability to generate 0 from exponential distribution is
    // 0. However, on CUDA side there is a protection to avoid 0s, but on MUSA
    // side, there is a very low probability to generate 0 from
    // exponential<double>. The probability is about 2^(-DBL_MANT_DIG), which
    // would cause metrics degradation of LLMs, so we disable gumbel-max trick
    // thing, just call stub kernel instead.
    at::div_out(q, self, q);
    if (num_samples == 1) {
      // for num_samples == 1, it doesn't matter that replacement is true or
      // false
      // TODO(@mt-ai): we should check this "generator" thing
      at::native::multinomial_with_replacement_stub(
          kMUSA, result, self, num_samples, generator);
    } else {
      Tensor vals = at::empty(result.sizes(), self.options());
      at::topk_out(vals, result, q, num_samples);
    }
    return result;
  }

  at::native::multinomial_with_replacement_stub(
      kMUSA, result, self, num_samples, generator);
  return result;
}

at::Tensor Multinomial(
    const at::Tensor& self,
    int64_t num_samples,
    bool replacement,
    c10::optional<at::Generator> generator) {
  Tensor result = at::empty({0}, self.options().dtype(at::kLong));
  return MultinomialOut(self, num_samples, replacement, generator, result);
}

ADVANCED_REGISTER(aten, PrivateUse1, "multinomial", Multinomial)
ADVANCED_REGISTER(aten, PrivateUse1, "multinomial.out", MultinomialOut)

} // namespace musa
} // namespace at
