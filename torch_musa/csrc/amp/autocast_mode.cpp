#include "torch_musa/csrc/amp/autocast_mode.h"

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <ATen/Operators.h>
#include <torch/library.h>

#include <c10/core/impl/LocalDispatchKeySet.h>
#include <c10/util/intrusive_ptr.h>

#include <exception>
#include <mutex>

namespace at {
namespace musa {
namespace autocast {

bool is_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCUDA);
}

void set_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(
      DispatchKey::AutocastCUDA, !new_enabled);
}

bool is_cpu_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(DispatchKey::AutocastCPU);
}

void set_cpu_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(
      DispatchKey::AutocastCPU, !new_enabled);
}

bool is_autocast_musa_enabled() {
  return !c10::impl::tls_is_dispatch_key_excluded(
      DispatchKey::AutocastPrivateUse1);
}

void set_autocast_musa_enabled(bool new_enabled) {
  c10::impl::tls_set_dispatch_key_excluded(
      DispatchKey::AutocastPrivateUse1, !new_enabled);
}

namespace {
// Imitate Apex and cache some of the casts to streamline parameter reuse.
// Our heuristic is to cache lower_precision_fp casts of fp32 model weights (see
// cached_cast below).
//
// After discussion with @ezyang, the cache uses the following structure:
// The key is the fp32 source tensor's TensorImpl*, a proxy for a Tensor uuid
// that's unchanged across shallow copies. The value is a tuple with a weakref
// to the source tensor's TensorImpl as the first element and the casted tensor
// as the second element.
//
// The weakref keeps the source's TensorImpl from being deleted.  We need to
// because we're using the source TensorImpl* as the key.  If it were deleted,
// another random Tensor could be allocated whose TensorImpl* happened to have
// the same value.  This TensorImpl* would then mistakenly hit in cache:  a
// rare, intermittent, unpredictable bug.
//
// I'm not using the weak_intrusive_ptr as the key because it's more difficult
// to compare directly against incoming TensorImpl*s.
using weakref_type = c10::weak_intrusive_ptr<TensorImpl, UndefinedTensorImpl>;
using val_type = std::tuple<weakref_type, Tensor>;
std::unordered_map<TensorImpl*, val_type> cached_casts;
std::mutex cached_casts_mutex;

// nesting tracks the nesting depth of the Python-side context manager.
// When the autocast context manager exits to a nesting level that's outside
// any instance of autocast (which should occur at the end of each forward pass)
// it calls clear_cache() to ensure cached Tensors don't leak outside the
// autocasting region.
thread_local int nesting = 0;
// autocast_cpu_dtype is the lower_precision_fp used by AutocastCPU.
thread_local at::ScalarType autocast_cpu_dtype = at::kBFloat16;

// should we enabled the cache inside autocast.
thread_local bool cache_enabled = true;

// autocast_privateuseone_dtype is the lower_precision_fp used by
// AutocastPrivateUse1.
thread_local at::ScalarType autocast_privateuseone_dtype = at::kHalf;

// autocast_musa_dtype is the lower_precision_fp used by AutocastPrivateUse1.
thread_local at::ScalarType autocast_musa_dtype = at::kHalf;
} // namespace

void clear_cache() {
  const std::lock_guard<std::mutex> lock(cached_casts_mutex);
  cached_casts.clear();
}

int increment_nesting() {
  return ++nesting;
}

int decrement_nesting() {
  return --nesting;
}

at::ScalarType get_autocast_cpu_dtype() {
  return autocast_cpu_dtype;
}

void set_autocast_cpu_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kBFloat16,
      "Currently, AutocastCPU only support Bfloat16 as the autocast_cpu_dtype");
  autocast_cpu_dtype = dtype;
}

bool is_autocast_cache_enabled() {
  return cache_enabled;
}

void set_autocast_cache_enabled(bool enabled) {
  cache_enabled = enabled;
}

at::ScalarType get_autocast_musa_dtype() {
  return autocast_musa_dtype;
}

void set_autocast_musa_dtype(at::ScalarType dtype) {
  TORCH_CHECK(
      dtype == at::kHalf || dtype == at::kFloat,
      "Currently, AutoCastMusa only support float16/float32 as the autocast_musa_dtype");
  autocast_musa_dtype = dtype;
}

// Overload to catch Tensor args
Tensor cached_cast(
    at::ScalarType to_type,
    const Tensor& arg,
    DeviceType device_type) {
  if (is_eligible(arg, device_type) && (arg.scalar_type() != to_type)) {
    // Heuristic:  Do what Apex does, and cache lower_precision_fp casts of fp32
    // model weights (leaves). See cached_casts declaration above for detailed
    // strategy.
    bool can_try_cache =
        (to_type == get_lower_precision_fp_from_device_type(device_type) &&
         arg.scalar_type() == at::kFloat && arg.requires_grad() &&
         arg.is_leaf() && !arg.is_view() && cache_enabled);
    if (can_try_cache) {
      const std::lock_guard<std::mutex> lock(cached_casts_mutex);
      auto it = cached_casts.find(arg.unsafeGetTensorImpl());
      if (it != cached_casts.end()) {
        return std::get<1>(it->second);
      } else {
        auto casted_arg = arg.to(to_type);
        cached_casts.emplace(
            arg.unsafeGetTensorImpl(),
            val_type{weakref_type(arg.getIntrusivePtr()), casted_arg});
        return casted_arg;
      }
    } else {
      return arg.to(to_type);
    }
  } else {
    return arg;
  }
}

// Policies correspond to op categories that need code-divergent handling.
// Wrapper templates below are specialized based on a policy template parameter.
enum class CastPolicy : uint8_t {
  lower_precision_fp =
      0, // Cast all inputs to lower_precision_fp before running the op.
         // Currently, lower_precision_fp is fp16 for AutocastPrivateuse1,
         // and is defined by user(default bf16) for AutocastCPU.
  fp32, // Cast all inputs to at::kFloat before running the op.
  fp32_set_opt_dtype, // Treats functions (like softmax) that
                      //   1. we'd like to run in fp32 and
                      //   2. have a c10::optional<ScalarType> arg that controls
                      //   the output type.
                      // fp32_set_opt_dtype wrappers' policy is:  if the output
                      // type is already set, don't touch it, otherwise, set it
                      // to at::kFloat.
  fp32_append_dtype, // Treats functions (like norm) that
                     //   1. we'd like to run in fp32 and
                     //   2. have some overloads that accept an output type and
                     //   other overloads that don't.
                     // fp32_append_dtype wrappers wrap the overloads that don't
                     // have an output dtype. The wrapper policy is:  append
                     // at::kFloat to the args, and redispatch to the type-aware
                     // overload.
  promote, // Run in the widest dtype among several args.
};

// Base template for WrapFunction_, which is specialized to contain a "call"
// method each CastPolicy
template <
    CastPolicy policy,
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class ArgList>
struct WrapFunction_ {};

// CastPolicy::lower_precision_fp General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::lower_precision_fp,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(
        get_lower_precision_fp_from_device_type(device_type),
        args,
        device_type)...);
  }
};

// CastPolicy::fp32 General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::fp32,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    return (*F)(cached_cast(at::kFloat, args, device_type)...);
  }
};

// CastPolicy::fp32_set_opt_dtype DeviceType::PrivateUse1
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::fp32_set_opt_dtype,
    ::at::musa::kMUSA,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        DispatchKey::AutocastPrivateUse1);
    if (firstarg_is_eligible(args...)) {
      return (*F)(set_opt_dtype(at::kFloat, args)...);
    } else {
      // If ineligible, calls F with unaltered args.  Does not set opt dtype,
      // because setting opt dtype explicitly may interfere with internal
      // implicit promotion decisions.
      return (*F)(args...);
    }
  }
};

// CastPolicy::fp32_append_dtype DeviceType::PrivateUse1
template <class Redispatch, Redispatch* F, class Ret, class... Args>
struct WrapFunction_<
    CastPolicy::fp32_append_dtype,
    ::at::musa::kMUSA,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        DispatchKey::AutocastPrivateUse1);
    at::ScalarType out_type = type_from_firstarg(at::kFloat, args...);
    return (*F)(args..., out_type);
  }
};

// CastPolicy::promote General_DeviceType
template <
    DeviceType device_type,
    class Redispatch,
    Redispatch* F,
    class Ret,
    class... Args>
struct WrapFunction_<
    CastPolicy::promote,
    device_type,
    Redispatch,
    F,
    Ret,
    guts::typelist::typelist<Args...>> {
  static Ret call(Args... args) {
    c10::impl::ExcludeDispatchKeyGuard no_autocast(
        get_autocast_dispatch_key_from_device_type(device_type));
    auto to_type = promote_type(
        get_lower_precision_fp_from_device_type(device_type),
        device_type,
        args...);
    return (*F)(cached_cast(to_type, args, device_type)...);
  }
};

// Wrapper to infer return_type and parameter_types for WrapFunction_ (imitating
// core/boxing/impl/WrapFunctionIntoFunctor.h)
template <
    CastPolicy policy,
    DeviceType device_type,
    class Registered, // The signature for which we're registering.  The
                      // dispatcher's calling code invokes our registered
                      // functions with arguments matching Registered, so we
                      // register WrapFunction_::call methods with a matching
                      // signature to properly field those arguments.
                      // guts::function_traits below extracts return_type and
                      // parameter_types from Registered, which WrapFunction_
                      // templates above use to declare their call methods.
    class Redispatch, // The signature for the function we're redispatching to.
                      // In most cases this is the same as Registered, but for
                      // some ops (for example, ops where we append a dtype)
                      // it's useful to redispatch to a function with a
                      // different signature.
    Redispatch* F> // The actual function we're redispatching to.
struct WrapFunction final {
  using type = WrapFunction_<
      policy,
      device_type,
      Redispatch,
      F,
      typename guts::function_traits<Registered>::return_type,
      typename guts::function_traits<Registered>::parameter_types>;
};

/*******************************
Banned functions
*******************************/

Tensor binary_cross_entropy_banned(
    const Tensor&,
    const Tensor&,
    const c10::optional<Tensor>&,
    int64_t) {
  AT_ERROR(
      "torch.nn.functional.binary_cross_entropy and torch.nn.BCELoss are unsafe to autocast.\n"
      "Many models use a sigmoid layer right before the binary cross entropy layer.\n"
      "In this case, combine the two layers using torch.nn.functional.binary_cross_entropy_with_logits\n"
      "or torch.nn.BCEWithLogitsLoss.  binary_cross_entropy_with_logits and BCEWithLogits are\n"
      "safe to autocast.");
}

namespace {
/*****************************************************************************************************************
This section performs load-time registration for autocast wrappers.
It's debatable at what level operations should be patched.  We'd like casts to
be autograd-exposed and precede autograd history recording, so that for
lower_precision_fp ops, input tensors are saved for backward in
lower_precision_fp rather than fp32.  Saving inputs in lower_precision_fp can
significantly reduce a model's memory footprint.
*****************************************************************************************************************/

#define ADD_NS(RAW_OP) at::RAW_OP

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)
#define KERNEL_MUSA(OP, POLICY)           \
  m.impl(                                 \
      TORCH_SELECTIVE_NAME("aten::" #OP), \
      &WrapFunction<                      \
          CastPolicy::POLICY,             \
          ::at::musa::kMUSA,              \
          decltype(ATEN_FN(OP)),          \
          decltype(ATEN_FN(OP)),          \
          &ATEN_FN(OP)>::type::call);
#define KERNEL_MUSA_FOR_MULTIFORM(OP, OVERLOAD, POLICY) \
  m.impl(                                               \
      TORCH_SELECTIVE_NAME("aten::" #OP "." #OVERLOAD), \
      &WrapFunction<                                    \
          CastPolicy::POLICY,                           \
          ::at::musa::kMUSA,                            \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          decltype(ATEN_FN2(OP, OVERLOAD)),             \
          &ATEN_FN2(OP, OVERLOAD)>::type::call);

// Less-common but still useful case: redispatching to a function with a new
// signature (e.g. appending a dtype)
#define KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE( \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  m.impl(                                           \
      TORCH_SELECTIVE_NAME("aten::" REGISTER_NAME), \
      &WrapFunction<                                \
          CastPolicy::POLICY,                       \
          ::at::musa::kMUSA,                        \
          REGISTER_SIGNATURE,                       \
          REDISPATCH_SIGNATURE,                     \
          &REDISPATCH_FUNC>::type::call);

/*****************************************
Explicit registration for out-of-place ops
*****************************************/
TORCH_LIBRARY_IMPL(_, AutocastPrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFallthrough());
}

TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, m) {
  // lower_precision_fp
  KERNEL_MUSA(_convolution, lower_precision_fp)
  KERNEL_MUSA(conv1d, lower_precision_fp)
  KERNEL_MUSA(conv2d, lower_precision_fp)
  KERNEL_MUSA(conv3d, lower_precision_fp)
  KERNEL_MUSA(conv_tbc, lower_precision_fp)
  KERNEL_MUSA(conv_transpose1d, lower_precision_fp)
  KERNEL_MUSA_FOR_MULTIFORM(conv_transpose2d, input, lower_precision_fp)
  KERNEL_MUSA_FOR_MULTIFORM(conv_transpose3d, input, lower_precision_fp)
  KERNEL_MUSA(convolution, lower_precision_fp)
  KERNEL_MUSA(prelu, lower_precision_fp)
  KERNEL_MUSA(addmm, lower_precision_fp)
  KERNEL_MUSA(addmv, lower_precision_fp)
  KERNEL_MUSA(addr, lower_precision_fp)
  KERNEL_MUSA(matmul, lower_precision_fp)
  KERNEL_MUSA(einsum, lower_precision_fp)
  KERNEL_MUSA(mm, lower_precision_fp)
  KERNEL_MUSA(mv, lower_precision_fp)
  KERNEL_MUSA(linear, lower_precision_fp)
  KERNEL_MUSA(addbmm, lower_precision_fp)
  KERNEL_MUSA(baddbmm, lower_precision_fp)
  KERNEL_MUSA(bmm, lower_precision_fp)
  KERNEL_MUSA(chain_matmul, lower_precision_fp)
  KERNEL_MUSA(linalg_multi_dot, lower_precision_fp)
  KERNEL_MUSA(_thnn_fused_lstm_cell, lower_precision_fp)
  KERNEL_MUSA(_thnn_fused_gru_cell, lower_precision_fp)
  KERNEL_MUSA(lstm_cell, lower_precision_fp)
  KERNEL_MUSA(gru_cell, lower_precision_fp)
  KERNEL_MUSA(rnn_tanh_cell, lower_precision_fp)
  KERNEL_MUSA(rnn_relu_cell, lower_precision_fp)
  KERNEL_MUSA(_scaled_dot_product_flash_attention, lower_precision_fp)
  KERNEL_MUSA(scaled_dot_product_attention, lower_precision_fp)

  // fp32
  KERNEL_MUSA(acos, fp32)
  KERNEL_MUSA(asin, fp32)
  KERNEL_MUSA(cosh, fp32)
  KERNEL_MUSA(erfinv, fp32)
  KERNEL_MUSA(exp, fp32)
  KERNEL_MUSA(expm1, fp32)
  KERNEL_MUSA(log, fp32)
  KERNEL_MUSA(log10, fp32)
  KERNEL_MUSA(log2, fp32)
  KERNEL_MUSA(log1p, fp32)
  KERNEL_MUSA(reciprocal, fp32)
  KERNEL_MUSA(rsqrt, fp32)
  KERNEL_MUSA(sinh, fp32)
  KERNEL_MUSA(tan, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Tensor_Scalar, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Tensor_Tensor, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(pow, Scalar, fp32)
  KERNEL_MUSA(softplus, fp32)
  KERNEL_MUSA(layer_norm, fp32)
  KERNEL_MUSA(native_layer_norm, fp32)
  KERNEL_MUSA(group_norm, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(frobenius_norm, dim, fp32)
  KERNEL_MUSA(nuclear_norm, fp32)
  KERNEL_MUSA_FOR_MULTIFORM(nuclear_norm, dim, fp32)
  KERNEL_MUSA(cosine_similarity, fp32)
  KERNEL_MUSA(poisson_nll_loss, fp32)
  KERNEL_MUSA(cosine_embedding_loss, fp32)
  KERNEL_MUSA(nll_loss, fp32)
  KERNEL_MUSA(nll_loss2d, fp32)
  KERNEL_MUSA(hinge_embedding_loss, fp32)
  KERNEL_MUSA(kl_div, fp32)
  KERNEL_MUSA(l1_loss, fp32)
  KERNEL_MUSA(smooth_l1_loss, fp32)
  KERNEL_MUSA(huber_loss, fp32)
  KERNEL_MUSA(mse_loss, fp32)
  KERNEL_MUSA(margin_ranking_loss, fp32)
  KERNEL_MUSA(multilabel_margin_loss, fp32)
  KERNEL_MUSA(soft_margin_loss, fp32)
  KERNEL_MUSA(triplet_margin_loss, fp32)
  KERNEL_MUSA(multi_margin_loss, fp32)
  KERNEL_MUSA(binary_cross_entropy_with_logits, fp32)
  KERNEL_MUSA(dist, fp32)
  KERNEL_MUSA(pdist, fp32)
  KERNEL_MUSA(cdist, fp32)
  KERNEL_MUSA(renorm, fp32)
  // fp32_set_opt_dtype
  KERNEL_MUSA(prod, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(prod, dim_int, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(prod, dim_Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(softmax, int, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(log_softmax, int, fp32_set_opt_dtype)

  KERNEL_MUSA_FOR_MULTIFORM(log_softmax, Dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(cumprod, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(cumprod, dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(cumsum, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(cumsum, dimname, fp32_set_opt_dtype)
  KERNEL_MUSA(linalg_vector_norm, fp32_set_opt_dtype)
  KERNEL_MUSA(linalg_matrix_norm, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(linalg_matrix_norm, str_ord, fp32_set_opt_dtype)

  // TODO(kangchen): sum calcuate error when the output and self dtype
  // are different It will get inf, which cuda registers fp32_set_opt_dtype,
  // So I registered as fp32 temporarily.
  KERNEL_MUSA(sum, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(sum, dim_IntList, fp32_set_opt_dtype)
  KERNEL_MUSA_FOR_MULTIFORM(sum, dim_DimnameList, fp32_set_opt_dtype)

  // fp32_append_dtype
  // The fp32_append_dtype wrapper overrides implicit promotion behavior.

  // norm does not implicitly promote, but be aware when adding new ops to this
  // policy.
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.Scalar",
      Tensor(const Tensor&, const Scalar&),
      Tensor(const Tensor&, const c10::optional<Scalar>&, ScalarType),
      fp32_append_dtype)
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, IntArrayRef, bool),
      Tensor(
          const Tensor&,
          const c10::optional<Scalar>&,
          IntArrayRef,
          bool,
          ScalarType),
      fp32_append_dtype)
  KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE(
      ADD_NS(norm),
      "norm.names_ScalarOpt_dim",
      Tensor(const Tensor&, const c10::optional<Scalar>&, DimnameList, bool),
      Tensor(
          const Tensor&,
          const c10::optional<Scalar>&,
          DimnameList,
          bool,
          ScalarType),
      fp32_append_dtype)
  // promote
  KERNEL_MUSA(addcdiv, promote)
  KERNEL_MUSA(addcmul, promote)
  KERNEL_MUSA(atan2, promote)
  KERNEL_MUSA(bilinear, promote)
  KERNEL_MUSA(cross, promote)
  KERNEL_MUSA(dot, promote)
  KERNEL_MUSA(grid_sampler, promote)
  KERNEL_MUSA(index_put, promote)
  KERNEL_MUSA(tensordot, promote)
  KERNEL_MUSA(scatter_add, promote)

  m.impl(
      TORCH_SELECTIVE_NAME("aten::binary_cross_entropy"),
      TORCH_FN((&at::musa::autocast::binary_cross_entropy_banned)));
}
} // namespace
} // namespace autocast
} // namespace musa
} // namespace at
