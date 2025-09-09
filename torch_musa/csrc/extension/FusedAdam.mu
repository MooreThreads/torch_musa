#include <ATen/TypeDefault.h>
#include <c10/util/Exception.h>

#include <ATen/Dispatch.h>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/musa/ForeachFunctors.muh>

#include "common/multi_tensor_apply.muh"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace musa_extension {
namespace {

enum class ADAM_MODE : uint8_t { ORIGINAL = 0, ADAMW = 1 };

constexpr uint8_t kParamIdx = 0;
constexpr uint8_t kGradIdx = 1;
constexpr uint8_t kExpAvgIdx = 2;
constexpr uint8_t kExpAvgSqIdx = 3;
constexpr uint8_t kMaxExpAvgSqIdx = 4;

template <int n, typename opmath_t>
struct FusedAdamOptimizerTensorListMetadata {
  const void* addresses[n][depth_to_max_tensors[n - 1]];
  int64_t numel_for_tensor[depth_to_max_tensors[n - 1]];

  // if lr is GPU Tensor, bias_correction1_r_or_step_size stands for
  // bias_correction1, step_size, i.e. lr * bias_correction1 otherwise.
  opmath_t
      bias_correction1_r_or_step_size[depth_to_max_tensors_scalarlist[n - 1]];
  opmath_t bias_correction2_rsqrt[depth_to_max_tensors_scalarlist[n - 1]];

  unsigned char block_to_tensor[depth_to_max_blocks[n - 1]];
  int block_to_chunk[depth_to_max_blocks[n - 1]];
  int start_tensor_this_launch;
};

template <typename opmath_t, int depth, typename T>
__device__ bool init_args(
    T** args,
    FusedAdamOptimizerTensorListMetadata<depth, opmath_t>& tl,
    const int64_t chunk_idx,
    const int64_t chunk_size,
    const int64_t tensor_loc) {
  bool all_aligned = true;
  for (int i = 0; i < depth; i++) {
    args[i] = (T*)tl.addresses[i][tensor_loc];
    args[i] += chunk_idx * chunk_size;

    if (!is_aligned(args[i])) {
      all_aligned = false;
    }
  }
  return all_aligned;
}

template <
    typename scalar_type,
    typename opmath_t,
    int depth,
    bool precomputed_step_size,
    ADAM_MODE adam_mode,
    bool amsgrad,
    std::enable_if_t<std::is_same<opmath_t, float>::value, int> = 0>
__device__ __forceinline__ void AdamMath(
    scalar_type r_args[depth][kILP],
    const double& lr,
    const double& beta1,
    const double& beta2,
    const opmath_t& bias_correction1_r_or_step_size,
    const opmath_t& bias_correction2_rsqrt,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  for (int ii = 0; ii < kILP; ii++) {
    // Load values.
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
    opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
    opmath_t max_exp_avg_sq;
    if constexpr (amsgrad) {
      max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
    }
    // Update param, grad, 1st and 2nd order momentum.

    if (weight_decay != 0) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad += param * weight_decay;
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param -= lr * weight_decay * param;
      }
    }

    // use fma to optimize lerp, ref:
    // https://developer.nvidia.com/blog/lerp-faster-cuda/
    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
    opmath_t grad_2 = grad * grad;
    exp_avg = fmaf(beta1, exp_avg, fmaf(-beta1, grad, grad));
    exp_avg_sq = fmaf(beta2, exp_avg_sq, fmaf(-beta2, grad_2, grad_2));

    opmath_t denom;
    if constexpr (amsgrad) {
      max_exp_avg_sq = fmaxf(max_exp_avg_sq, exp_avg_sq);
      denom = (sqrtf(max_exp_avg_sq) * bias_correction2_rsqrt) + eps;
    } else {
      // denom = (sqrtf(exp_avg_sq) * bias_correction2_rsqrt) + eps;
      denom = fmaf(sqrtf(exp_avg_sq), bias_correction2_rsqrt, eps);
    }

    if constexpr (precomputed_step_size) {
      param -= bias_correction1_r_or_step_size * exp_avg / denom;
    } else {
      param -= lr * bias_correction1_r_or_step_size * exp_avg / denom;
    }

    // Store results.
    r_args[kParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = grad_to_store;
    }
    r_args[kExpAvgIdx][ii] = exp_avg;
    r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    if constexpr (amsgrad) {
      r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
    }
  }
}

template <
    typename scalar_type,
    typename opmath_t,
    int depth,
    bool precomputed_step_size,
    ADAM_MODE adam_mode,
    bool amsgrad,
    std::enable_if_t<!std::is_same<opmath_t, float>::value, int> = 0>
__device__ __forceinline__ void AdamMath(
    scalar_type r_args[depth][kILP],
    const double& lr,
    const double& beta1,
    const double& beta2,
    const opmath_t& bias_correction1_r_or_step_size,
    const opmath_t& bias_correction2_rsqrt,
    const double& weight_decay,
    const double& eps,
    const bool& maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  for (int ii = 0; ii < kILP; ii++) {
    // Load values.
    opmath_t param = static_cast<opmath_t>(r_args[kParamIdx][ii]);
    opmath_t grad = static_cast<opmath_t>(r_args[kGradIdx][ii]);
    if (grad_scale_ptr) {
      grad /= (static_cast<double>(*grad_scale_ptr));
    }
    const opmath_t grad_to_store = grad;
    if (maximize) {
      grad = -grad;
    }
    opmath_t exp_avg = static_cast<opmath_t>(r_args[kExpAvgIdx][ii]);
    opmath_t exp_avg_sq = static_cast<opmath_t>(r_args[kExpAvgSqIdx][ii]);
    opmath_t max_exp_avg_sq;
    if constexpr (amsgrad) {
      max_exp_avg_sq = static_cast<opmath_t>(r_args[kMaxExpAvgSqIdx][ii]);
    }
    // Update param, grad, 1st and 2nd order momentum.

    if (weight_decay != 0) {
      if constexpr (adam_mode == ADAM_MODE::ORIGINAL) {
        grad += param * weight_decay;
      } else if constexpr (adam_mode == ADAM_MODE::ADAMW) {
        param -= lr * weight_decay * param;
      }
    }

    // use fma to optimize lerp, ref:
    // https://developer.nvidia.com/blog/lerp-faster-cuda/
    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad;
    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad * grad;
    opmath_t grad_2 = grad * grad;
    exp_avg = fma(beta1, exp_avg, fma(-beta1, grad, grad));
    exp_avg_sq = fma(beta2, exp_avg_sq, fma(-beta2, grad_2, grad_2));

    opmath_t denom;
    if (amsgrad) {
      max_exp_avg_sq = fmax(max_exp_avg_sq, exp_avg_sq);
      denom = (sqrt(max_exp_avg_sq) * bias_correction2_rsqrt) + eps;
    } else {
      // denom = (sqrtf(exp_avg_sq) * bias_correction2_rsqrt) + eps;
      denom = fma(sqrt(exp_avg_sq), bias_correction2_rsqrt, eps);
    }

    if constexpr (precomputed_step_size) {
      param -= bias_correction1_r_or_step_size * exp_avg / denom;
    } else {
      param -= lr * bias_correction1_r_or_step_size * exp_avg / denom;
    }

    // Store results.
    r_args[kParamIdx][ii] = param;
    if (grad_scale_ptr) {
      r_args[kGradIdx][ii] = grad_to_store;
    }
    r_args[kExpAvgIdx][ii] = exp_avg;
    r_args[kExpAvgSqIdx][ii] = exp_avg_sq;
    if constexpr (amsgrad) {
      r_args[kMaxExpAvgSqIdx][ii] = max_exp_avg_sq;
    }
  }
}

template <typename scalar_type, int depth, ADAM_MODE adam_mode, bool amsgrad>
struct FusedAdamMathFunctor {
  static_assert(
      depth == 4 || depth == 5,
      "depth of 4 for Adam, depth of 5 for Adam with AMSGrad.");
  using opmath_t = at::opmath_type<scalar_type>;
  __device__ __forceinline__ void operator()(
      int chunk_size,
      FusedAdamOptimizerTensorListMetadata<depth, opmath_t>& tl,
      const float* lr_ptr,
      const double& lr,
      const double& beta1,
      const double& beta2,
      const double& weight_decay,
      const double& eps,
      const bool& maximize,
      const float* grad_scale_ptr,
      const float* found_inf_ptr) {
    int tensor_loc = tl.block_to_tensor[blockIdx.x];
    int chunk_idx = tl.block_to_chunk[blockIdx.x];
    int n = tl.numel_for_tensor[tensor_loc];
    double lr_double = lr_ptr ? *lr_ptr : lr;

    if (found_inf_ptr && *found_inf_ptr == 1) {
      return;
    }

    auto bias_correction1_r_or_step_size =
        tl.bias_correction1_r_or_step_size[tensor_loc];
    auto bias_correction2_rsqrt = tl.bias_correction2_rsqrt[tensor_loc];

    scalar_type* args[depth];
    const bool all_aligned{init_args<opmath_t, depth>(
        args, tl, chunk_idx, chunk_size, tensor_loc)};
    n -= chunk_idx * chunk_size;
    scalar_type r_args[depth][kILP];

    if ((n % kILP == 0) && (chunk_size % kILP == 0) && all_aligned) {
      // for (int64_t i_start = threadIdx.x * kILP;
      //      i_start < n && i_start < chunk_size;
      //      i_start += blockDim.x * kILP) {
      for (int64_t i_start = threadIdx.x;
           i_start * kILP < n && i_start * kILP < chunk_size;
           i_start += blockDim.x) {
        for (int i = 0; i < depth; i++) {
          load_store(r_args[i], args[i], 0, i_start);
        }
        if (lr_ptr) {
          AdamMath<scalar_type, opmath_t, depth, false, adam_mode, amsgrad>(
              r_args,
              lr_double,
              beta1,
              beta2,
              bias_correction1_r_or_step_size,
              bias_correction2_rsqrt,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        } else {
          AdamMath<scalar_type, opmath_t, depth, true, adam_mode, amsgrad>(
              r_args,
              lr_double,
              beta1,
              beta2,
              bias_correction1_r_or_step_size,
              bias_correction2_rsqrt,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        }
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            load_store(args[i], r_args[i], i_start, 0);
          }
        }
      }
    } else {
      for (int64_t i_start = 0; i_start < n && i_start < chunk_size;
           i_start += blockDim.x * kILP) {
        load_args<depth>(r_args, args, i_start, chunk_size, n);
        if (lr_ptr) {
          AdamMath<scalar_type, opmath_t, depth, false, adam_mode, amsgrad>(
              r_args,
              lr_double,
              beta1,
              beta2,
              bias_correction1_r_or_step_size,
              bias_correction2_rsqrt,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        } else {
          AdamMath<scalar_type, opmath_t, depth, true, adam_mode, amsgrad>(
              r_args,
              lr_double,
              beta1,
              beta2,
              bias_correction1_r_or_step_size,
              bias_correction2_rsqrt,
              weight_decay,
              eps,
              maximize,
              grad_scale_ptr,
              found_inf_ptr);
        }
        for (int i = 0; i < depth; i++) {
          if (i != kGradIdx || grad_scale_ptr) {
            store_args(args[i], r_args[i], i_start, chunk_size, n);
          }
        }
      }
    }
  }
};

// diffs from multi_tensor_apply_for_fused_optimizer, the step_size
// and bias_correction2_rsqrt will be precomputed on the CPU side.
// assume same step across group
template <typename scalar_t, int depth, typename T>
void multi_tensor_apply_for_fused_adam(
    std::vector<std::vector<at::Tensor>>& tensor_lists,
    std::vector<at::Tensor> state_steps,
    T callable,
    const float* lr_ptr, // points to GPU address
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const float* grad_scale_ptr,
    const float* found_inf_ptr) {
  using opmath_t = at::opmath_type<scalar_t>;
  TORCH_CHECK(
      tensor_lists.size() == depth,
      "Number of tensor lists has to match the depth");
  const auto num_tensors = tensor_lists[0].size();
  FusedAdamOptimizerTensorListMetadata<depth, opmath_t> tensorListMeta;

  int loc_block_info = 0;
  int loc_tensor_info = 0;
  for (const auto& tensor_index : c10::irange(num_tensors)) {
    // short-circuit to avoid adding empty tensors to tensorListMeta
    if (tensor_lists[0][tensor_index].numel() == 0) {
      continue;
    }
    auto* step_count =
        reinterpret_cast<float*>(state_steps[tensor_index].data_ptr());
    // assume same step across group
    *step_count += 1;
    if (lr_ptr) {
      tensorListMeta.bias_correction1_r_or_step_size[loc_tensor_info] =
          static_cast<opmath_t>(1.0) / (1 - std::pow(beta1, *step_count));
    } else {
      tensorListMeta.bias_correction1_r_or_step_size[loc_tensor_info] =
          static_cast<opmath_t>(lr) / (1 - std::pow(beta1, *step_count));
    }
    tensorListMeta.bias_correction2_rsqrt[loc_tensor_info] =
        1.0 / std::sqrt((1 - std::pow(beta2, *step_count)));
    tensorListMeta.numel_for_tensor[loc_tensor_info] =
        tensor_lists[0][tensor_index].numel();
    for (const auto& d : c10::irange(depth)) {
      tensorListMeta.addresses[d][loc_tensor_info] =
          tensor_lists[d][tensor_index].const_data_ptr();
    }
    loc_tensor_info++;

    // see above note: [chunking territory]
    const auto numel = tensor_lists[0][tensor_index].numel();
    const auto chunks = numel / kChunkSize + (numel % kChunkSize != 0);
    TORCH_CHECK(chunks > -1);
    for (const auto& chunk : c10::irange(chunks)) {
      tensorListMeta.block_to_tensor[loc_block_info] = loc_tensor_info - 1;
      tensorListMeta.block_to_chunk[loc_block_info] = chunk;
      loc_block_info++;

      const auto tensor_full =
          (loc_tensor_info == depth_to_max_tensors[depth - 1] &&
           chunk == chunks - 1);
      const auto blocks_full = loc_block_info == depth_to_max_blocks[depth - 1];

      if (tensor_full || blocks_full) {
        multi_tensor_apply_kernel<<<
            loc_block_info,
            kBlockSize,
            0,
            c10::musa::getCurrentMUSAStream()>>>(
            tensorListMeta,
            callable,
            lr_ptr,
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
        C10_MUSA_KERNEL_LAUNCH_CHECK();

        // Reset.
        loc_block_info = 0;
        if (chunk == chunks - 1) {
          loc_tensor_info = 0;
        } else {
          tensorListMeta.numel_for_tensor[0] =
              tensorListMeta.numel_for_tensor[loc_tensor_info - 1];
          tensorListMeta.bias_correction1_r_or_step_size[0] =
              tensorListMeta
                  .bias_correction1_r_or_step_size[loc_tensor_info - 1];
          tensorListMeta.bias_correction2_rsqrt[0] =
              tensorListMeta.bias_correction2_rsqrt[loc_tensor_info - 1];
          for (const auto& d : c10::irange(depth)) {
            tensorListMeta.addresses[d][0] =
                tensorListMeta.addresses[d][loc_tensor_info - 1];
          }
          loc_tensor_info = 1;
        }
      }
    }
  }

  // see above note: [finishing what we've started]
  if (loc_block_info != 0) {
    multi_tensor_apply_kernel<<<
        loc_block_info,
        kBlockSize,
        0,
        c10::musa::getCurrentMUSAStream()>>>(
        tensorListMeta,
        callable,
        lr_ptr,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale_ptr,
        found_inf_ptr);
    C10_MUSA_KERNEL_LAUNCH_CHECK();
  }
}

} // anonymous namespace

namespace {

void FusedAdamKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ORIGINAL, false>(),
            lr_ptr, // unused
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

// The following overload simply has a Tensor lr
void FusedAdamKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ORIGINAL, false>(),
            lr_ptr,
            1.0, // unused
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void FusedAdamAMSGradKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ORIGINAL, true>(),
            lr_ptr, // unused
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void FusedAdamAMSGradKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adam_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ORIGINAL, true>(),
            lr_ptr,
            1.0, // unused
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void FusedAdamWKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adamw_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ADAMW, false>(),
            lr_ptr, // unused
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

// The following overload simply has a Tensor lr
void FusedAdamWKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adamw_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 4>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 4, ADAM_MODE::ADAMW, false>(),
            lr_ptr,
            1.0, // unused
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void FusedAdamWAMSGradKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = nullptr;

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adamw_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ADAMW, true>(),
            lr_ptr, // unused
            lr,
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

void FusedAdamWAMSGradKernelImpl(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  std::vector<std::vector<at::Tensor>> tensor_lists{
      params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs};

  float* grad_scale_ptr =
      grad_scale.has_value() ? grad_scale->data_ptr<float>() : nullptr;
  float* found_inf_ptr =
      found_inf.has_value() ? found_inf->data_ptr<float>() : nullptr;
  float* lr_ptr = lr.data_ptr<float>();

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf,
      at::kBFloat16,
      params[0].scalar_type(),
      "fused_adamw_kernel_musa",
      [&]() {
        multi_tensor_apply_for_fused_adam<scalar_t, 5>(
            tensor_lists,
            state_steps,
            FusedAdamMathFunctor<scalar_t, 5, ADAM_MODE::ADAMW, true>(),
            lr_ptr,
            1.0, // unused
            beta1,
            beta2,
            weight_decay,
            eps,
            maximize,
            grad_scale_ptr,
            found_inf_ptr);
      });
}

} // anonymous namespace

} // namespace musa_extension

void FusedAdamKernel(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(state_steps[0].is_cpu(), "state_steps should be CPU Tensors");
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamAMSGradKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

// The following overload simply has a Tensor lr
void FusedAdamKernel(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(state_steps[0].is_cpu(), "state_steps should be CPU Tensors");
  if (lr.is_cpu()) {
    FusedAdamKernel(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamAMSGradKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

void FusedAdamWKernel(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const double lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(state_steps[0].is_cpu(), "state_steps should be CPU Tensors");
  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamWAMSGradKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamWKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}

// The following overload simply has a Tensor lr
void FusedAdamWKernel(
    std::vector<at::Tensor> params,
    std::vector<at::Tensor> grads,
    std::vector<at::Tensor> exp_avgs,
    std::vector<at::Tensor> exp_avg_sqs,
    std::vector<at::Tensor> max_exp_avg_sqs,
    std::vector<at::Tensor> state_steps,
    const at::Tensor& lr,
    const double beta1,
    const double beta2,
    const double weight_decay,
    const double eps,
    const bool amsgrad,
    const bool maximize,
    const c10::optional<at::Tensor>& grad_scale,
    const c10::optional<at::Tensor>& found_inf) {
  TORCH_CHECK(state_steps[0].is_cpu(), "state_steps should be CPU Tensors");
  if (lr.is_cpu()) {
    FusedAdamWKernel(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr.item<double>(),
        beta1,
        beta2,
        weight_decay,
        eps,
        amsgrad,
        maximize,
        grad_scale,
        found_inf);
    return;
  }

  if (amsgrad) {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs}),
        "params, grads, exp_avgs, exp_avg_sqs, and max_exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamWAMSGradKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  } else {
    TORCH_CHECK(
        at::native::check_fast_path_restrictions(
            {params, grads, exp_avgs, exp_avg_sqs}),
        "params, grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout");
    musa_extension::FusedAdamWKernelImpl(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        lr,
        beta1,
        beta2,
        weight_decay,
        eps,
        maximize,
        grad_scale,
        found_inf);
  }
}
