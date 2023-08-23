#include "torch_musa/csrc/aten/ops/Amp.h"

#include <math.h>
#include <mudnn.h>
#include <musa_fp16.h>

#include <ATen/core/Tensor.h>
#include <ATen/DeviceGuard.h>
#include <ATen/Dispatch.h>
#include <ATen/musa/MUSAConfig.h>
#include <ATen/native/musa/ForeachFunctors.muh>
#include <ATen/native/musa/Loops.muh>
#include <ATen/native/ForeachUtils.h>
#include <ATen/native/TensorIterator.h>


namespace at {
namespace musa {


// amp_update_scale_musa_kernel is launched with a single thread to compute the new scale.
// The scale factor is maintained and updated on the GPU to avoid synchronization.
__global__ void amp_update_scale_musa_kernel(float* current_scale,
                                             int* growth_tracker,
                                             float* found_inf,
                                             float growth_factor,
                                             float backoff_factor,
                                             int growth_interval)
{
  if (*found_inf) {
    *current_scale = (*current_scale)*backoff_factor;
    *growth_tracker = 0;
  } else {
    // Entering this branch means we just carried out a successful step,
    // so growth_tracker is incremented before comparing to growth_interval.
    auto successful = (*growth_tracker) + 1;
    if (successful == growth_interval) {
      *current_scale = (*current_scale) * growth_factor;
      *growth_tracker = 0;
    } else {
      *growth_tracker = successful;
    }
  }
}


// AmpUpdateScaleMusa asynchronously updates the scale tensor in place.
//
// Args:
// current_scale:  A one-element musa float tensor containing the scale value.
// growth_tracker:  A one-element torch.musa.IntTensor containing the number of recent consecutive unskipped steps.
// found_inf:  A one-element musa float tensor. If > 0, indicates that infs/nans were found by the relevant
//             prior _amp_non_finite_check_and_unscale_cuda call, and 0 if no infs/nans were found.
// growth_factor:  Multiplier if no infs/NaNs were found (typically slightly > 1).
// backoff_factor:  Multiplier if infs/NaNs were found (typically 0.5).
// growth_interval:  Number of consecutive unskipped steps that must occur for current_scale to be multiplied by
//                   growth_factor.
//
// Returns:
// current_scale
Tensor& AmpUpdateScaleMusa(Tensor& current_scale,
                                Tensor& growth_tracker,
                                const Tensor& found_inf,
                                float growth_factor,
                                float backoff_factor,
                                int64_t growth_interval)
{
  TORCH_CHECK(growth_tracker.is_privateuseone(), "growth_tracker must be a MUSA tensor.");
  TORCH_CHECK(current_scale.is_privateuseone(), "current_scale must be a MUSA tensor.");
  TORCH_CHECK(found_inf.is_privateuseone(), "found_inf must be a MUSA tensor.");
  TORCH_CHECK(growth_tracker.numel() == 1, "growth_tracker must be a 1-element tensor.");
  TORCH_CHECK(current_scale.numel() == 1, "current_scale must be a 1-element tensor.");
  TORCH_CHECK(found_inf.numel() == 1, "found_inf must be a 1-element tensor.");
  TORCH_CHECK(growth_tracker.scalar_type() == at::ScalarType::Int, "growth_tracker must be an int tensor.");
  TORCH_CHECK(current_scale.scalar_type() == at::ScalarType::Float, "current_scale must be a float tensor.");
  TORCH_CHECK(found_inf.scalar_type() == at::ScalarType::Float, "found_inf must be a float tensor.");

  amp_update_scale_musa_kernel<<<1, 1, 0, at::musa::getCurrentMUSAStream()>>>(
    current_scale.data_ptr<float>(),
    growth_tracker.data_ptr<int>(),
    found_inf.data_ptr<float>(),
    growth_factor,
    backoff_factor,
    growth_interval);
  C10_MUSA_KERNEL_LAUNCH_CHECK();

  return current_scale;
}

}  // namespace musa
}  // namespace at
