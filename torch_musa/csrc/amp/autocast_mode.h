#ifndef TORCH_MUSA_CSRC_AMP_AUTOCAST_MODE_H_
#define TORCH_MUSA_CSRC_AMP_AUTOCAST_MODE_H_

#include <ATen/ATen.h>
#include <ATen/autocast_mode.h>
#include <pybind11/pybind11.h>

#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::autocast::musa {

TORCH_API bool is_autocast_musa_enabled();

TORCH_API void set_autocast_musa_enabled(bool new_enabled);

TORCH_API at::ScalarType get_autocast_musa_dtype();

TORCH_API void set_autocast_musa_dtype(at::ScalarType dtype);

PyMethodDef* GetAutocastMethods();

} // namespace at::autocast::musa

#define ADD_NS(RAW_OP) at::RAW_OP

// Common cases where registration signature matches redispatch signature
// (that's why SIGNATURE is repeated in the WrapFunction instantiation)

// KERNEL_CUDA/KERNEL_CUDA2/KERNEL_DIFFERENT_REDISPATCH_SIGNATURE_CUDA
// registration for AutocastCUDA
#define KERNEL_MUSA(OP, POLICY) KERNEL(c10::DeviceType::PrivateUse1, OP, POLICY)

#define KERNEL_MUSA_FOR_MULTIFORM(OP, OVERLOAD, POLICY) \
  KERNEL2(c10::DeviceType::PrivateUse1, OP, OVERLOAD, POLICY)

#define KERNEL_MUSA_DIFFERENT_REDISPATCH_SIGNATURE( \
    REDISPATCH_FUNC,                                \
    REGISTER_NAME,                                  \
    REGISTER_SIGNATURE,                             \
    REDISPATCH_SIGNATURE,                           \
    POLICY)                                         \
  KERNEL_DIFFERENT_REDISPATCH_SIGNATURE(            \
      c10::DeviceType::PrivateUse1,                 \
      REDISPATCH_FUNC,                              \
      REGISTER_NAME,                                \
      REGISTER_SIGNATURE,                           \
      REDISPATCH_SIGNATURE,                         \
      POLICY)

#endif // TORCH_MUSA_CSRC_AMP_AUTOCAST_MODE_H_
