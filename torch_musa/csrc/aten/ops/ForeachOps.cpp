#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

namespace aten {
namespace musa {

namespace {

#define FOREACH_BINARY_OP_SCALAR(OP)                                           \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_scalar_kernel_musa(          \
      at::TensorList self, const at::Scalar& scalar) {                         \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));          \
    return at::native::foreach_tensor_##OP##_scalar_kernel_cuda(self, scalar); \
  }                                                                            \
                                                                               \
  void foreach_tensor_##OP##_scalar_kernel_musa_(                              \
      at::TensorList self, const at::Scalar& scalar) {                         \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));          \
    return at::native::foreach_tensor_##OP##_scalar_kernel_cuda_(              \
        self, scalar);                                                         \
  }

#define FOREACH_BINARY_OP_LIST(OP)                                       \
  std::vector<at::Tensor> foreach_tensor_##OP##_list_kernel_musa(        \
      at::TensorList tensor1, at::TensorList tensor2) {                  \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1)); \
    return at::native::foreach_tensor_##OP##_list_kernel_cuda(           \
        tensor1, tensor2);                                               \
  }                                                                      \
                                                                         \
  void foreach_tensor_##OP##_list_kernel_musa_(                          \
      at::TensorList tensor1, at::TensorList tensor2) {                  \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1)); \
    return at::native::foreach_tensor_##OP##_list_kernel_cuda_(          \
        tensor1, tensor2);                                               \
  }

#define FOREACH_BINARY_OP_SCALARLIST(OP)                                  \
  void foreach_tensor_##OP##_scalarlist_kernel_musa_(                     \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {            \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));     \
    return at::native::foreach_tensor_##OP##_scalarlist_kernel_cuda_(     \
        self, scalars);                                                   \
  }                                                                       \
                                                                          \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_scalarlist_kernel_musa( \
      at::TensorList self, at::ArrayRef<at::Scalar> scalars) {            \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));     \
    return at::native::foreach_tensor_##OP##_scalarlist_kernel_cuda(      \
        self, scalars);                                                   \
  }

#define FOREACH_BINARY_OP_LIST_ALPHA(OP)                                    \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_list_alpha_kernel_musa(   \
      at::TensorList self, at::TensorList other, const at::Scalar& alpha) { \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));       \
    return at::native::foreach_tensor_##OP##_list_kernel_cuda(              \
        self, other, alpha);                                                \
  }                                                                         \
                                                                            \
  void foreach_tensor_##OP##_list_alpha_kernel_musa_(                       \
      at::TensorList self, at::TensorList other, const at::Scalar& alpha) { \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));       \
    return at::native::foreach_tensor_##OP##_list_kernel_cuda_(             \
        self, other, alpha);                                                \
  }

#define FOREACH_UNARY_OP(OP)                                                  \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_musa(at::TensorList self) { \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));         \
    return at::native::foreach_tensor_##OP##_cuda(self);                      \
  }                                                                           \
                                                                              \
  void foreach_tensor_##OP##_musa_(at::TensorList self) {                     \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self));         \
    return at::native::foreach_tensor_##OP##_cuda_(self);                     \
  }

#define FOREACH_POINTWISE_OP_SCALAR(OP)                               \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_scalar_musa(        \
      at::TensorList self,                                            \
      at::TensorList tensor1,                                         \
      at::TensorList tensor2,                                         \
      const at::Scalar& value) {                                      \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self)); \
    return at::native::foreach_tensor_##OP##_scalar_cuda(             \
        self, tensor1, tensor2, value);                               \
  }                                                                   \
                                                                      \
  void foreach_tensor_##OP##_scalar_musa_(                            \
      at::TensorList self,                                            \
      at::TensorList tensor1,                                         \
      at::TensorList tensor2,                                         \
      const at::Scalar& value) {                                      \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self)); \
    return at::native::foreach_tensor_##OP##_scalar_cuda_(            \
        self, tensor1, tensor2, value);                               \
  }

#define FOREACH_POINTWISE_OP_SCALARLIST(OP)                           \
  ::std::vector<at::Tensor> foreach_tensor_##OP##_scalarlist_musa(    \
      at::TensorList self,                                            \
      at::TensorList tensor1,                                         \
      at::TensorList tensor2,                                         \
      at::ArrayRef<at::Scalar> scalars) {                             \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self)); \
    return at::native::foreach_tensor_##OP##_scalarlist_cuda(         \
        self, tensor1, tensor2, scalars);                             \
  }                                                                   \
                                                                      \
  void foreach_tensor_##OP##_scalarlist_musa_(                        \
      at::TensorList self,                                            \
      at::TensorList tensor1,                                         \
      at::TensorList tensor2,                                         \
      at::ArrayRef<at::Scalar> scalars) {                             \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(self)); \
    return at::native::foreach_tensor_##OP##_scalarlist_cuda_(        \
        self, tensor1, tensor2, scalars);                             \
  }

FOREACH_BINARY_OP_SCALAR(add);
FOREACH_BINARY_OP_SCALAR(mul);
FOREACH_BINARY_OP_LIST(mul);
FOREACH_BINARY_OP_LIST(div);
FOREACH_BINARY_OP_LIST(clamp_min);
FOREACH_BINARY_OP_LIST(clamp_max);
FOREACH_BINARY_OP_SCALARLIST(div);
FOREACH_BINARY_OP_LIST_ALPHA(add);

FOREACH_UNARY_OP(sqrt);

FOREACH_POINTWISE_OP_SCALAR(addcmul);
FOREACH_POINTWISE_OP_SCALAR(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv);

} // anonymous namespace

// binary op register
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_add.Scalar",
    foreach_tensor_add_scalar_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_add_.Scalar",
    foreach_tensor_add_scalar_kernel_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_mul.Scalar",
    foreach_tensor_mul_scalar_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_mul_.Scalar",
    foreach_tensor_mul_scalar_kernel_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_mul_.List",
    foreach_tensor_mul_list_kernel_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_mul.List",
    foreach_tensor_mul_list_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_div_.List",
    foreach_tensor_div_list_kernel_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_div.List",
    foreach_tensor_div_list_kernel_musa)

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_clamp_min.List",
    foreach_tensor_clamp_min_list_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_clamp_min_.List",
    foreach_tensor_clamp_min_list_kernel_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_clamp_max.List",
    foreach_tensor_clamp_max_list_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_clamp_max_.List",
    foreach_tensor_clamp_max_list_kernel_musa_)

REDEFINE_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_maximum.List",
    foreach_tensor_clamp_min_list_kernel_musa)
REDEFINE_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_maximum_.List",
    foreach_tensor_clamp_min_list_kernel_musa_)
REDEFINE_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_minimum.List",
    foreach_tensor_clamp_max_list_kernel_musa)
REDEFINE_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_minimum_.List",
    foreach_tensor_clamp_max_list_kernel_musa_)

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_add.List",
    foreach_tensor_add_list_alpha_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_add_.List",
    foreach_tensor_add_list_alpha_kernel_musa_)

ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_div.ScalarList",
    foreach_tensor_div_scalarlist_kernel_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_div_.ScalarList",
    foreach_tensor_div_scalarlist_kernel_musa_)

// unary op register
ADVANCED_REGISTER(aten, PrivateUse1, "_foreach_sqrt", foreach_tensor_sqrt_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_sqrt_",
    foreach_tensor_sqrt_musa_)

// pointwise op register
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcmul.Scalar",
    foreach_tensor_addcmul_scalar_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcmul_.Scalar",
    foreach_tensor_addcmul_scalar_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcdiv.Scalar",
    foreach_tensor_addcdiv_scalar_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcdiv_.Scalar",
    foreach_tensor_addcdiv_scalar_musa_)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcdiv.ScalarList",
    foreach_tensor_addcdiv_scalarlist_musa)
ADVANCED_REGISTER(
    aten,
    PrivateUse1,
    "_foreach_addcdiv_.ScalarList",
    foreach_tensor_addcdiv_scalarlist_musa_)

} // namespace musa
} // namespace aten
