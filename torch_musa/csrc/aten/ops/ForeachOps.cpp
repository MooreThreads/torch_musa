#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace musa {

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

#define FOREACH_TERNARY_OP_LIST(OP)                                      \
  std::vector<at::Tensor> foreach_tensor_##OP##_list_kernel_musa(        \
      at::TensorList tensor1,                                            \
      at::TensorList tensor2,                                            \
      at::TensorList tensor3) {                                          \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1)); \
    return at::native::foreach_tensor_##OP##_ternary_cuda(               \
        tensor1, tensor2, tensor3);                                      \
  }                                                                      \
                                                                         \
  void foreach_tensor_##OP##_list_kernel_musa_(                          \
      at::TensorList tensor1,                                            \
      at::TensorList tensor2,                                            \
      at::TensorList tensor3) {                                          \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1)); \
    return at::native::foreach_tensor_##OP##_ternary_cuda_(              \
        tensor1, tensor2, tensor3);                                      \
  }

#define FOREACH_TERNARY_OP_SCALAR(OP)                                         \
  std::vector<at::Tensor> foreach_tensor_##OP##_scalar_kernel_musa(           \
      at::TensorList tensor1, at::TensorList tensor2, const Scalar& weight) { \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1));      \
    return at::native::foreach_tensor_##OP##_list_cuda(                       \
        tensor1, tensor2, weight);                                            \
  }                                                                           \
                                                                              \
  void foreach_tensor_##OP##_scalar_kernel_musa_(                             \
      at::TensorList tensor1, at::TensorList tensor2, const Scalar& weight) { \
    const c10::musa::OptionalMUSAGuard device_guard(device_of(tensor1));      \
    return at::native::foreach_tensor_##OP##_list_cuda_(                      \
        tensor1, tensor2, weight);                                            \
  }

FOREACH_BINARY_OP_SCALAR(add);
FOREACH_BINARY_OP_SCALAR(mul);
FOREACH_BINARY_OP_SCALAR(sub);
FOREACH_BINARY_OP_LIST(mul);
FOREACH_BINARY_OP_LIST(div);
FOREACH_BINARY_OP_LIST(clamp_min);
FOREACH_BINARY_OP_LIST(clamp_max);
FOREACH_BINARY_OP_SCALARLIST(add);
FOREACH_BINARY_OP_SCALARLIST(div);
FOREACH_BINARY_OP_SCALARLIST(mul);
FOREACH_BINARY_OP_SCALARLIST(sub);
FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST_ALPHA(sub);

FOREACH_UNARY_OP(sqrt);
FOREACH_UNARY_OP(sign);
FOREACH_UNARY_OP(reciprocal);

FOREACH_POINTWISE_OP_SCALAR(addcmul);
FOREACH_POINTWISE_OP_SCALAR(addcdiv);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv);

FOREACH_TERNARY_OP_LIST(lerp);
FOREACH_TERNARY_OP_SCALAR(lerp);

void foreach_tensor_copy_list_kernel_musa_(
    TensorList self,
    TensorList src,
    const bool non_blocking) {
  const c10::musa::OptionalMUSAGuard device_guard(device_of(self));
  return at::native::foreach_tensor_copy_list_kernel_cuda_(
      self, src, non_blocking);
}

} // namespace musa
} // namespace at
