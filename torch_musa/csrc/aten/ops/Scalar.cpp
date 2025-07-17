#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <torch/library.h>

#include "ATen/EmptyTensor.h"
#include "ATen/native/TensorFactories.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <mudnn.h>

namespace at {
namespace musa {

Scalar LocalScalarDense_(const Tensor& self) {
  Scalar r;
  c10::musa::MUSAGuard device_guard(self.device());
  AT_DISPATCH_V2(
      self.scalar_type(),
      "LocalScalarDense_",
      AT_WRAP([&] {
        auto value = at::detail::empty_cpu(
            {1}, /* size */
            c10::CppTypeToScalarType<scalar_t>(), /* dtype */
            std::nullopt, /* layout */
            std::nullopt, /* device */
            true, /* pin_memory */
            std::nullopt /* memory format */
        );
        musaStream_t stream = at::musa::getCurrentMUSAStream();
        at::musa::memcpy_and_sync(
            (void*)value.const_data_ptr<scalar_t>(),
            self.const_data_ptr<scalar_t>(),
            sizeof(scalar_t),
            musaMemcpyDeviceToHost,
            stream);
        r = Scalar(*value.const_data_ptr<scalar_t>());
      }),
      AT_EXPAND(AT_ALL_TYPES_AND_COMPLEX),
      kComplexHalf,
      kHalf,
      kBool,
      kBFloat16,
      AT_EXPAND(AT_BAREBONES_UNSIGNED_TYPES));

  return r;
}

} // namespace musa
} // namespace at
