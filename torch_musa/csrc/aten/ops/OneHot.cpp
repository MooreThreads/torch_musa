#include <ATen/ATen.h>
#include <ATen/Config.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty.h>
#include <ATen/ops/one_hot_native.h>
#endif

#include "torch_musa/csrc/aten/ops/musa/OneHot.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at::musa {

Tensor OneHot(const Tensor& self, int64_t num_classes) {
  TORCH_CHECK(
      self.scalar_type() == at::ScalarType::Long,
      "one_hot is only applicable to index tensor of type LongTensor.");

  c10::musa::MUSAGuard device_guard(self.device());
  auto shape = self.sizes().vec();

  // empty tensor could be converted to on hot representation,
  // but shape inference is not possible.
  if (self.numel() == 0) {
    if (num_classes <= 0) {
      AT_ERROR("Can not infer total number of classes from empty tensor.");
    } else {
      shape.push_back(num_classes);
      return at::empty(shape, self.options());
    }
  }

  if (num_classes == -1) {
    num_classes = self.max().item().toLong() + 1;
  } else {
    TORCH_CHECK(num_classes >= 1, "num_classes should be positive");
  }

  shape.push_back(num_classes);
  Tensor ret = at::empty(
      shape, self.options().memory_format(c10::MemoryFormat::Contiguous));

  // onehot kernel dosen't support un-contiguous tensors currently
  Tensor self_contiguous = self.contiguous();

  OneHotRun(ret, self_contiguous, num_classes);

  return ret;
}

} // namespace at::musa
