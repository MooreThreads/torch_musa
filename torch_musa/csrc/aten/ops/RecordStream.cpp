#include <ATen/TensorUtils.h>
#include <c10/core/Stream.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

void RecordStream(at::Tensor& self, at::Stream s) {
  const MUSAGuard device_guard(self.device());
  struct c10::StreamData3 data = s.pack3();
  c10::musa::MUSACachingAllocator::recordStream(
      self.storage().data_ptr(),
      at::musa::MUSAStream::unpack3(
          data.stream_id, data.device_index, data.device_type));
}

} // namespace musa
} // namespace at
