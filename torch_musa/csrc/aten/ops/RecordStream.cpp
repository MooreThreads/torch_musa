#include <ATen/TensorUtils.h>
#include <c10/core/Stream.h>
#include <torch/library.h>

#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {

void record_stream(at::Tensor& self, at::Stream s) {
  const MUSAGuard device_guard(self.device());
  struct c10::StreamData3 data = s.pack3();
  c10::musa::MUSACachingAllocator::recordStream(
      self.storage().data_ptr(),
      at::musa::MUSAStream::unpack3(
          data.stream_id, data.device_index, data.device_type));
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("record_stream", &record_stream);
}
} // namespace musa
} // namespace at
