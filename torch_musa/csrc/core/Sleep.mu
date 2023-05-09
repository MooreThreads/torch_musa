#include "torch_musa/csrc/core/Sleep.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "musa_runtime_api.h"


namespace at {
namespace musa {
namespace {
__global__ void spin_kernel(int64_t cycles) {
  // see concurrentKernels MUSA sampl
  int64_t start_clock = clock64();
  int64_t clock_offset = 0;
  while (clock_offset < cycles)
  {
    // TODO(MT-AI): need https://jira.mthreads.com/browse/SW-21640 fixed.
    /* clock_offset = clock64() - start_clock; */
    clock_offset += 1;
  }
}
}

void sleep(int64_t cycles) {
  dim3 grid(1);
  dim3 block(1);
  spin_kernel<<<grid, block, 0, c10::musa::getCurrentMUSAStream()>>>(cycles);
  TORCH_MUSA_CHECK(musaGetLastError());
}

}}  // namespace at::musa
