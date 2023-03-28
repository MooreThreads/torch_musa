#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <c10/core/CPUAllocator.h>
#include <c10/core/DeviceType.h>
#pragma GCC diagnostic pop

#include <mudnn.h>
#include <torch_musa/csrc/core/Allocator.h>
#include "torch_musa/csrc/aten/utils/Utils.h"

using ::musa::dnn::Tensor;

namespace c10 {

struct C10_API DefaultMTGPUAllocator final : at::Allocator {
  DefaultMTGPUAllocator() = default;

  at::DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
    if (nbytes) {
      musa::AutoGrowthBestFitAllocator::get_allocator()->AllocateImpl(
          nbytes, &data);
    }
    // TODO(songtao.liu): complete the device index selection for distributed
    // training.
    return {
        data, data, &ReportAndDelete, at::Device(at::native::musa::kMUSA, 0)};
  }

  static void ReportAndDelete(void* ptr) {
    if (!ptr) {
      return;
    }
    musa::AutoGrowthBestFitAllocator::get_allocator()->FreeImpl(ptr);
  }

  at::DeleterFnPtr raw_deleter() const override {
    return &ReportAndDelete;
  }
};

static DefaultMTGPUAllocator g_mtgpu_alloc;

at::Allocator* GetDefaultMTGPUAllocator() {
  return &g_mtgpu_alloc;
}

REGISTER_ALLOCATOR(at::native::musa::kMUSA, &g_mtgpu_alloc);
} // namespace c10
