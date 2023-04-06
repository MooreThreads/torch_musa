#include "torch_musa/csrc/core/Allocator.h"
#include <mudnn.h>
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace c10 {

struct C10_API DefaultMTGPUAllocator final : at::Allocator {
  DefaultMTGPUAllocator() = default;

  at::DataPtr allocate(size_t nbytes) const override {
    void* data = nullptr;
    if (nbytes) {
      musa::AutoGrowthBestFitAllocator::get_allocator()->AllocateImpl(
          nbytes, &data);
    }
    int device;
    TORCH_MUSARUNTIME_CHECK(musaGetDevice(&device));
    return {
        data,
        data,
        &ReportAndDelete,
        at::Device(at::native::musa::kMUSA, device)};
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
