#pragma once
#include <c10/core/Allocator.h>
#include <c10/core/DeviceType.h>
#include <musa_runtime_api.h>

namespace c10 {
struct UMACPUAllocator : public Allocator {
  DataPtr allocate(size_t size) override;

  static void raw_delete(void* ctx);

  DeleterFnPtr raw_deleter() const override;

  void copy_data(void* dest, const void* src, std::size_t count) const override;
};

struct UMACPUAllocatorContext {
 public:
  void set_allocator();
  void reset_allocator();

 private:
  Allocator* prev_allocator_ptr_{nullptr};
};

inline static UMACPUAllocatorContext uma_cpu_alloc_context;

} // namespace c10
