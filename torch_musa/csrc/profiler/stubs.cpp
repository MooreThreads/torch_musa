#include <sstream>

#include <c10/util/ApproximateClock.h>
#include <c10/util/irange.h>
#include <torch/csrc/profiler/stubs/base.h>
#include <torch/csrc/profiler/util.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace torch {
namespace profiler {
namespace impl {
namespace {

static inline void musaCheck(musaError_t result, const char* file, int line) {
  if (result != musaSuccess) {
    std::stringstream ss;
    ss << file << ":" << line << ": ";
    if (result == musaErrorInitializationError) {
      // It is common for users to use DataLoader with multiple workers
      // and the autograd profiler. Throw a nice error message here.
      ss << "MUSA initialization error. "
         << "This can occur if one runs the profiler in MUSA mode on code "
         << "that creates a DataLoader with num_workers > 0. This operation "
         << "is currently unsupported; potential workarounds are: "
         << "(1) don't use the profiler in MUSA mode or (2) use num_workers=0 "
         << "in the DataLoader or (3) Don't profile the data loading portion "
         << "of your code. https://github.com/pytorch/pytorch/issues/6313 "
         << "tracks profiler support for multi-worker DataLoader.";
    } else {
      ss << musaGetErrorString(result);
    }
    throw std::runtime_error(ss.str());
  }
}
#define TORCH_MUSA_STUBS_CHECK(result) musaCheck(result, __FILE__, __LINE__);

struct MUSAMethods : public ProfilerStubs {
  void record(
      c10::DeviceIndex* device,
      ProfilerVoidEventStub* event,
      int64_t* cpu_ns) const override {
    if (device) {
      TORCH_MUSA_STUBS_CHECK(c10::musa::GetDevice(device));
    }
    MUevent_st* musa_event_ptr{nullptr};
    TORCH_MUSA_STUBS_CHECK(musaEventCreate(&musa_event_ptr));
    *event = std::shared_ptr<MUevent_st>(musa_event_ptr, [](MUevent_st* ptr) {
      TORCH_MUSA_STUBS_CHECK(musaEventDestroy(ptr));
    });
    auto stream = at::musa::getCurrentMUSAStream();
    if (cpu_ns) {
      *cpu_ns = c10::getTime();
    }
    TORCH_MUSA_STUBS_CHECK(musaEventRecord(musa_event_ptr, stream));
  }

  float elapsed(
      const ProfilerVoidEventStub* event_,
      const ProfilerVoidEventStub* event2_) const override {
    auto event = (const ProfilerEventStub*)(event_);
    auto event2 = (const ProfilerEventStub*)(event2_);
    TORCH_MUSA_STUBS_CHECK(musaEventSynchronize(event->get()));
    TORCH_MUSA_STUBS_CHECK(musaEventSynchronize(event2->get()));
    float ms = 0;
    TORCH_MUSA_STUBS_CHECK(
        musaEventElapsedTime(&ms, event->get(), event2->get()));
    // NOLINTNEXTLINE(bugprone-narrowing-conversions,cppcoreguidelines-avoid-magic-numbers,cppcoreguidelines-narrowing-conversions)
    return ms * 1000.0;
  }

  void mark(const char* name) const override {}

  void rangePush(const char* name) const override {}

  void rangePop() const override {}

  void onEachDevice(std::function<void(int)> op) const override {
    at::musa::OptionalMUSAGuard device_guard;
    for (const auto i : c10::irange(at::musa::device_count())) {
      device_guard.set_index(i);
      op(i);
    }
  }

  void synchronize() const override {
    TORCH_MUSA_STUBS_CHECK(musaDeviceSynchronize());
  }

  bool enabled() const override {
    return true;
  }
};

struct RegisterPrivateUse1Methods {
  RegisterPrivateUse1Methods() {
    static MUSAMethods methods;
    registerPrivateUse1Methods(&methods);
  }
};
RegisterPrivateUse1Methods reg;

} // namespace
} // namespace impl
} // namespace profiler
} // namespace torch
