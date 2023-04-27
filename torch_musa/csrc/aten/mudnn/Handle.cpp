#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/DeviceThreadHandles.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {
namespace {

void CreateMuDNNHandle(mudnnHandle_t* handle) {
  TORCH_CHECK(handle, "Handle pointer is no-nullptr");
  int device;
  TORCH_MUSA_CHECK(musaGetDevice(&device));
  TORCH_CHECK(device >= 0);
  *handle = new musa::muHandle(device);
}

void DestroyMuDNNHandle(mudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
}

using MudnnPoolType = at::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;

} // namespace

::musa::dnn::Handle& GetMudnnHandle() {
  int device;
  TORCH_MUSA_CHECK(musaGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  mudnnHandle_t handle = myPoolWindow->reserve(device);
  handle->SetStream(torch_musa::getCurrentMUSAStream());
  return *handle;
}

} // namespace native
} // namespace at
