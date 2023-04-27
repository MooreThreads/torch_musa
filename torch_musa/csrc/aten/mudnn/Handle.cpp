#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/DeviceThreadHandles.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {
namespace {

void createMuDNNHandle(mudnnHandle_t* handle) {
  TORCH_CHECK(handle, "Handle pointer is no-nullptr");
  int device;
  TORCH_MUSA_CHECK(musaGetDevice(&device));
  TORCH_CHECK(device >= 0);
  *handle = new musa::muHandle(device);
  /* TORCH_MUDNN_CHECK(mudnnCreate(handle)); */
}

void destroyMuDNNHandle(mudnnHandle_t /*handle*/) {
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
}

using MudnnPoolType = at::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    createMuDNNHandle,
    destroyMuDNNHandle>;

} // namespace

::musa::dnn::Handle& getMudnnHandle() {
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
  /* TORCH_MUDNN_CHECK(mudnnSetStream(handle,
   * torch_musa::getCurrentMUSAStream())); */
  return *handle;
}

} // namespace native
} // namespace at
