#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/DeviceThreadHandles.h"
#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace {

void CreateMuDNNHandle(mudnnHandle_t* handle) {
#ifdef TORCH_MUSA_USE_MUDNN_C_API
  AT_MUDNN_CHECK(mudnnCreate(handle));
#else
  TORCH_CHECK(handle, "Handle pointer is no-nullptr");
  int device;
  C10_MUSA_CHECK(musaGetDevice(&device));
  TORCH_CHECK(device >= 0);
  *handle = new musa::muHandle(device);
#endif // TORCH_MUSA_USE_MUDNN_C_API
}

void DestroyMuDNNHandle(mudnnHandle_t handle) {
#ifdef TORCH_MUSA_USE_MUDNN_C_API
  AT_MUDNN_CHECK(mudnnDestroy(handle));
#else
  (void)handle;
  // this is because of something dumb in the ordering of
  // destruction. Sometimes atexit, the musa context (or something)
  // would already be destroyed by the time this gets destroyed. It
  // happens in fbcode setting. Not destroy the handle as a workaround.
#endif // TORCH_MUSA_USE_MUDNN_C_API
}

using MudnnPoolType = at::musa::DeviceThreadHandlePool<
    mudnnHandle_t,
    CreateMuDNNHandle,
    DestroyMuDNNHandle>;

} // namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API
mudnnHandle_t& GetMudnnHandle() {
#else
::musa::dnn::Handle& GetMudnnHandle() {
#endif // TORCH_MUSA_USE_MUDNN_C_API
  int device;
  C10_MUSA_CHECK(musaGetDevice(&device));

  // Thread local PoolWindows are lazily-initialized
  // to avoid initialization issues that caused hangs on Windows.
  // See: https://github.com/pytorch/pytorch/pull/22405
  // This thread local unique_ptrs will be destroyed when the thread terminates,
  // releasing its reserved handles back to the pool.
  static auto pool = std::make_shared<MudnnPoolType>();
  thread_local std::unique_ptr<MudnnPoolType::PoolWindow> myPoolWindow(
      pool->NewPoolWindow());

  auto& handle = myPoolWindow->reserve(device);
#ifdef TORCH_MUSA_USE_MUDNN_C_API
  AT_MUDNN_CHECK(mudnnSetStream(handle, c10::musa::getCurrentMUSAStream()));
  return handle;
#else
  CHECK_MUDNN_STATUS(
      handle->SetStream(c10::musa::getCurrentMUSAStream()), "SetStream");
  return *handle;
#endif // TORCH_MUSA_USE_MUDNN_C_API
}

} // namespace at
