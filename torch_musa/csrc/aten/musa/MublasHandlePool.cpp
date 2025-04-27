#include "torch_musa/csrc/aten/musa/MUSAContext.h"

#include <regex>

#include <ATen/musa/detail/DeviceThreadHandles.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

namespace at {
namespace musa {

namespace {

std::map<std::tuple<void*, void*>, at::DataPtr>&
mublas_handle_stream_to_workspace() {
  static auto& instance = *new std::map<std::tuple<void*, void*>, at::DataPtr>;
  return instance;
}

void createMublasHandle(mublasHandle_t* handle) {
  TORCH_MUSABLAS_CHECK(mublasCreate(handle));
}

void destroyMublasHandle(mublasHandle_t handle) {
  mublasDestroy(handle);
}

using MuBlasPoolType = DeviceThreadHandlePool<
    mublasHandle_t,
    createMublasHandle,
    destroyMublasHandle>;

} // namespace

// TODO(MTAI): mublas_set_workspace() is not supported by MUBLAS now!
// void clearMublasWorkspaces() {
//   mublas_handle_stream_to_workspace().clear();
// }

// size_t parseChosenWorkspaceSize() {
//   const size_t default_size = 4096 * 8 * 1024;
//   return default_size;
// }

// size_t getChosenWorkspaceSize() {
//   size_t pool_size = parseChosenWorkspaceSize();
//   return pool_size;
// }

// at::DataPtr getNewWorkspace() {
//   return c10::musa::MUSACachingAllocator::get()->allocate(
//       getChosenWorkspaceSize());
// }
// TODO(MTAI):END

mublasHandle_t getCurrentMUSABlasHandle() {
  int device = 0;
  AT_MUSA_CHECK(at::musa::GetDevice(&device));

  // Use a leaky singleton for the pool following standard practice around
  // singletons: https://isocpp.org/wiki/faq/ctors#construct-on-first-use-v2
  static auto pool = std::shared_ptr<MuBlasPoolType>(
      new MuBlasPoolType(), [](MuBlasPoolType* p) {
        // Leak the memory.
      });
  thread_local std::unique_ptr<MuBlasPoolType::PoolWindow> myPoolWindow(
      pool->newPoolWindow());

  auto handle = myPoolWindow->reserve(device);
  auto stream = c10::musa::getCurrentMUSAStream();
  TORCH_MUSABLAS_CHECK(mublasSetStream(handle, stream));
  musaStream_t key_stream = stream;
  auto key = std::make_tuple(
      static_cast<void*>(handle), static_cast<void*>(key_stream));

  // TODO(MTAI): mublas_set_workspace() is not supported by MUBLAS now!
  // auto workspace_it = mublas_handle_stream_to_workspace().find(key);
  // if (workspace_it == mublas_handle_stream_to_workspace().end()) {
  //   workspace_it = mublas_handle_stream_to_workspace().insert(
  //       workspace_it, {key, getNewWorkspace()});
  // }
  // TORCH_MUSABLAS_CHECK(mublas_set_workspace(
  //     handle, workspace_it->second.get(), getChosenWorkspaceSize()));

  // TODO(MTAI): TF32 is not supported by MUBLAS now!
  // if (!NoTF32Guard::should_disable_tf32() &&
  //  at::globalContext().allowTF32MuBLAS()) {
  //   TORCH_MUSABLAS_CHECK(mublasSetMathMode(handle,
  //   MUBLAS_MATH_MODE_TP32_TENSOR));
  // }

  TORCH_MUSABLAS_CHECK(mublasSetMathMode(handle, MUBLAS_MATH_MODE_DEFAULT));
  return handle;
}

} // namespace musa
} // namespace at
