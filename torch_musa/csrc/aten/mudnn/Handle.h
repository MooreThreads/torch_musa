#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H_

#include <functional>
#include <memory>

#include <mudnn.h>

namespace at {

#ifdef TORCH_MUSA_USE_MUDNN_C_API
typedef ::mudnnHandle_t mudnnHandle_t;
mudnnHandle_t& GetMudnnHandle();
#else
typedef ::musa::dnn::Handle* mudnnHandle_t;
::musa::dnn::Handle& GetMudnnHandle();
#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API
typedef mudnnHandle_t muHandle;
#else
typedef ::musa::dnn::Handle muHandle;
#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace musa

} // namespace at

#ifdef TORCH_MUSA_USE_MUDNN_C_API

namespace musa::dnn {
using MemoryHandler = ::std::unique_ptr<void, ::std::function<void(void*)>>;
using MemoryMaintainer = ::std::function<MemoryHandler(size_t)>;
} // namespace musa::dnn

namespace at::musa {
using ::musa::dnn::MemoryHandler;
using ::musa::dnn::MemoryMaintainer;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API

namespace at::musa {
::musa::dnn::MemoryHandler InternalMemAlloc(size_t s);
} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_HANDLE_H_
