#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <mudnn.h>
#include <string>

namespace at {
namespace native {
namespace musa {

using muTensor = ::musa::dnn::Tensor;
using muHandle = ::musa::dnn::Handle;

constexpr DeviceType kMUSA = DeviceType::PrivateUse1;
constexpr ::c10::DispatchKey kMUSAKey = ::c10::DispatchKey::PrivateUse1;

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

muTensor CreateMUTensor(const Tensor& t, bool use_stride = false);

// use for memory handler
void internalMemFree(void* ptr);
::musa::dnn::MemoryHandler internalMemAlloc(size_t s);

} // namespace musa
} // namespace native
} // namespace at
