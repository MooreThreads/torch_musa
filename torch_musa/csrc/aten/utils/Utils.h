#pragma once

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>

#include <mudnn.h>

namespace at {
namespace native {

using muTensor = ::musa::dnn::Tensor;
using muHandle = ::musa::dnn::Handle;

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

muTensor CreateMUTensor(const Tensor& t, bool use_stride = false);

// use for memory handler
void musaInternalMemFree(void* ptr);
::musa::dnn::MemoryHandler musaInternalMemAlloc(size_t s);

} // namespace native
} // namespace at
