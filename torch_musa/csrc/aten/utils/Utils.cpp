#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wunused-variable"
#pragma GCC diagnostic ignored "-Wunused"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#pragma GCC diagnostic pop

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

#include "torch_musa/csrc/c10/Allocator.h"

#include <mudnn.h>

namespace at {
namespace native {

void ConfigFormat(Tensor& t, muTensor& mt, bool auto_contiguous) {
  if (t.is_contiguous()) {
    if (t.dim() == 4) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NCHW);
    } else if (t.dim() == 5) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NCDHW);
    }
  } else if (t.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    if (t.dim() == 4) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NHWC);
    } else if (t.dim() == 5) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NDHWC);
    }
  } else if (auto_contiguous) {
    t = MusaContiguous(t);
    mt.SetAddr(t.data_ptr());
    if (t.dim() == 4) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NCHW);
    } else if (t.dim() == 5) {
      mt.SetFormat(::musa::dnn::Tensor::Format::NCDHW);
    }
  } else {
    TORCH_CHECK(false, "Failed to config MTensor format");
  }
}

namespace {
inline void SetTensorTypeAndAddr(const Tensor& t, muTensor& m_t) {
  auto offset = t.storage_offset();
  auto t_type = t.scalar_type();
  switch (t_type) {
    case ScalarType::Float:
      m_t.SetType(muTensor::Type::FLOAT);
      m_t.SetAddr(reinterpret_cast<float*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Int:
      m_t.SetType(muTensor::Type::INT32);
      m_t.SetAddr(reinterpret_cast<int*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Long:
      m_t.SetType(muTensor::Type::INT64);
      m_t.SetAddr(reinterpret_cast<int64_t*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Double:
      m_t.SetType(muTensor::Type::DOUBLE);
      m_t.SetAddr(reinterpret_cast<double*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Bool:
      m_t.SetType(muTensor::Type::BOOL);
      m_t.SetAddr(reinterpret_cast<bool*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Char:
      m_t.SetType(muTensor::Type::BOOL);
      m_t.SetAddr(reinterpret_cast<char*>(t.data_ptr()) - offset);
      return;
    case ScalarType::Byte:
      m_t.SetType(muTensor::Type::UINT8);
      m_t.SetAddr(reinterpret_cast<uint8_t*>(t.data_ptr()) - offset);
      return;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype: ", t);
      throw;
  }
}

} // namespace

muTensor CreateMUTensor(const Tensor& t, bool use_stride) {
  muTensor rst;
  if (use_stride) {
    rst.SetNdInfo(t.dim(), t.sizes().data(), t.strides().data());
  } else {
    rst.SetNdInfo(t.dim(), t.sizes().data());
  }
  SetTensorTypeAndAddr(t, rst);
  return rst;
}

muTensor CreateEmptyMUTensor() {
  muTensor rst;
  return rst;
}

void musaInternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  musa::AutoGrowthBestFitAllocator::get_allocator()->FreeImpl(ptr);
}

::musa::dnn::MemoryHandler musaInternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    musa::AutoGrowthBestFitAllocator::get_allocator()->AllocateImpl(s, &data);
  }
  return ::musa::dnn::MemoryHandler(data, musaInternalMemFree);
}

void Synchronize() {
  musaDeviceSynchronize();
}

} // namespace native
} // namespace at
