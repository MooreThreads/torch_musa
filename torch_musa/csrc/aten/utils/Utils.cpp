#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"

#include <mudnn.h>

namespace at {
namespace musa {

void ConfigFormat(Tensor& t, muTensor& mt, bool auto_contiguous) {
  if (t.is_contiguous()) {
    if (t.dim() == 4) {
      mt.SetFormat(muTensor::Format::NCHW);
    } else if (t.dim() == 5) {
      mt.SetFormat(muTensor::Format::NCDHW);
    }
  } else if (t.is_contiguous(at::MemoryFormat::ChannelsLast)) {
    if (t.dim() == 4) {
      mt.SetFormat(muTensor::Format::NHWC);
    } else if (t.dim() == 5) {
      mt.SetFormat(muTensor::Format::NDHWC);
    }
  } else if (auto_contiguous) {
    t = Contiguous(t);
    mt.SetAddr(t.data_ptr());
    if (t.dim() == 4) {
      mt.SetFormat(muTensor::Format::NCHW);
    } else if (t.dim() == 5) {
      mt.SetFormat(muTensor::Format::NCDHW);
    }
  } else {
    TORCH_CHECK(false, "Failed to config MTensor format");
  }
}

inline void SetTensorTypeAndAddr(const Tensor& t, muTensor& m_t) {
  auto t_type = t.scalar_type();
  switch (t_type) {
    case ScalarType::Float:
      m_t.SetType(muTensor::Type::FLOAT);
      break;
    case ScalarType::Int:
      m_t.SetType(muTensor::Type::INT32);
      break;
    case ScalarType::Long:
      m_t.SetType(muTensor::Type::INT64);
      break;
    case ScalarType::Double:
      m_t.SetType(muTensor::Type::DOUBLE);
      break;
    case ScalarType::Bool:
      m_t.SetType(muTensor::Type::BOOL);
      break;
    case ScalarType::Char:
      m_t.SetType(muTensor::Type::BOOL);
      break;
    case ScalarType::Byte:
      m_t.SetType(muTensor::Type::UINT8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype: ", t.dtype());
      throw;
  }
  m_t.SetAddr(t.data_ptr());
}

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

void InternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  c10::musa::raw_delete(ptr);
}

::musa::dnn::MemoryHandler InternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    data = c10::musa::raw_alloc(s);
  }
  return ::musa::dnn::MemoryHandler(data, InternalMemFree);
}

bool is_musa(const Tensor& t) {
  return t.device().type() == kMUSA;
}

} // namespace musa
} // namespace at
