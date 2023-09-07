#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/extension.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"

#include <mudnn.h>

namespace at {
namespace musa {
namespace {
Tensor empty_strided_musa(
    IntArrayRef size,
    IntArrayRef stride,
    const TensorOptions& options) {
  return at::musa::empty_strided_musa(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}
} // namespace

void ConfigFormat(const Tensor& t, muTensor& mt) {
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
  }
}

inline void SetTensorTypeAndAddr(const Tensor& t, muTensor& m_t) {
  auto t_type = t.scalar_type();
  switch (t_type) {
    case ScalarType::Half:
      m_t.SetType(muTensor::Type::HALF);
      break;
    case ScalarType::Float:
      m_t.SetType(muTensor::Type::FLOAT);
      break;
    case ScalarType::Short:
      m_t.SetType(muTensor::Type::INT16);
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
    case ScalarType::QInt32:
      m_t.SetType(muTensor::Type::INT32);
      break;
    case ScalarType::QUInt8:
      m_t.SetType(muTensor::Type::UINT8);
      break;
    case ScalarType::QInt8:
      m_t.SetType(muTensor::Type::QINT8);
      break;
    default:
      TORCH_CHECK(false, "Unsupported tensor dtype: ", t.dtype());
      throw;
  }
  m_t.SetAddr(t.data_ptr());
}

muTensor CreateMUTensor(const Tensor& t) {
  muTensor rst;
  rst.SetNdInfo(t.dim(), t.sizes().data(), t.strides().data());
  SetTensorTypeAndAddr(t, rst);
  ConfigFormat(t, rst);
  return rst;
}

void InternalMemFree(void* ptr) {
  if (!ptr) {
    return;
  }
  c10::musa::MUSACachingAllocator::raw_delete(ptr);
}

::musa::dnn::MemoryHandler InternalMemAlloc(size_t s) {
  void* data = nullptr;
  if (s) {
    data = c10::musa::MUSACachingAllocator::raw_alloc(s);
  }
  return ::musa::dnn::MemoryHandler(data, InternalMemFree);
}

bool is_musa(const Tensor& t) {
  return t.device().type() == kMUSA;
}

c10::optional<Tensor> maybe_create_proxy(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options) {
  if (out.strides() != strides) {
    return empty_strided_musa(sizes, strides, options);
  }
  return c10::nullopt;
}

} // namespace musa
} // namespace at
