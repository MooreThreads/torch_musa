#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>
#include <ATen/native/Resize.h>
#include <c10/core/ScalarTypeToTypeMeta.h>
#include <torch/extension.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/as_strided.h>
#endif

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
  return at::musa::EmptyStridedMUSA(
      size,
      stride,
      optTypeMetaToScalarType(options.dtype_opt()),
      options.layout_opt(),
      options.device_opt(),
      options.pinned_memory_opt());
}
} // namespace

void ConfigFormat(
    const Tensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous) {
  TORCH_CHECK(
      t.dim() <= 8,
      "mudnn only support intput tensors'dim <= 8, but it is ",
      t.dim());
  const auto t_dim = t.dim();
  const auto memory_format = t.suggest_memory_format();
  muTensor::Format mudnn_format = muTensor::Format::NCHW;
  Tensor mu_t = t;

  if (memory_format == at::MemoryFormat::Contiguous) {
    if (t_dim == 4) {
      mudnn_format = muTensor::Format::NCHW;
    } else if (t_dim == 5) {
      mudnn_format = muTensor::Format::NCDHW;
    }
  } else if (memory_format == at::MemoryFormat::ChannelsLast) {
    if (t_dim == 4) {
      mudnn_format = muTensor::Format::NHWC;
      if (permute_if_not_contiguous) {
        mu_t = t.transpose(-3, -1).transpose(-3, -2);
      }
    }
  } else {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(
        memory_format == at::MemoryFormat::ChannelsLast3d);
    if (t_dim == 5) {
      mudnn_format = muTensor::Format::NDHWC;
      if (permute_if_not_contiguous) {
        mu_t = t.transpose(-4, -1).transpose(-4, -2).transpose(-4, -3);
      }
    }
  }

  mt.SetFormat(mudnn_format);
  mt.SetNdInfo(mu_t.dim(), mu_t.sizes().data(), mu_t.strides().data());
}

void SetMUTensorDType(ScalarType dtype, muTensor& m_t) {
  switch (dtype) {
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
      m_t.SetType(muTensor::Type::INT8);
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
    case ScalarType::BFloat16:
      m_t.SetType(muTensor::Type::BFLOAT16);
      break;
#if defined(TORCH_MUSA_ARCH) && TORCH_MUSA_ARCH >= 310
    case ScalarType::Float8_e5m2:
      m_t.SetType(muTensor::Type::FP8_E5M2);
      break;
    case ScalarType::Float8_e4m3fn:
      m_t.SetType(muTensor::Type::FP8_E4M3);
      break;
#endif
    default:
      TORCH_CHECK(false, "SetMUTensorDType Unsupported tensor dtype: ", dtype);
      throw;
  }
}

void SetMUTensorAddr(void* addr, muTensor& m_t) {
  m_t.SetAddr(addr);
}

void SetMudnnQuantizationInfo(
    muTensor& self,
    double scales,
    int64_t zero_points) {
  float scales_ = static_cast<float>(scales);
  unsigned int zero_points_ = static_cast<unsigned int>(zero_points);
  CHECK_MUDNN_STATUS(
      self.SetQuantizationInfo(1, &scales_, &zero_points_),
      "SetQuantizationInfo");
}

void SetMudnnQuantizationInfo(muTensor& self, Tensor& scales) {
  // TODO(@fan.mo): we currently only support set per-tensor scale
  if (scales.numel() == 1) {
    float* scale = static_cast<float*>(scales.data_ptr());
    CHECK_MUDNN_STATUS(
        self.SetQuantizationInfo({scale[0]}, {0}), "SetQuantizationInfo");
  }
}

muTensor CreateMUTensorByCompressDim(const Tensor& t) {
  // mudnn only support dim <= 8, need to compress the shapes
  TORCH_CHECK(t.dim() > 8, "Now only compress the tensor whose dim > 8");
  muTensor rst;
  SetMUTensorDType(t.scalar_type(), rst);
  SetMUTensorAddr(t.data_ptr(), rst);
  // init reverse shapes and strides, easier to compress.
  DimVector shape(t.sizes().rbegin(), t.sizes().rend());
  DimVector stride(t.strides().rbegin(), t.strides().rend());

  auto can_compress = [&](int dim0, int dim1) {
    auto shape0 = shape[dim0];
    auto shape1 = shape[dim1];
    if (shape0 == 1 || shape1 == 1) {
      return true;
    }
    if (shape0 * stride[dim0] != stride[dim1]) {
      return false;
    }
    return true;
  };

  int prev_dim = 0;
  for (const auto dim : c10::irange(1, t.dim())) {
    if (can_compress(prev_dim, dim)) {
      if (shape[prev_dim] == 1) {
        stride[prev_dim] = stride[dim];
      }
      shape[prev_dim] *= shape[dim];
    } else {
      prev_dim++;
      if (prev_dim != dim) {
        stride[prev_dim] = stride[dim];
        shape[prev_dim] = shape[dim];
      }
    }
  }
  int ndim = prev_dim + 1;
  TORCH_CHECK(ndim <= 8, "mudnn only support dim <= 8, but it is ", ndim);
  shape.resize(ndim);
  stride.resize(ndim);
  // reverse back
  std::reverse(shape.begin(), shape.end());
  std::reverse(stride.begin(), stride.end());
  rst.SetNdInfo(ndim, shape.data(), stride.data());
  return rst;
}

muTensor CreateMUTensor(const Tensor& t, bool permute_if_not_contiguous) {
  muTensor rst;
  SetMUTensorDType(t.scalar_type(), rst);
  SetMUTensorAddr(t.data_ptr(), rst);
  ConfigFormat(t, rst, permute_if_not_contiguous);
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

bool MatContiguous(const Tensor& mat) {
  for (int i = 0; i < mat.dim() - 1; i++) {
    if (mat.stride(i) != mat.stride(i + 1) * mat.size(i + 1)) {
      return false;
    }
  }
  return mat.is_contiguous();
}

// If a matrix is ​​transposed, the following two conditions
// need to be met
// 1. stride(i)=stride(i+1)*shape(i+1) for the origin matrix
// 2. the origin matrix(untransposed matrix) should be contiguous
bool IsTranspose(const Tensor& mat, bool strict) {
  if (mat.dim() >= 2) {
    const Tensor t_mat = mat.transpose(-2, -1);
    return strict ? MatContiguous(t_mat) : t_mat.is_contiguous();
  }
  return false;
}

Tensor FormatContiguous(const Tensor& t, at::MemoryFormat memory_format) {
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(memory_format != at::MemoryFormat::Preserve);
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(t.defined());
  Tensor contig_t;
  if (t.is_contiguous(memory_format)) {
    contig_t = t;
    contig_t.unsafeGetTensorImpl()->empty_tensor_restride(memory_format);
  } else {
    contig_t = t.contiguous(memory_format);
  }
  // Retain dim && memory_format verification
  return contig_t;
}

size_t DTypeSize(c10::ScalarType type) {
  size_t size;
  switch (type) {
    case at::ScalarType::Bool:
    case at::ScalarType::Char:
    case at::ScalarType::Byte:
    case at::ScalarType::QInt8:
    case at::ScalarType::QUInt8:
    case at::ScalarType::Float8_e5m2:
    case at::ScalarType::Float8_e4m3fn:
      size = 1;
      break;
    case at::ScalarType::Half:
    case at::ScalarType::BFloat16:
    case at::ScalarType::Short:
      size = 2;
      break;
    case at::ScalarType::Int:
    case at::ScalarType::Float:
    case at::ScalarType::QInt32:
      size = 4;
      break;
    case at::ScalarType::Long:
    case at::ScalarType::Double:
      size = 8;
      break;
    default:
      TORCH_CHECK(false, "DTypeSize Unsupported tensor dtype: ", type);
  }
  return size;
}

at::Tensor ContiguousIfZeroInStrides(const at::Tensor& t) {
  for (auto s : t.strides()) {
    if (s == 0) {
      return t.contiguous();
    }
  }
  return t;
}

} // namespace musa
} // namespace at
