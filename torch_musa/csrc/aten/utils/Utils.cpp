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
#include "torch_musa/csrc/aten/utils/MudnnUtils.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"

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
    case ScalarType::UInt16:
      m_t.SetType(muTensor::Type::UINT16);
      break;
    case ScalarType::UInt32:
      m_t.SetType(muTensor::Type::UINT32);
      break;
    case ScalarType::UInt64:
      m_t.SetType(muTensor::Type::UINT64);
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
// musa-4.0.0 or later
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 4000
    case ScalarType::Float8_e5m2:
      m_t.SetType(muTensor::Type::FP8_E5M2);
      break;
    case ScalarType::Float8_e4m3fn:
      m_t.SetType(muTensor::Type::FP8_E4M3);
      break;
#endif
    default:
      TORCH_CHECK(false, "SetMUTensorDType Unsupported tensor dtype: ", dtype);
  }
}

void SetMUTensorAddr(void* addr, muTensor& m_t) {
  m_t.SetAddr(addr);
}

void SetMudnnQuantizationInfo(
    muTensor& self,
    double scales,
    int64_t zero_points) {
  TORCH_CHECK(
      0 == zero_points,
      "torch_musa only supports symmetric quantization",
      "which requires zero_point == 0, got: ",
      zero_points);
  float scales_ = static_cast<float>(scales);
  CHECK_MUDNN_STATUS(self.SetQuantizationInfo(scales_), "SetQuantizationInfo");
}

std::pair<muTensor, muTensor> CreateMUTensorsCompression(
    const Tensor& t1,
    const Tensor& t2) {
  TORCH_CHECK(
      t1.sizes().equals(t2.sizes()), "Input tensors must have the same shape");
  TORCH_CHECK(t1.dim() > 8, "Now only compress tensors whose dim > 8");

  // init reverse shapes and strides, easier to compress.
  DimVector shape1(t1.sizes().rbegin(), t1.sizes().rend());
  DimVector stride1(t1.strides().rbegin(), t1.strides().rend());
  DimVector shape2(t2.sizes().rbegin(), t2.sizes().rend());
  DimVector stride2(t2.strides().rbegin(), t2.strides().rend());

  int prev_dim = 0;
  const int total_dims = t1.dim();

  for (int dim = 1; dim < total_dims; ++dim) {
    const auto& shape0 = shape1[prev_dim];
    const auto& shape1_val = shape1[dim];

    bool can_compress = false;
    if (shape0 == 1 || shape1_val == 1) {
      can_compress = true;
    } else {
      // both tensors satisfy the compression condition
      can_compress = (shape0 * stride1[prev_dim] == stride1[dim]) &&
          (shape0 * stride2[prev_dim] == stride2[dim]);
    }

    if (can_compress) {
      if (shape1[prev_dim] == 1) {
        stride1[prev_dim] = stride1[dim];
        stride2[prev_dim] = stride2[dim];
      }
      shape1[prev_dim] *= shape1[dim];
      shape2[prev_dim] *= shape2[dim];
    } else {
      ++prev_dim;
      if (prev_dim != dim) {
        stride1[prev_dim] = stride1[dim];
        shape1[prev_dim] = shape1[dim];
        stride2[prev_dim] = stride2[dim];
        shape2[prev_dim] = shape2[dim];
      }
    }
  }

  const int ndim = prev_dim + 1;
  // adjust to the compressed dim.
  shape1.resize(ndim);
  stride1.resize(ndim);
  shape2.resize(ndim);
  stride2.resize(ndim);

  std::reverse(shape1.begin(), shape1.end());
  std::reverse(stride1.begin(), stride1.end());
  std::reverse(shape2.begin(), shape2.end());
  std::reverse(stride2.begin(), stride2.end());

  muTensor rst1, rst2;
  SetMUTensorDType(t1.scalar_type(), rst1);
  SetMUTensorAddr(t1.data_ptr(), rst1);
  rst1.SetNdInfo(ndim, shape1.data(), stride1.data());

  SetMUTensorDType(t2.scalar_type(), rst2);
  SetMUTensorAddr(t2.data_ptr(), rst2);
  rst2.SetNdInfo(ndim, shape2.data(), stride2.data());

  return {std::move(rst1), std::move(rst2)};
}

muTensor CreateMUTensor(const Tensor& t, bool permute_if_not_contiguous) {
  if (!t.defined()) {
    return muTensor();
  }
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

bool IsContiguousAfterTranspose(const Tensor& mat, bool strict) {
  // mat.dim() should be >= 2, and this is guaranteed by the
  // caller(IsTranspose), so we do not check it here.
  int64_t d = mat.dim();
  std::vector<int64_t> t_size(mat.sizes().begin(), mat.sizes().end());
  std::vector<int64_t> t_stride(mat.strides().begin(), mat.strides().end());
  std::swap(t_size[d - 2], t_size[d - 1]);
  std::swap(t_stride[d - 2], t_stride[d - 1]);
  int64_t expected_stride = 1;
  for (int64_t i = d - 1; i >= 0; --i) {
    const auto& size_i = t_size[i];
    if (!strict && size_i == 1) {
      continue;
    }

    if (t_stride[i] != expected_stride) {
      return false;
    }
    expected_stride *= size_i;
  }

  return true;
}

// If a matrix is ​​transposed, the following two conditions
// need to be met
// 1. stride(i)=stride(i+1)*shape(i+1) for the origin matrix
// 2. the origin matrix(untransposed matrix) should be contiguous
bool IsTranspose(const Tensor& mat, bool strict) {
  if (mat.dim() >= 2) {
    bool t_mat_contiguous_loose = IsContiguousAfterTranspose(mat, false);
    if (!strict && t_mat_contiguous_loose && mat.is_contiguous()) {
      return false;
    }
    return strict ? IsContiguousAfterTranspose(mat, true)
                  : t_mat_contiguous_loose;
  }
  return false;
}

bool IsLastDimContiguous(const Tensor& input) {
  return input.dim() > 0 && input.stride(-1) == 1;
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
    case at::ScalarType::ComplexFloat:
      size = 8;
      break;
    case at::ScalarType::ComplexDouble:
      size = 16;
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
