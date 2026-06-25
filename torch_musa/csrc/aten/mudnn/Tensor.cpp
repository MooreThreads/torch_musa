#include "torch_musa/csrc/aten/mudnn/Tensor.h"

#include "torch_musa/csrc/aten/mudnn/Exception.h"

#include <c10/core/Contiguity.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/accumulate.h>

#include "torch_musa/csrc/aten/utils/Context.h"

namespace at::musa {

namespace {

bool IsTensorCoreType(ScalarType dtype) {
  return dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
      dtype == at::ScalarType::QInt8 || dtype == at::ScalarType::QUInt8 ||
      dtype == at::ScalarType::Float8_e5m2 ||
      dtype == at::ScalarType::Float8_e4m3fn;
}

} // anonymous namespace

#ifdef TORCH_MUSA_USE_MUDNN_C_API

struct muTensor::Meta {
  mudnnDataType_t dtype_ = MUDNN_DATA_FLOAT;
  mudnnTensorFormat_t format_ = MUDNN_TENSOR_NCHW;
  int ndim_ = 0;
  muTensor::SizeType dims_[MUDNN_DIM_MAX];
  muTensor::SizeType strides_[MUDNN_DIM_MAX];
  std::optional<float> scale_;
};

muTensor::muTensor() : meta_(std::make_unique<muTensor::Meta>()) {}

muTensor::~muTensor() = default;

muTensor::muTensor(muTensor&& other) noexcept : meta_(nullptr) {
  this->operator=(std::move(other));
}

muTensor& muTensor::operator=(muTensor&& other) noexcept {
  if (this == (&other)) {
    return *this;
  }
  Descriptor<
      mudnnTensorStruct,
      mudnnCreateTensorDescriptor,
      mudnnDestroyTensorDescriptor>::operator=(std::move(other));
  addr_ = std::exchange(other.addr_, nullptr);
  meta_ = std::move(other.meta_);
  return *this;
}

mudnnStatus_t muTensor::SetFormat(mudnnTensorFormat_t format) {
  TORCH_INTERNAL_ASSERT(static_cast<bool>(meta_));
  meta_->format_ = format;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::SetFormat(muTensor::Format format) {
#define CASE(FMT1, FMT2)    \
  case Format::FMT1: {      \
    return SetFormat(FMT2); \
  }

  switch (format) {
    MUDNN_FORALL_FORMATS(CASE)
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

mudnnStatus_t muTensor::SetType(mudnnDataType_t dtype) {
  TORCH_INTERNAL_ASSERT(static_cast<bool>(meta_));
  meta_->dtype_ = dtype;
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::SetType(muTensor::Type dtype) {
#define CASE(TYPE1, TYPE2) \
  case Type::TYPE1: {      \
    return SetType(TYPE2); \
  }

  switch (dtype) {
    MUDNN_FORALL_TYPES_BASE(CASE)
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 4000
    MUDNN_FORALL_TYPES_FP8(CASE)
#endif
    default:
      throw std::runtime_error("unreachable");
  }
#undef CASE
}

mudnnStatus_t muTensor::SetQuantizationInfo(const float scale) {
  TORCH_INTERNAL_ASSERT(static_cast<bool>(meta_));
  (meta_->scale_).emplace(scale);
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::SetNdInfo(int64_t ndim, const int64_t* dims) {
  TORCH_INTERNAL_ASSERT(static_cast<bool>(meta_));
  meta_->ndim_ = static_cast<int>(ndim);
  for (SizeType i = 0; i < meta_->ndim_; ++i) {
    meta_->dims_[i] = static_cast<SizeType>(dims[i]);
  }
  ForceContig();
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::SetNdInfo(
    std::initializer_list<int64_t> dims,
    std::initializer_list<int64_t> strides) {
  TORCH_INTERNAL_ASSERT(dims.size() == strides.size());
  return SetNdInfo(
      static_cast<int64_t>(dims.size()), std::data(dims), std::data(strides));
}

mudnnStatus_t muTensor::SetNdInfo(
    int64_t ndim,
    const int64_t* dims,
    const int64_t* strides) {
  TORCH_INTERNAL_ASSERT(static_cast<bool>(meta_));
  meta_->ndim_ = static_cast<int>(ndim);
  for (SizeType i = 0; i < meta_->ndim_; ++i) {
    meta_->dims_[i] = static_cast<SizeType>(dims[i]);
    meta_->strides_[i] = static_cast<SizeType>(strides[i]);
  }
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::GetNdInfo(std::vector<int64_t>& dim) {
  dim.clear();
  if (meta_) {
    for (SizeType i = 0; i < meta_->ndim_; ++i) {
      dim.emplace_back(meta_->dims_[i]);
    }
  } else {
    int tmp_dims[MUDNN_DIM_MAX];
    int tmp_strs[MUDNN_DIM_MAX];
    int tmp_ndim;
    mudnnDataType_t tmp_dt;
    CHECK_MUDNN_STATUS(
        mudnnGetTensorNdDescriptor(
            Desc(), MUDNN_DIM_MAX, &tmp_dt, &tmp_ndim, tmp_dims, tmp_strs),
        "mudnnGetTensorNdDescriptor");
    for (SizeType i = 0; i < tmp_ndim; ++i) {
      dim.emplace_back(tmp_dims[i]);
    }
  }
  return MUDNN_STATUS_SUCCESS;
}

mudnnStatus_t muTensor::GetNdInfo(
    std::vector<int64_t>& dim,
    std::vector<int64_t>& stride) {
  dim.clear();
  stride.clear();
  if (meta_) {
    for (SizeType i = 0; i < meta_->ndim_; ++i) {
      dim.emplace_back(meta_->dims_[i]);
      stride.emplace_back(meta_->strides_[i]);
    }
  } else {
    int tmp_dims[MUDNN_DIM_MAX];
    int tmp_strs[MUDNN_DIM_MAX];
    int tmp_ndim;
    mudnnDataType_t tmp_dt;
    CHECK_MUDNN_STATUS(
        mudnnGetTensorNdDescriptor(
            Desc(), MUDNN_DIM_MAX, &tmp_dt, &tmp_ndim, tmp_dims, tmp_strs),
        "mudnnGetTensorNdDescriptor");
    for (SizeType i = 0; i < tmp_ndim; ++i) {
      dim.emplace_back(tmp_dims[i]);
      stride.emplace_back(tmp_strs[i]);
    }
  }
  return MUDNN_STATUS_SUCCESS;
}

bool muTensor::GetDescriptorInfoIfSet(
    mudnnDataType_t* dtype,
    mudnnTensorFormat_t* format,
    int* ndims,
    const int64_t** dims,
    const int64_t** strides,
    float* scale) const {
  if (!meta_) {
    return false;
  }
  *dtype = meta_->dtype_;
  *format = meta_->format_;
  *ndims = meta_->ndim_;
  *dims = meta_->dims_;
  *strides = meta_->strides_;
  *scale = meta_->scale_.value_or(1.0f);
  return true;
}

// void muTensor::CheckContig() const {
//   ArrayRef<SizeType> dims(meta_->dims_, meta_->ndim_);
//   ArrayRef<SizeType> strides(meta_->strides_, meta_->ndim_);
//   const auto numel = c10::multiply_integers(dims.begin(), dims.end());
//   TORCH_CHECK(
//       c10::_compute_contiguous(dims, strides, static_cast<SizeType>(numel)));
// }

// bool muTensor::IsChannelsFirst() const noexcept {
//   return (meta_->format_ == MUDNN_TENSOR_NCHW) ||
//       (meta_->format_ == MUDNN_TENSOR_NCW) ||
//       (meta_->format_ == MUDNN_TENSOR_NCDHW);
// }

void muTensor::ForceContig() {
  SizeType last_stride = 1;
  for (SizeType i = meta_->ndim_ - 1; i >= 0; --i) {
    meta_->strides_[i] = last_stride;
    last_stride *= meta_->dims_[i];
  }
}

void muTensor::Build() const {
  if (!meta_) {
    return;
  }
  const auto s = (meta_->scale_).value_or(1.0f);
  CHECK_MUDNN_STATUS(
      mudnnSetTensorNdDescriptorWithScale(
          Desc(),
          meta_->format_,
          meta_->dtype_,
          meta_->ndim_,
          meta_->dims_,
          meta_->strides_,
          s),
      "mudnnSetTensorNdDescriptorWithScale");
  meta_.reset();
  return;
}

ComputeMode GetComputeModeFromCtx(const at::ScalarType& dtype) {
  auto& ctx = GlobalContext();
  auto is_tensor_mode = ctx.GetAllowTF32() || IsTensorCoreType(dtype);
  return is_tensor_mode ? MUDNN_TENSOR_OP_MATH : MUDNN_FMA_MATH;
}

ComputeMode GetMatmulComputeModeFromCtx(const at::ScalarType& dtype) {
  auto& ctx = GlobalContext();
  auto is_tensor_mode = ctx.GetAllowMublasTF32() || IsTensorCoreType(dtype);
  return is_tensor_mode ? MUDNN_TENSOR_OP_MATH : MUDNN_FMA_MATH;
}

#else

ComputeMode GetComputeModeFromCtx(const at::ScalarType& dtype) {
  auto& ctx = GlobalContext();
  auto is_tensor_mode = ctx.GetAllowTF32() || IsTensorCoreType(dtype);
  return is_tensor_mode ? ComputeMode::TENSOR : ComputeMode::SCALAR;
}

ComputeMode GetMatmulComputeModeFromCtx(const at::ScalarType& dtype) {
  auto& ctx = GlobalContext();
  auto is_tensor_mode = ctx.GetAllowMublasTF32() || IsTensorCoreType(dtype);
  return is_tensor_mode ? ComputeMode::TENSOR : ComputeMode::SCALAR;
}

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa
