#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_TENSOR_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_TENSOR_H_

#include <utility>
#include <vector>

#include <ATen/core/TensorBase.h>

#include <mudnn.h>

#include "torch_musa/csrc/aten/mudnn/Descriptors.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#define MUDNN_FORALL_TYPES_BASE(_) \
  _(HALF, MUDNN_DATA_HALF)         \
  _(BFLOAT16, MUDNN_DATA_BFLOAT16) \
  _(FLOAT, MUDNN_DATA_FLOAT)       \
  _(DOUBLE, MUDNN_DATA_DOUBLE)     \
  _(BOOL, MUDNN_DATA_BOOLEAN)      \
  _(INT8, MUDNN_DATA_INT8)         \
  _(INT16, MUDNN_DATA_INT16)       \
  _(INT32, MUDNN_DATA_INT32)       \
  _(INT64, MUDNN_DATA_INT64)       \
  _(UINT8, MUDNN_DATA_UINT8)       \
  _(UINT16, MUDNN_DATA_UINT16)     \
  _(UINT32, MUDNN_DATA_UINT32)     \
  _(UINT64, MUDNN_DATA_UINT64)     \
  _(QINT8, MUDNN_DATA_QINT8)

#define MUDNN_FORALL_TYPES_FP8(_)  \
  _(FP8_E5M2, MUDNN_DATA_FP8_E5M2) \
  _(FP8_E4M3, MUDNN_DATA_FP8_E4M3)

#define MUDNN_FORALL_FORMATS(_) \
  _(NCW, MUDNN_TENSOR_NCW)      \
  _(NWC, MUDNN_TENSOR_NWC)      \
  _(NCHW, MUDNN_TENSOR_NCHW)    \
  _(NHWC, MUDNN_TENSOR_NHWC)    \
  _(HWCN, MUDNN_TENSOR_HWCN)    \
  _(NCDHW, MUDNN_TENSOR_NCDHW)  \
  _(NDHWC, MUDNN_TENSOR_NDHWC)  \
  _(DHWCN, MUDNN_TENSOR_DHWCN)

class muTensor : public Descriptor<
                     mudnnTensorStruct,
                     mudnnCreateTensorDescriptor,
                     mudnnDestroyTensorDescriptor> {
 private:
  using SizeType = int64_t;
  struct Meta;

  void* addr_ = nullptr;
  mutable std::unique_ptr<Meta> meta_;

  // void CheckContig() const;

  void ForceContig();

  // bool IsChannelsFirst() const noexcept;

 public:
  enum class Type {
#define CASE(TYPE, _2) TYPE,
    MUDNN_FORALL_TYPES_BASE(CASE)
#if defined(REAL_MUSA_VERSION) && REAL_MUSA_VERSION >= 4000
        MUDNN_FORALL_TYPES_FP8(CASE)
#endif
#undef CASE
  };

  enum class Format {
#define CASE(FMT, _2) FMT,
    MUDNN_FORALL_FORMATS(CASE)
#undef CASE
  };

  muTensor();
  ~muTensor();

  muTensor(const muTensor&) = delete;
  muTensor(muTensor&&) noexcept;

  muTensor& operator=(const muTensor&) = delete;
  muTensor& operator=(muTensor&&) noexcept;

  void Build() const;

  void* DataPtr() noexcept {
    return addr_;
  }

  void* DataPtr() const noexcept {
    return addr_;
  }

  mudnnStatus_t SetAddr(void* addr) noexcept {
    addr_ = addr;
    return MUDNN_STATUS_SUCCESS;
  }

  mudnnStatus_t SetType(Type dtype);
  mudnnStatus_t SetType(mudnnDataType_t dtype);

  mudnnStatus_t SetFormat(Format format);
  mudnnStatus_t SetFormat(mudnnTensorFormat_t format);

  mudnnStatus_t SetQuantizationInfo(const float scale);

  mudnnStatus_t SetNdInfo(int64_t ndim, const int64_t* dims);
  mudnnStatus_t SetNdInfo(std::initializer_list<int64_t> dims) {
    return SetNdInfo(static_cast<int64_t>(dims.size()), std::data(dims));
  }
  mudnnStatus_t SetNdInfo(
      int64_t ndim,
      const int64_t* dims,
      const int64_t* strides);
  mudnnStatus_t SetNdInfo(
      std::initializer_list<int64_t> dims,
      std::initializer_list<int64_t> strides);

  mudnnStatus_t GetNdInfo(std::vector<int64_t>& dim);
  mudnnStatus_t GetNdInfo(
      std::vector<int64_t>& dim,
      std::vector<int64_t>& stride);
  bool GetDescriptorInfoIfSet(
      mudnnDataType_t* dtype,
      mudnnTensorFormat_t* format,
      int* ndims,
      const int64_t** dims,
      const int64_t** strides,
      float* scale) const;
};

using ComputeMode = mudnnMathType_t;

#define MUDNN_UNPACK_TENSOR(mt) mt.Desc(), mt.DataPtr()

#else

using ComputeMode = ::musa::dnn::Convolution::ComputeMode;
using muTensor = ::musa::dnn::Tensor;

#endif // TORCH_MUSA_USE_MUDNN_C_API

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_TENSOR_H_
