#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_UTILS_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_UTILS_H_

#include <ATen/Dispatch.h>
#include <c10/core/Backend.h>

#include <mudnn.h>
#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/core/MUSAException.h"

namespace at {

namespace musa {

#define UNUSED(x) (void)(x)

#define CheckContiguous(input)                          \
  TORCH_CHECK(                                          \
      input.is_contiguous() && !input.storage_offset(), \
      "check contiguous and offset failed for pooling!");

#define CheckContiguousWithName(op_name, input)                  \
  TORCH_CHECK(                                                   \
      input.is_contiguous() && !input.storage_offset(),          \
      "check contiguous and no offset failed for unary op!(op:", \
      op_name,                                                   \
      "; contiguous:",                                           \
      input.is_contiguous(),                                     \
      "; offset:",                                               \
      input.storage_offset(),                                    \
      ")");

#define AT_DISPATCH_ALL_MTGPU_TYPES_AND_HALF(TYPE, NAME, ...)                 \
  AT_DISPATCH_SWITCH(                                                         \
      TYPE,                                                                   \
      NAME,                                                                   \
      AT_DISPATCH_CASE(at::ScalarType::Byte, __VA_ARGS__) AT_DISPATCH_CASE(   \
          at::ScalarType::Char,                                               \
          __VA_ARGS__) AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)    \
          AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)             \
              AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)            \
                  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__)        \
                      AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)      \
                          AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
                              AT_DISPATCH_CASE(                               \
                                  at::ScalarType::Double, __VA_ARGS__));

using muTensor = ::musa::dnn::Tensor;
using muHandle = ::musa::dnn::Handle;

constexpr c10::Backend kMUSABackend = c10::Backend::PrivateUse1;
constexpr DeviceType kMUSA = DeviceType::PrivateUse1;
constexpr c10::DispatchKey kMUSAKey = c10::DispatchKey::PrivateUse1;

#define MUSA_TENSOR_TYPE_CHECK(self)                   \
  TORCH_CHECK(                                         \
      ((self.scalar_type() == ScalarType::Double) ||   \
       (self.scalar_type() == ScalarType::Float) ||    \
       (self.scalar_type() == ScalarType::Half) ||     \
       (self.scalar_type() == ScalarType::BFloat16) || \
       (self.scalar_type() == ScalarType::Int) ||      \
       (self.scalar_type() == ScalarType::Long)),      \
      "Now muDNN only support float64, float32, half, bfloat16, int32, and int64");

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

muTensor CreateMUTensor(const Tensor& t, bool permute_if_not_contiguous = true);

muTensor CreateMUTensorByCompressDim(const Tensor& t);

inline muTensor CreateEmptyMUTensor() {
  return muTensor();
}

// May need to contiguous the input pytorch tensor according the needed
// tensor format, so need to pass tensor as reference
void ConfigFormat(
    const Tensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous = true);

void SetMUTensorDType(ScalarType dtype, muTensor& m_t);

void SetMUTensorAddr(void* addr, muTensor& m_t);

// Set quantized mudnn tensor info
void SetMudnnQuantizationInfo(
    muTensor& self,
    double scales,
    int64_t zero_points);

// use for memory handler
void InternalMemFree(void* ptr);
::musa::dnn::MemoryHandler InternalMemAlloc(size_t s);

bool is_musa(const Tensor& t);

Tensor create_out(
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

void check_inplace(
    const Tensor& self,
    IntArrayRef sizes,
    const TensorOptions& options);

void resize_out(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

c10::optional<Tensor> maybe_create_proxy(
    const Tensor& out,
    IntArrayRef sizes,
    IntArrayRef strides,
    const TensorOptions& options);

bool MatContiguous(const Tensor& mat);

bool IsTranspose(const Tensor& mat, bool strict = true);

bool IsLastDimContiguous(const Tensor& input);

Tensor FormatContiguous(const Tensor& t, at::MemoryFormat memory_format);

size_t DTypeSize(c10::ScalarType type);

/**
 * @brief There is a buggy check for zero strides. We should change the
 *        zero stride to 1 to keep the mudnn to get right data format.
 * @todo TODO: (lms) Actually we should make this func inline, but inline
 *       this func in Utils.h will cause compilation-chain broken, so just
 *       keep the declaration and definition separate temporarily.
 * @param t Tensor to change stride from zero to 1
 * @return at::Tensor output tensor of this func.
 */
at::Tensor ContiguousIfZeroInStrides(const at::Tensor& t);

template <typename... AllowedDtypes>
inline constexpr int NAllowedDtypes(AllowedDtypes... allowed_dtypes) {
  return sizeof...(allowed_dtypes);
}

inline constexpr bool IsAllowedDtypeImpl(ScalarType dtype) {
  return false;
}

template <typename... AllowedDtypes>
inline bool IsAllowedDtypeImpl(
    ScalarType input_dtype,
    ScalarType allowed_dtype,
    AllowedDtypes... allowed_dtypes) {
  return input_dtype == allowed_dtype ||
      IsAllowedDtypeImpl(input_dtype, allowed_dtypes...);
}

template <typename... AllowedDtypes>
inline bool IsAllowedDtype(
    ScalarType input_dtype,
    AllowedDtypes... allowed_dtypes) {
  return IsAllowedDtypeImpl(input_dtype, allowed_dtypes...);
}

#define IS_ALLOWED_DTYPES(input_dtype, ...) \
  IsAllowedDtype(input_dtype, __VA_ARGS__);

#define IS_ALLOWED_FLOATING_DTYPES(input_dtype) \
  IS_ALLOWED_DTYPES(                            \
      input_dtype, ScalarType::Float, ScalarType::Half, ScalarType::BFloat16);

#define _TORCH_MUSA_CHECK_DTYPES(type, func, cond) \
  TORCH_CHECK(                                     \
      cond, '"', func, "\" not implemented for '", c10::toString(type), "'");

#define TORCH_MUSA_CHECK_DTYPES(input_dtype, func_name, ...) \
  static_assert(NAllowedDtypes(__VA_ARGS__) != 0);           \
  const auto& the_type = input_dtype;                        \
  ScalarType _st = ::detail::scalar_type(the_type);          \
  bool cond = IS_ALLOWED_DTYPES(_st, __VA_ARGS__);           \
  _TORCH_MUSA_CHECK_DTYPES(_st, func_name, cond)

#define TORCH_MUSA_CHECK_FLOATING_TYPES(input_dtype, func_name) \
  const auto& the_type = input_dtype;                           \
  ScalarType _st = ::detail::scalar_type(the_type);             \
  bool cond = IS_ALLOWED_FLOATING_DTYPES(_st);                  \
  _TORCH_MUSA_CHECK_DTYPES(_st, func_name, cond)

#define TORCH_MUSA_CHECK_FLOATING_TYPES_AND_N(input_dtype, func_name, ...) \
  static_assert(NAllowedDtypes(__VA_ARGS__) != 0);                         \
  const auto& the_type = input_dtype;                                      \
  ScalarType _st = ::detail::scalar_type(the_type);                        \
  bool cond1 = IS_ALLOWED_FLOATING_DTYPES(_st);                            \
  bool cond2 = IS_ALLOWED_DTYPES(_st, __VA_ARGS__);                        \
  _TORCH_MUSA_CHECK_DTYPES(_st, func_name, cond1 || cond2)

template <typename Args1, typename... ArgsN>
inline Device OutDevice(Args1&& first, ArgsN&&... others) {
  auto dev = first.device();
  if (!dev.is_cpu()) {
    return dev;
  }
  for (const auto& o : {others...}) {
    if (!o.is_cpu()) {
      dev = o.device();
      break;
    }
  }
  return dev;
}

} // namespace musa

using musa::kMUSA;

} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_UTILS_H_
