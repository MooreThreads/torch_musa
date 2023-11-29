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
          at::ScalarType::Char, __VA_ARGS__)                                  \
          AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)                 \
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
      ((self.scalar_type() == ScalarType::Float) ||    \
       (self.scalar_type() == ScalarType::Half) ||     \
       (self.scalar_type() == ScalarType::BFloat16) || \
       (self.scalar_type() == ScalarType::Int) ||      \
       (self.scalar_type() == ScalarType::Long)),      \
      "Now muDNN only support float32, half, bfloat16, int32, and int64");

#define CHECK_MUDNN_STATUS(rst, msg)       \
  TORCH_CHECK(                             \
      rst == ::musa::dnn::Status::SUCCESS, \
      __FUNCTION__,                        \
      " MUDNN failed in: ",                \
      msg);

muTensor CreateMUTensor(const Tensor& t, bool permute_if_not_contiguous = true);

inline muTensor CreateEmptyMUTensor() {
  return muTensor();
}

// May need to contiguous the input pytorch tensor according the needed
// tensor format, so need to pass tensor as reference
void ConfigFormat(
    const Tensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous = true);

// Set quantized mudnn tensor info
void inline SetMudnnQuantizationInfo(
    at::musa::muTensor& self,
    double scales,
    int64_t zero_points) {
  float scales_ = static_cast<float>(scales);
  unsigned int zero_points_ = static_cast<unsigned int>(zero_points);
  CHECK_MUDNN_STATUS(
      self.SetQuantizationInfo(1, &scales_, &zero_points_),
      "Set quantization info");
}

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

} // namespace musa

using musa::kMUSA;

} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_UTILS_H_
