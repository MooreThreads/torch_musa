#ifndef TORCH_MUSA_CSRC_ATEN_OPS_ELEMWISEHELPERS_H_
#define TORCH_MUSA_CSRC_ATEN_OPS_ELEMWISEHELPERS_H_

#include <ATen/Config.h>

#include <c10/core/Scalar.h>
#include <c10/core/ScalarType.h>

#include <utility>

#include "torch_musa/csrc/aten/utils/TensorIterator.h"

namespace at {
namespace musa {

using BINARY_MODE = ::musa::dnn::Binary::Mode;
using UNARY_MODE = ::musa::dnn::Unary::Mode;

Scalar CPUScalarMulScalar(const Tensor& l_t, const Scalar& r_s);

Scalar CPUScalarReciprocal(const Scalar& input);

void UnaryCall(
    MusaTensorIterator& iter,
    UNARY_MODE mode,
    const std::string op_name);

void UnaryAlphaCall(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    UNARY_MODE mode,
    const std::string op_name);

void BinaryCall(
    MusaTensorIterator& iter,
    BINARY_MODE mode,
    const std::string& op_name);

void BinaryAlphaCall(
    MusaTensorIterator& iter,
    const Scalar& alpha,
    BINARY_MODE mode,
    const std::string& op_name);

inline bool BinaryMulFallThroughCPU(MusaTensorIterator& iter) {
  const auto l_dtype = iter.dtype(1);
  const auto r_dtype = iter.dtype(2);
  if (l_dtype == ScalarType::Bool || r_dtype == ScalarType::Bool) {
    return true;
  }
  if (l_dtype == ScalarType::Double && r_dtype == ScalarType::Double) {
    return true;
  }
  if (iter.is_cpu_scalar(1) && r_dtype == ScalarType::Double) {
    return true;
  }
  if (iter.is_cpu_scalar(2) && l_dtype == ScalarType::Double) {
    return true;
  }
  return false;
}

inline bool BinaryDivFallThroughCPU(MusaTensorIterator& iter) {
  return BinaryMulFallThroughCPU(iter);
}

std::pair<ScalarType, ScalarType> BinaryTrueDivSuggestInputTypes(
    MusaTensorIterator& iter);

} // namespace musa
} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_OPS_ELEMWISEHELPERS_H_
