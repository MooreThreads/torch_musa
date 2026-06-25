#ifndef TORCH_MUSA_CSRC_ATEN_UTILS_MUDNNUTILS_H_
#define TORCH_MUSA_CSRC_ATEN_UTILS_MUDNNUTILS_H_

#include "torch_musa/csrc/aten/utils/Utils.h"

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/musa/Exceptions.h"

namespace at::musa {

muTensor CreateMUTensor(const Tensor& t, bool permute_if_not_contiguous = true);

std::pair<muTensor, muTensor> CreateMUTensorsCompression(
    const Tensor& t1,
    const Tensor& t2);

inline muTensor CreateEmptyMUTensor() {
  return muTensor();
}

void ConfigFormat(
    const Tensor& t,
    muTensor& mt,
    bool permute_if_not_contiguous = true);

void SetMUTensorDType(ScalarType dtype, muTensor& m_t);

void SetMUTensorAddr(void* addr, muTensor& m_t);

void SetMudnnQuantizationInfo(
    muTensor& self,
    double scales,
    int64_t zero_points);

} // namespace at::musa

#endif // TORCH_MUSA_CSRC_ATEN_UTILS_MUDNNUTILS_H_
