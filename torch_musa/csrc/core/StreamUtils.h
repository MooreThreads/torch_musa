#ifndef TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_
#define TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_

#include <torch/csrc/musa/THCP.h>

#include "torch_musa/csrc/core/MUSAStream.h"

namespace torch::musa {

std::vector<std::optional<c10::musa::MUSAStream>>
THPUtils_PySequence_to_MUSAStreamList(PyObject* obj);

} // namespace torch::musa

#endif // TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_
