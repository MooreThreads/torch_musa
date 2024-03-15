#ifndef TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_
#define TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_
#include <torch/csrc/musa/THCP.h>
#include <torch/csrc/python_headers.h>
#include <cstdarg>
#include <string>

namespace torch {
namespace musa {
std::vector<c10::optional<c10::musa::MUSAStream>>
THPUtils_PySequence_to_MUSAStreamList(PyObject* obj);
}
} // namespace torch

#endif // TORCH_MUSA_CSRC_CORE_STREAMUTILS_H_
