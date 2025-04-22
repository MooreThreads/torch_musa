#ifndef TORCH_MUSA_CSRC_CORE_STORAGE_H
#define TORCH_MUSA_CSRC_CORE_STORAGE_H

#include <torch/csrc/Types.h>
#include <torch/csrc/python_headers.h>

namespace at {
namespace musa {
PyMethodDef* GetStorageMethods();
}
} // namespace at

#endif // !TORCH_MUSA_CSRC_CORE_STORAGE_H
