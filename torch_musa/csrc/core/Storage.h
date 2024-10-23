#ifndef TORCH_MUSA_CSRC_CORE_STORAGE_H
#define TORCH_MUSA_CSRC_CORE_STORAGE_H

#include <ATen/Utils.h>
#include <c10/core/StorageImpl.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/autograd/python_autograd.h>
#include <torch/csrc/Storage.h>
#include <torch/csrc/THP.h>

namespace at {
namespace musa {
PyMethodDef* GetStorageMethods();
}
} // namespace at

#endif // !TORCH_MUSA_CSRC_CORE_STORAGE_H
