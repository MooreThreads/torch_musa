#ifndef TORCH_MUSA_CSRC_CORE_PYTHONTENSOR_H_
#define TORCH_MUSA_CSRC_CORE_PYTHONTENSOR_H_

#include <torch/csrc/tensor/python_tensor.h>

namespace torch::musa {

// Helper functions that initialize the Python bindings lazily
void InitializePythonBindings();

PyMethodDef* GetTensorMethods();

} // namespace torch::musa

#endif
