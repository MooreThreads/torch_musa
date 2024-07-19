#ifndef TORCH_MUSA_CSRC_CORE_PYTHONTENSOR_H_
#define TORCH_MUSA_CSRC_CORE_PYTHONTENSOR_H_

#include <torch/csrc/tensor/python_tensor.h>

namespace torch {
namespace musa {

// Helper functions that initialize the Python bindings lazily
void InitializePythonBindings();
PyMethodDef* GetTensorMethods();

// Same as set_default_tensor_type() but takes a PyObject*
void PySetDefaultTensorType(PyObject* obj);

// Same as py_set_default_tensor_type, but only changes the dtype (ScalarType).
void PySetDefaultDtype(PyObject* obj);

} // namespace musa
} // namespace torch

#endif
