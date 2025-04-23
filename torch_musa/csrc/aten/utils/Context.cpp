#include <pybind11/pybind11.h>

#include <c10/core/ScalarType.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>

#include "torch_musa/csrc/aten/utils/Context.h"

namespace at {
namespace musa {

Context& GlobalContext() {
  static Context global_context_;
  return global_context_;
}

// Check if a type is a tensor core type.
inline bool IsTensorCoreType(const at::ScalarType& dtype) {
  return dtype == at::ScalarType::Half || dtype == at::ScalarType::BFloat16 ||
      dtype == at::ScalarType::QInt8 || dtype == at::ScalarType::QUInt8 ||
      dtype == at::ScalarType::Float8_e5m2 ||
      dtype == at::ScalarType::Float8_e4m3fn;
}

::musa::dnn::Convolution::ComputeMode GetComputeModeFromCtx(
    const at::ScalarType& dtype) {
  auto& ctx = GlobalContext();
  auto is_tensor_mode = ctx.GetAllowTF32() || IsTensorCoreType(dtype);
  return is_tensor_mode ? ::musa::dnn::Convolution::ComputeMode::TENSOR
                        : ::musa::dnn::Convolution::ComputeMode::SCALAR;
}

bool Context::GetAllowTF32() const {
  return allow_tf32_;
}

void Context::SetAllowTF32(bool allow_tf32) {
  allow_tf32_ = allow_tf32;
}

PyObject* THPModuleSetAllowTF32(PyObject* /*unused*/, PyObject* arg) {
  THPUtils_assert(
      PyBool_Check(arg),
      "set_allow_tf32_cublas expects a bool, "
      "but got %s",
      THPUtils_typename(arg));
  at::musa::GlobalContext().SetAllowTF32(arg == Py_True);
  Py_RETURN_NONE;
}

PyObject* THPModuleGetAllowTF32(PyObject* /*_unused*/, PyObject* /*_unused*/) {
  if (at::musa::GlobalContext().GetAllowTF32())
    Py_RETURN_TRUE;
  else
    Py_RETURN_FALSE;
}

static PyMethodDef ContextMethods[] = { // NOLINT
    {"_get_allow_tf32", THPModuleGetAllowTF32, METH_NOARGS, nullptr},
    {"_set_allow_tf32", THPModuleSetAllowTF32, METH_O, nullptr},
    {nullptr, nullptr, 0, nullptr}};

PyMethodDef* GetContextMethods() {
  return ContextMethods;
}

} // namespace musa
} // namespace at
