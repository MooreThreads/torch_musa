#include <pybind11/pybind11.h>

namespace torch_musa {
namespace {
namespace py = pybind11;

void MUSARegisterFunc() {
  // static std::once_flag once;
  // std::call_once(once, []() { RegisterCustomFunc(); });
}

void PythonVariableMethods(py::module& m) {
  m.def("_musa_register_function", []() { MUSARegisterFunc(); });
}

// Init methods of py::module
void initMUSAModule(PyObject* m) {
  auto t = py::handle(m).cast<py::module>();
  PythonVariableMethods(t);
}

} // anonymous namespace
} // namespace torch_musa

PYBIND11_MODULE(_MUSAC, m) {
  torch_musa::initMUSAModule(m.ptr());
}
