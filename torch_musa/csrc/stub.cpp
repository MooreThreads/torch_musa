#include <pybind11/pybind11.h>

extern void initMUSAModule(PyObject* m);

// Init methods of py::module
PYBIND11_MODULE(_MUSAC, m) {
  initMUSAModule(m.ptr());
}
