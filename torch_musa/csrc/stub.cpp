#include <pybind11/pybind11.h>

extern void InitMusaModule(PyObject* module);

PYBIND11_MODULE(_MUSAC, m) {
  InitMusaModule(m.ptr());
}
