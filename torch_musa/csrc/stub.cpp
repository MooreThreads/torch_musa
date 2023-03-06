#include <pybind11/pybind11.h>

extern PyObject* InitMusaModule();

PyMODINIT_FUNC PyInit__MUSAC(void) {
  return InitMusaModule();
}
