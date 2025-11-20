#include <musa.h>
#include <musa_profiler_api.h>
#include <musa_runtime.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace torch::musa {

void initMusartBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  auto musart = m.def_submodule("_musart", "libmusart.so bindings");

  py::enum_<musaError_t>(musart, "musaError").value("success", musaSuccess);

  musart.def("musaGetErrorString", musaGetErrorString);

  musart.def(
      "musaHostRegister",
      [](uintptr_t ptr, size_t size, unsigned int flags) -> musaError_t {
        py::gil_scoped_release no_gil;
        return C10_MUSA_ERROR_HANDLED(
            musaHostRegister((void*)ptr, size, flags));
      });
  musart.def("musaHostUnregister", [](uintptr_t ptr) -> musaError_t {
    py::gil_scoped_release no_gil;
    return C10_MUSA_ERROR_HANDLED(musaHostUnregister((void*)ptr));
  });

  musart.def("musaStreamCreate", [](uintptr_t ptr) -> musaError_t {
    py::gil_scoped_release no_gil;
    return C10_MUSA_ERROR_HANDLED(musaStreamCreate((musaStream_t*)ptr));
  });

  musart.def("musaStreamDestroy", [](uintptr_t ptr) -> musaError_t {
    py::gil_scoped_release no_gil;
    return C10_MUSA_ERROR_HANDLED(musaStreamDestroy((musaStream_t)ptr));
  });

  musart.def(
      "musaMemGetInfo",
      [](c10::DeviceIndex device) -> std::pair<size_t, size_t> {
        c10::musa::MUSAGuard guard(device);
        size_t device_free = 0;
        size_t device_total = 0;
        py::gil_scoped_release no_gil;
        C10_MUSA_CHECK(musaMemGetInfo(&device_free, &device_total));
        return {device_free, device_total};
      });
}

} // namespace torch::musa
