#include <ATen/ATen.h>
#include <ATen/core/functional.h>
#include <torch/csrc/musa/comm.h>
#include <torch/csrc/utils/auto_gil.h>
#include <torch/csrc/utils/pybind.h>

#include "torch_musa/csrc/core/utils.h"

#include <cstddef>
#include <vector>

namespace torch {
namespace musa {
namespace python {
void InitCommMethods(PyObject* module) {
  auto m = py::cast<py::module>(module);
  m.def(
       "_broadcast_coalesced",
       [](std::vector<at::Tensor>& tensors,
          std::vector<int64_t> devices,
          size_t buffer_size) {
         return broadcast_coalesced(tensors, devices, buffer_size);
       },
       py::arg("tensors"),
       py::arg("devices"),
       py::arg("buffer_size"),
       py::call_guard<py::gil_scoped_release>())
      .def(
          "_broadcast",
          [](at::Tensor& tensor, std::vector<int64_t> devices) {
            return broadcast(tensor, devices);
          },
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_scatter",
          [](at::Tensor& tensor,
             std::vector<int64_t>& devices,
             c10::optional<std::vector<int64_t>> chunk_sizes,
             int64_t dim,
             c10::optional<py::object> py_streams) {
            c10::optional<std::vector<c10::optional<c10::musa::MUSAStream>>>
                streams;
            if (py_streams) {
              py::handle handle = *py_streams;
              streams = torch::musa::THPUtils_PySequence_to_MUSAStreamList(
                  handle.ptr());
            }
            // Note: We're holding the GIL up to here.
            AutoNoGIL no_gil;
            return scatter(tensor, devices, chunk_sizes, dim, streams);
          },
          py::arg("tensor"),
          py::arg("devices"),
          py::arg("chunk_sizes"),
          py::arg("dim"),
          py::arg("streams"))
      .def(
          "_gather",
          [](std::vector<at::Tensor>& tensors,
             int64_t dim,
             c10::optional<int32_t> destination_index) {
            return gather(tensors, dim, destination_index);
          },
          py::arg("tensors"),
          py::arg("dim"),
          py::arg("destination_index"),
          py::call_guard<py::gil_scoped_release>());
}
} // namespace python
} // namespace musa
} // namespace torch