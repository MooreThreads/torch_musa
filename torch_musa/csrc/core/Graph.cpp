#include <torch/csrc/python_headers.h>
#include <torch/csrc/utils/python_arg_parser.h>

#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"

template <typename T>
using shared_ptr_class_ = py::class_<T, std::shared_ptr<T>>;

void THCPGraph_init(PyObject* module) {
  auto torch_C_m = py::handle(module).cast<py::module>();

  torch_C_m.def("_graph_pool_handle", &::at::musa::graph_pool_handle);

  shared_ptr_class_<::at::musa::MUSAGraph>(torch_C_m, "_MUSAGraph")
      .def(py::init<>())
      .def(
          "capture_begin",
          [](::at::musa::MUSAGraph& self,
             c10::optional<c10::musa::MempoolId_t> pool_opt,
             std::string capture_error_mode) {
            musaStreamCaptureMode capture_mode;
            c10::musa::MempoolId_t pool = pool_opt.has_value()
                ? pool_opt.value()
                : c10::musa::MempoolId_t{0, 0};
            if (capture_error_mode == "global") {
              capture_mode = musaStreamCaptureModeGlobal;
            } else if (capture_error_mode == "thread_local") {
              capture_mode = musaStreamCaptureModeThreadLocal;
            } else if (capture_error_mode == "relaxed") {
              capture_mode = musaStreamCaptureModeRelaxed;
            } else {
              TORCH_CHECK(
                  false,
                  "Unknown capture error mode. Expected `global`, `thread_local`, or `relaxed`, got ",
                  capture_error_mode);
            }
            return self.capture_begin(pool, capture_mode);
          },
          py::arg("pool"),
          py::arg("capture_error_mode"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "capture_end",
          torch::wrap_pybind_function_no_gil(&at::musa::MUSAGraph::capture_end))
      .def(
          "reinstantiate_graph",
          torch::wrap_pybind_function_no_gil(
              &at::musa::MUSAGraph::reinstantiate_graph))
      .def(
          "replay",
          torch::wrap_pybind_function_no_gil(&at::musa::MUSAGraph::replay))
      .def(
          "reset",
          torch::wrap_pybind_function_no_gil(&at::musa::MUSAGraph::reset))
      .def(
          "pool",
          torch::wrap_pybind_function_no_gil(&at::musa::MUSAGraph::pool))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::MUSAGraph::debug_dump))
      .def(
          "enable_debug_mode",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::MUSAGraph::enable_debug_mode))
      .def(
          "debug_dump",
          torch::wrap_pybind_function_no_gil(
              &::at::musa::MUSAGraph::debug_dump),
          py::arg("debug_path"));
}
