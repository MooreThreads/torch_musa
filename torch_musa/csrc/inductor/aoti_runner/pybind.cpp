
#include <torch/csrc/inductor/aoti_torch/utils.h>
#include <torch_musa/csrc/inductor/aoti_runner/model_container_runner_musa.h>

#include <torch/csrc/utils/pybind.h>

namespace torch::inductor {

void initAOTIMUSARunnerBindings(PyObject* module) {
  auto rootModule = py::handle(module).cast<py::module>();
  auto m = rootModule.def_submodule("_aoti");

  py::class_<AOTIModelContainerRunnerMusa>(m, "AOTIModelContainerRunnerMusa")
      .def(py::init<const std::string&, int>())
      .def(py::init<const std::string&, int, const std::string&>())
      .def(py::init<
           const std::string&,
           int,
           const std::string&,
           const std::string&>())
      .def("run", &AOTIModelContainerRunnerMusa::run)
      .def("get_call_spec", &AOTIModelContainerRunnerMusa::get_call_spec)
      .def(
          "get_constant_names_to_original_fqns",
          &AOTIModelContainerRunnerMusa::getConstantNamesToOriginalFQNs)
      .def(
          "get_constant_names_to_dtypes",
          &AOTIModelContainerRunnerMusa::getConstantNamesToDtypes);
}
} // namespace torch::inductor
