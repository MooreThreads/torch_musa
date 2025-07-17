#include "torch_musa/csrc/core/PythonTensor.h"

#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/utils/device_lazy_init.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/tensor_new.h>

#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"

namespace torch::musa {

struct PyTensorType {
  PyTypeObject py_type;
  THPDtype* dtype;
  THPLayout* layout;
  bool is_musa;
  char name[64];
  int backend;
  int scalar_type;

  at::Backend GetBackend() const {
    return static_cast<at::Backend>(backend);
  }

  at::DispatchKey GetDispatchKey() const {
    return at::backendToDispatchKey(static_cast<at::Backend>(backend));
  }

  at::ScalarType GetScalarType() const {
    return static_cast<at::ScalarType>(scalar_type);
  }
};

static_assert(
    std::is_standard_layout<PyTensorType>::value,
    "PyTensorType must be standard layout");

static const char* GetBackendName(at::Backend backend) {
  switch (backend) {
    case at::Backend::CPU:
      return "torch";
    case at::musa::kMUSABackend:
      return "torch_musa";
    default:
      AT_ERROR("Invalid backend: ", toString(backend));
  }
}

static PyObject* TensorNew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  TORCH_CHECK_TYPE(
      !tensor_type.is_musa || c10::musa::is_musa_available(),
      "type ",
      tensor_type.name,
      " not available. Torch not compiled with MUSA enabled.")
  if (tensor_type.is_musa) {
    TORCH_WARN_ONCE(
        "The torch.musa.*DtypeTensor constructors are no longer recommended. "
        "It's best to use methods such as torch.tensor(data, dtype=*, device='musa') to create tensors.")
  }
  return THPVariable_Wrap(torch::utils::legacy_tensor_ctor(
      tensor_type.GetDispatchKey(), tensor_type.GetScalarType(), args, kwargs));
  END_HANDLE_TH_ERRORS
}

PyObject* TensorDtype(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->dtype);
}

PyObject* TensorLayout(PyTensorType* self, void* unused) {
  return torch::autograd::utils::wrap(self->layout);
}

static at::Tensor dispatch_to(
    const at::Tensor& self,
    c10::Device device,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // NOTE: this is where we record aten::to in the graph during tracing.
  // However, the behavior of aten::to is different with respect to
  // TensorOptions fields that are not present: aten::to inherits fields that
  // are missing from the self argument while the tracer assumes that they
  // should be populated with the default values (eg. float for scalar type). By
  // explicitly copying over the tensor options here we fully specify all tensor
  // options and thus record the proper trace
  return self.to(
      self.options().device(device).memory_format(optional_memory_format),
      non_blocking,
      copy);
}

static at::Tensor dispatch_to(
    const at::Tensor& self,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(
      self.options().memory_format(optional_memory_format), non_blocking, copy);
}

static at::Tensor dispatch_to(
    const at::Tensor& self,
    c10::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  // TODO(mt-ai): Make this call the TensorOptions version, maybe?
  return self.to(dtype, non_blocking, copy, optional_memory_format);
}

static at::Tensor dispatch_to(
    const at::Tensor& self,
    c10::Device device,
    c10::ScalarType dtype,
    bool non_blocking,
    bool copy,
    c10::optional<c10::MemoryFormat> optional_memory_format) {
  pybind11::gil_scoped_release no_gil;
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

static PyObject* TensorIsMusa(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser({"is_musa(Tensor temp)"

  });
  torch::ParsedArgs<1> parsed_args;
  auto r = parser.parse(args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  return torch::autograd::utils::wrap(at::musa::is_musa(self_));
  END_HANDLE_TH_ERRORS
}

static PyObject* TensorInstancecheck(PyObject* _self, PyObject* arg) {
  HANDLE_TH_ERRORS
  auto self = (PyTensorType*)_self;
  if (THPVariable_Check(arg)) {
    const auto& var = THPVariable_Unpack(arg);
    if (legacyExtractDispatchKey(var.key_set()) == self->GetDispatchKey() &&
        var.scalar_type() == static_cast<at::ScalarType>(self->scalar_type)) {
      Py_RETURN_TRUE;
    }
  }
  Py_RETURN_FALSE;
  END_HANDLE_TH_ERRORS
}

static PyObject* THPVariable_musa(
    PyObject* self,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  // NOTE: it is different from `THPVariable_cuda` in
  // `pytorch/tools/autograd/templates/python_variable_methods.cpp`
  // because `THPVariable_cuda` binds to "cuda" method of `class _TensorBase`.
  // and `THPVariable_musa` binds to "_musa" function which doesn't belong to
  // any class. Hence we should place `Tensor temp` parameter to accept `self`
  static PythonArgParser parser(
      {"musa(Tensor temp, Device? device=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
       "musa(Tensor temp, Device? device=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"});
  ParsedArgs<4> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return handle_torch_function(
        r, self, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  auto device =
      r.isNone(1) ? at::Device(at::DeviceType::PrivateUse1) : r.device(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  TORCH_CHECK(
      device.type() == at::DeviceType::PrivateUse1,
      "Invalid device, must be musa device");
  torch::utils::device_lazy_init(at::musa::kMUSA);
  return THPVariable_Wrap(
      dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

static struct PyMethodDef metaclass_methods[] = {
    {"__instancecheck__", TensorInstancecheck, METH_O, nullptr},
    {nullptr}};

static struct PyGetSetDef metaclass_properties[] = {
    {"dtype", (getter)TensorDtype, nullptr, nullptr, nullptr},
    {"layout", (getter)TensorLayout, nullptr, nullptr, nullptr},
    {nullptr}};

static PyTypeObject metaclass = {
    PyVarObject_HEAD_INIT(nullptr, 0) "torch.tensortype", /* tp_name */
    sizeof(PyTypeObject) /* tp_basicsize */
};

static void PyInitializeMetaclass(PyTypeObject& metaclass) {
  metaclass.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  metaclass.tp_methods = metaclass_methods;
  metaclass.tp_getset = metaclass_properties;
  metaclass.tp_base = &PyType_Type;
  if (PyType_Ready(&metaclass) < 0) {
    throw python_error();
  }
}

static PyTypeObject tensor_type_prototype = {
    PyVarObject_HEAD_INIT(&metaclass, 0) nullptr, /* tp_name */
    sizeof(PyTensorType) /* tp_basicsize */
};

static void PyInitializeTensorType(
    PyTypeObject& type,
    const char* name,
    PyObject* tp_dict) {
  // NOTE: we don't use the typical static declaration of PyTypeObject because
  // we need to initialize as many types as there are VariableType instances.
  // We copy the basic object fields from a prototype definition and initialize
  // the remaining fields below.
  memcpy(&type, &tensor_type_prototype, sizeof(PyTypeObject));
  // Subclassing from torch.<at::ScalarType>Tensor isn't supported.
  // (Py_TPFLAGS_BASETYPE omitted). Subclassing torch.Tensor still allowed.
  type.tp_flags = Py_TPFLAGS_DEFAULT;
  type.tp_name = name;
  type.tp_new = TensorNew;
  if (PyType_Ready(&type) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    throw python_error();
  }
}

static std::string GetName(at::Backend backend, at::ScalarType scalarType) {
  std::ostringstream ss;
  ss << GetBackendName(backend) << "." << toString(scalarType) << "Tensor";
  return ss.str();
}

static void SetType(
    PyTensorType& type_obj,
    at::Backend backend,
    at::ScalarType scalarType) {
  // This field is lazily initialized from backend and scalar_type
  type_obj.backend = static_cast<int>(backend);
  type_obj.scalar_type = static_cast<int>(scalarType);
  type_obj.layout =
      (THPLayout*)Py_NewRef(torch::getTHPLayout(layout_from_backend(backend)));
  type_obj.dtype = (THPDtype*)Py_NewRef(torch::getTHPDtype(scalarType));
  type_obj.is_musa = (backend == at::musa::kMUSABackend);
}

static void SetName(PyTensorType& type_obj, const std::string& name) {
  size_t n = sizeof(type_obj.name);
  strncpy(type_obj.name, name.c_str(), n);
  type_obj.name[n - 1] = '\0';
}

static THPObjectPtr GetTensorDict() {
  auto torch = THPObjectPtr(PyImport_ImportModule("torch"));
  if (!torch) {
    throw python_error();
  }

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) {
    throw python_error();
  }

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) {
    throw python_error();
  }

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    throw python_error();
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    throw python_error();
  }

  return res;
}

static std::vector<PyTensorType> tensor_types;

static void InitializeMusaAtenTypes(std::vector<PyTensorType>& tensor_types) {
  std::vector<std::pair<at::Backend, at::ScalarType>> declared_types;
  std::vector<at::ScalarType> scalar_types = {
      at::ScalarType::Bool,
      at::ScalarType::Byte,
      at::ScalarType::Char,
      at::ScalarType::Double,
      at::ScalarType::Float,
      at::ScalarType::Int,
      at::ScalarType::Long,
      at::ScalarType::Short,
      at::ScalarType::Half,
      at::ScalarType::BFloat16};

  for (auto& scalar_type : scalar_types) {
    declared_types.emplace_back(
        std::make_pair(at::musa::kMUSABackend, scalar_type));
  }

  tensor_types.resize(declared_types.size());

  for (size_t i = 0; i != declared_types.size(); i++) {
    tensor_types[i] = PyTensorType();
    auto& tensor_type = tensor_types[i];
    at::Backend backend = declared_types[i].first;
    at::ScalarType scalar_type = declared_types[i].second;
    SetType(tensor_type, backend, scalar_type);
    SetName(tensor_type, GetName(backend, scalar_type));
  }
}

static void PyBindTensorTypes(const std::vector<PyTensorType>& tensor_types) {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch_musa"));
  if (!torch_module) {
    throw python_error();
  }

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) {
    throw python_error();
  }

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) {
      throw python_error();
    }

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      throw python_error();
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      throw python_error();
    }
  }
}

static PyMethodDef MusaTensorMethods[] = {
    {"_musa",
     castPyCFunctionWithKeywords(THPVariable_musa),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_is_musa",
     castPyCFunctionWithKeywords(TensorIsMusa),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {nullptr}};

PyMethodDef* GetTensorMethods() {
  return MusaTensorMethods;
}

void InitializePythonBindings() {
  // Initialize the at::Type* pointers, name, and properties of the PyTensorType
  // vector. After this call, the vector must not be resized.
  InitializeMusaAtenTypes(tensor_types);

  // Initialize the Python metaclass for the torch.FloatTensor, etc. types.
  // The metaclass handles __instancecheck__ checks and binds the dtype property
  // on the type objects.
  PyInitializeMetaclass(metaclass);

  // Get the tp_dict of the Variable class. We copy function definitions
  // onto each Tensor type object so that they can be accessed via e.g.
  // `torch.musa.FloatTensor.add`.
  auto tensor_dict = GetTensorDict();

  // Initialize each Python type object torch.musa.FloatTensor,
  // torch.musa.DoubleTensor, etc.
  for (auto& tensor_type : tensor_types) {
    PyInitializeTensorType(
        tensor_type.py_type, tensor_type.name, tensor_dict.get());
  }

  // Add the type objects to their corresponding modules. e.g.
  // torch.musa.FloatTensor is added to the `torch` module as `FloatTensor`.
  // Also add all the type objects to the set torch_musa._tensor_classes.
  PyBindTensorTypes(tensor_types);
}

} // namespace torch::musa
