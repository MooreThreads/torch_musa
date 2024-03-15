#include "PythonTensor.h"

#include <torch/csrc/autograd/generated/VariableType.h>
#include <torch/csrc/autograd/utils/wrap_outputs.h>
#include <torch/csrc/tensor/python_tensor.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/tensor_new.h>
#include <torch/csrc/utils/tensor_types.h>

#include "Device.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/utils/Logging.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

namespace torch {
namespace musa {

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

static TypeError UnavailableType(const PyTensorType& type) {
  return TypeError(
      "type %s not available. Torch not compiled with musa enabled.",
      type.name);
}

std::vector<at::DeprecatedTypeProperties*> AllTypesForBackends(
    at::ArrayRef<at::Backend> backends) {
  std::vector<at::DeprecatedTypeProperties*> res;
  res.reserve(backends.size());
  for (auto p : backends) {
    for (int64_t s = 0; s < static_cast<int64_t>(at::ScalarType::NumOptions);
         s++) {
      auto& type = at::getDeprecatedTypeProperties(
          static_cast<at::Backend>(p), static_cast<at::ScalarType>(s));
      res.emplace_back(&type);
    }
  }
  return res;
}

std::vector<at::DeprecatedTypeProperties*> AllMusaTypes() {
  return AllTypesForBackends({at::musa::kMUSABackend});
}

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

std::string replace_torch_musa(std::string str) {
  if (str.find("torch_musa") != std::string::npos) {
    // replace torch_musa with torch.musa
    std::replace(str.begin(), str.begin() + 6, '_', '.');
  }
  return str;
}

std::string OptionsToString(const at::TensorOptions options) {
  std::ostringstream ss;
  ss << GetBackendName(options.backend()) << "."
     << toString(at::typeMetaToScalarType(options.dtype())) << "Tensor";
  return replace_torch_musa(ss.str());
}

std::string TypeToString(const at::DeprecatedTypeProperties& type) {
  std::ostringstream ss;
  ss << GetBackendName(type.backend()) << "." << toString(type.scalarType())
     << "Tensor";
  return ss.str();
}

at::TensorOptions OptionsFromString(const std::string& str) {
  static std::once_flag cpu_once;
  static std::once_flag musa_once;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*> cpu_map;
  static std::unordered_map<std::string, at::DeprecatedTypeProperties*>
      musa_map;

  const std::unordered_map<std::string, at::DeprecatedTypeProperties*>*
      type_map = nullptr;

  if (str == "torch.Tensor") {
    auto backend =
        dispatchKeyToBackend(torch::tensors::get_default_dispatch_key());
    auto scalar_type = torch::tensors::get_default_scalar_type();
    return at::getDeprecatedTypeProperties(backend, scalar_type).options();
  }

  if ((str.find("torch_musa.") != std::string::npos) ||
      (str.find("torch.musa.") != std::string::npos)) {
    // torch.musa. or torch_musa. is prefix of str
    std::call_once(musa_once, []() {
      for (auto type : AllMusaTypes()) {
        std::string origStr = TypeToString(*type);
        musa_map.emplace(origStr, type);
        // hence torch.musa.xxx key is also included
        musa_map.emplace(replace_torch_musa(origStr), type);
      }
    });
    type_map = &musa_map;
  } else {
    std::call_once(cpu_once, []() {
      for (auto type : torch::autograd::VariableType::allCPUTypes()) {
        cpu_map.emplace(torch::utils::type_to_string(*type), type);
      }
    });
    type_map = &cpu_map;
  }

  auto it = type_map->find(str);
  if (it == type_map->end()) {
    throw torch::ValueError("invalid type: '%s'", str.c_str());
  }
  return it->second->options();
}

static PyObject* TensorNew(
    PyTypeObject* type,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  auto& tensor_type = *((PyTensorType*)type);
  if (tensor_type.is_musa && c10::musa::device_count() == 0) {
    throw UnavailableType(tensor_type);
  }
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
  AutoNoGIL no_gil;
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
  // TODO: Make this call the TensorOptions version, maybe?
  return self.to(device, dtype, non_blocking, copy, optional_memory_format);
}

PyObject* GetTensorType(PyObject* self, PyObject* args, PyObject* kwargs) {
  HANDLE_TH_ERRORS
  static torch::PythonArgParser parser(
      {"type(Tensor temp, PyObject* dtype=None, bool non_blocking=False, *, MemoryFormat? memory_format=None)",
       "type(Tensor temp, PyObject* dtype=None, bool async=False, *, MemoryFormat? memory_format=None)|deprecated"});

  torch::ParsedArgs<4> parsed_args;
  auto r = parser.parse(self, args, kwargs, parsed_args);
  auto self_ = r.tensor(0);
  if (r.has_torch_function()) {
    return torch::handle_torch_function(
        r, args, kwargs, THPVariableClass, "torch.Tensor");
  }

  if (r.isNone(1)) {
    return THPUtils_packString(OptionsToString(self_.options()));
  }
  auto obj = r.pyobject(1);
  auto opt_memory_format = r.memoryformatOptional(3);
  std::string type_name;
  bool is_dtype = false;
  if (PyType_Check(obj)) {
    if (obj == THPVariableClass) {
      type_name = "torch.Tensor";
    } else {
      type_name = ((PyTypeObject*)obj)->tp_name;
    }
  } else if (THPUtils_checkString(obj)) {
    type_name = THPUtils_unpackString(obj);
  } else if (THPDtype_Check(obj)) {
    is_dtype = true;
  } else {
    throw torch::TypeError("dtype must be a type, str, or dtype object");
  }
  c10::ScalarType scalar_type;
  c10::Device device = self_.device();
  if (is_dtype) {
    scalar_type = r.scalartype(1);
  } else {
    at::TensorOptions options = OptionsFromString(type_name);
    scalar_type = at::typeMetaToScalarType(options.dtype());
    auto device_type = options.device().type();
    if (device_type != device.type()) {
      device = at::Device(device_type);
    }
  }
  return THPVariable_Wrap(dispatch_to(
      self_, device, scalar_type, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
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
  torch::utils::musa_lazy_init();
  return THPVariable_Wrap(
      dispatch_to(self_, device, r.toBool(2), false, opt_memory_format));
  END_HANDLE_TH_ERRORS
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,cppcoreguidelines-avoid-non-const-global-variables,modernize-avoid-c-arrays)
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
    logging::LOG_FATAL << "Metaclass initialization failed!"
                       << "\n";
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
    logging::LOG_FATAL << "Tensor type initialization failed!"
                       << "\n";
  }

  if (PyDict_Merge(type.tp_dict, tp_dict, 0) < 0) {
    logging::LOG_FATAL << "Merge tensor type failed!"
                       << "\n";
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
  type_obj.layout = torch::getTHPLayout(layout_from_backend(backend));
  type_obj.dtype = torch::getTHPDtype(scalarType);
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
    logging::LOG_FATAL << "torch module not found!"
                       << "\n";
  }

  auto tensor_class = THPObjectPtr(PyObject_GetAttrString(torch, "Tensor"));
  if (!tensor_class) {
    logging::LOG_FATAL << "Get tensor attribute from torch failed!"
                       << "\n";
  }

  auto tensor_type = (PyTypeObject*)tensor_class.get();
  TORCH_CHECK(tensor_type->tp_base, "missing base type for Tensor");

  auto res = THPObjectPtr(PyDict_New());
  if (!res) {
    logging::LOG_FATAL << "Get tensor class failed!"
                       << "\n";
  }

  if (PyDict_Merge(res.get(), tensor_type->tp_dict, 0) < 0) {
    logging::LOG_FATAL << "Merge tensor dict failed!"
                       << "\n";
  }
  if (PyDict_Merge(res.get(), tensor_type->tp_base->tp_dict, 0) < 0) {
    logging::LOG_FATAL << "Merge tensor base dict failed!"
                       << "\n";
  }

  return res;
}

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
    logging::LOG_FATAL << "torch_musa module not found!"
                       << "\n";
  }

  auto tensor_classes = THPObjectPtr(
      PyObject_GetAttrString(torch_module.get(), "_tensor_classes"));
  if (!tensor_classes) {
    logging::LOG_FATAL << "_tensor_classes not found in torch_musa!"
                       << "\n";
  }

  for (auto& tensor_type : tensor_types) {
    auto name = std::string(tensor_type.name);
    auto idx = name.rfind('.');
    auto type_name = name.substr(idx + 1);
    auto module_name = name.substr(0, idx);

    auto module_obj = THPObjectPtr(PyImport_ImportModule(module_name.c_str()));
    if (!module_obj) {
      logging::LOG_FATAL << module_name << " not found!"
                         << "\n";
    }

    PyObject* type_obj = (PyObject*)&tensor_type;
    Py_INCREF(type_obj);
    if (PyModule_AddObject(module_obj.get(), type_name.c_str(), type_obj) < 0) {
      logging::LOG_FATAL << name << " tenosr type not found!"
                         << "\n";
    }
    if (PySet_Add(tensor_classes.get(), type_obj) < 0) {
      logging::LOG_FATAL << " Add type object to classes failed!"
                         << "\n";
    }
  }
}

static PyMethodDef MusaTensorMethods[] = {
    {"_musa",
     castPyCFunctionWithKeywords(THPVariable_musa),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_type",
     castPyCFunctionWithKeywords(GetTensorType),
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
  static std::vector<PyTensorType> tensor_types;
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

} // namespace musa
} // namespace torch
