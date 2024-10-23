#include <ATen/Parallel.h>
#include <ATen/Utils.h>
#include <c10/util/Backtrace.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/Device.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/Generator.h>
#include <torch/csrc/THP.h>
#include <torch/csrc/utils/invalid_arguments.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/pycfunction_helpers.h>
#include <torch/csrc/utils/python_numbers.h>
#include <torch/csrc/utils/python_strings.h>
#include <vector>

#include "torch_musa/csrc/amp/autocast_mode.h"
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Allocator.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/Event.h"
#include "torch_musa/csrc/core/PythonTensor.h"
#include "torch_musa/csrc/core/Sleep.h"
#include "torch_musa/csrc/core/Stream.h"
#ifdef USE_MCCL
#include "torch_musa/csrc/core/PythonComm.h"
#include "torch_musa/csrc/distributed/Register.h"
#endif
#include "torch_musa/csrc/aten/utils/Context.h"
#include "torch_musa/csrc/core/MusaIPCTypes.h"
#include "torch_musa/csrc/core/Storage.h"
#include "torch_musa/csrc/core/StorageSharing.h"
#include "torch_musa/csrc/utils/Logging.h"
#include "torch_musa/csrc/utils/musa_lazy_init.h"

#include <pthread.h>

bool in_bad_fork = false; // True for children forked after musa init

// Called in the forked child if musa has already been initialized
static void forked_child() {
  in_bad_fork = true;
  torch::utils::set_requires_musa_init(true);
}

static void poison_fork() {
  static c10::once_flag flag;
  c10::call_once(flag, [] { pthread_atfork(nullptr, nullptr, forked_child); });
}

void AddPyMethodDefs(std::vector<PyMethodDef>& vector, PyMethodDef* methods) {
  if (!vector.empty()) {
    // remove nullptr terminator
    vector.pop_back();
  }
  while (true) {
    vector.push_back(*methods);
    if (!methods->ml_name) {
      break;
    }
    methods++;
  }
}

// yang.zhao: copied from torch/csrc/utils.cpp to avoid including other things.
void THPUtils_invalidArguments(
    PyObject* given_args,
    PyObject* given_kwargs,
    const char* function_name,
    size_t num_options,
    ...) {
  std::vector<std::string> option_strings;
  va_list option_list;
  va_start(option_list, num_options);
  std::generate_n(
      std::back_inserter(option_strings), num_options, [&option_list] {
        return va_arg(option_list, const char*);
      });
  va_end(option_list);

  PyErr_SetString(
      PyExc_TypeError,
      torch::format_invalid_args(
          given_args, given_kwargs, function_name, option_strings)
          .c_str());
}

PyObject* PyMusaEmptyCache(PyObject* unused0, PyObject* unused1) {
  HANDLE_TH_ERRORS
  c10::musa::MUSACachingAllocator::EmptyCache();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaResetPeakStats(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  c10::musa::MUSACachingAllocator::ResetPeakStats();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMemoryStats(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to memory_allocated");
  int device = (int)THPUtils_unpackLong(arg);

  using c10::musa::MUSACachingAllocator::DeviceStats;
  using c10::musa::MUSACachingAllocator::Stat;
  using c10::musa::MUSACachingAllocator::StatArray;
  using c10::musa::MUSACachingAllocator::StatType;

  const auto statToDict = [](const Stat& stat) {
    py::dict dict;

    dict["current"] = stat.current;
    dict["peak"] = stat.peak;
    dict["allocated"] = stat.allocated;
    dict["freed"] = stat.freed;
    return dict;
  };

  const auto statArrayToDict = [=](const StatArray& statArray) {
    const std::array<const char*, static_cast<size_t>(StatType::NUM_TYPES)>
        statTypeNames = {"all", "small_pool", "large_pool"};
    py::dict dict;
    for (size_t i = 0; i < statTypeNames.size(); ++i) {
      dict[statTypeNames[i]] = statToDict(statArray[i]);
    }
    return dict;
  };

  const DeviceStats stats =
      c10::musa::MUSACachingAllocator::GetDeviceStats(device);

  py::dict result;
  result["num_alloc_retries"] = stats.num_alloc_retries;
  result["num_ooms"] = stats.num_ooms;
  result["max_split_size"] = stats.max_split_size;
  result["allocation"] = statArrayToDict(stats.allocation);
  result["segment"] = statArrayToDict(stats.segment);
  result["active"] = statArrayToDict(stats.active);
  result["inactive_split"] = statArrayToDict(stats.inactive_split);
  result["allocated_bytes"] = statArrayToDict(stats.allocated_bytes);
  result["reserved_bytes"] = statArrayToDict(stats.reserved_bytes);
  result["active_bytes"] = statArrayToDict(stats.active_bytes);
  result["inactive_split_bytes"] = statArrayToDict(stats.inactive_split_bytes);
  result["oversize_allocations"] = statToDict(stats.oversize_allocations);
  result["oversize_segments"] = statToDict(stats.oversize_segments);

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

struct Frame {
  PyCodeObject* code;
  int lasti;
};

struct StackContext : public c10::musa::MUSACachingAllocator::Context {
  std::vector<Frame> frames;
  // Empty if cpp traces weren't enabled
  std::string cpp_frames;
  ~StackContext() {
    py::gil_scoped_acquire acquire;
    for (auto& f : frames) {
      Py_XDECREF((PyObject*)f.code);
    }
  }
  static std::shared_ptr<StackContext> _gather() {
    py::gil_scoped_acquire acquire;
    auto r = std::make_shared<StackContext>();
    PyFrameObject* f = PyEval_GetFrame();
    Py_XINCREF(f);
    while (f) {
      r->frames.emplace_back(Frame{PyFrame_GetCode(f), PyFrame_GetLasti(f)});
      auto f_back = PyFrame_GetBack(f);
      Py_XDECREF(f);
      f = f_back;
    }
    return r;
  }
  static std::shared_ptr<c10::musa::MUSACachingAllocator::Context> gather() {
    return _gather();
  }
  static std::shared_ptr<c10::musa::MUSACachingAllocator::Context>
  gather_with_cpp() {
    auto r = _gather();
    r->cpp_frames = c10::get_backtrace();
    return std::move(r);
  }
};

PyObject* PyMusaMemorySnapshot(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS

  using c10::musa::MUSACachingAllocator::BlockInfo;
  using c10::musa::MUSACachingAllocator::History;
  using c10::musa::MUSACachingAllocator::SegmentInfo;

  py::str device_s = "device";
  py::str address_s = "address";
  py::str total_size_s = "total_size";
  py::str allocated_size_s = "allocated_size";
  py::str active_size_s = "active_size";
  py::str requested_size_s = "requested_size";
  py::str stream_s = "stream";
  py::str segment_type_s = "segment_type";
  py::str large_s = "large";
  py::str small_s = "small";
  py::str size_s = "size";
  py::str state_s = "state";
  py::str active_allocated_s = "active_allocated";
  py::str active_pending_free_s = "active_pending_free";
  py::str inactive_s = "inactive";
  py::str addr_s = "addr";
  py::str real_size_s = "real_size";
  py::str filename_s = "filename";
  py::str name_s = "name";
  py::str line_s = "line";
  py::str frames_s = "frames";
  py::str cpp_frames_s = "cpp_frames";
  py::str history_s = "history";
  py::str blocks_s = "blocks";

  std::unordered_map<StackContext*, py::list> cached_frames;
  const auto get_frames = [&](StackContext* sc) -> py::list {
    auto it = cached_frames.find(sc);
    if (it != cached_frames.end()) {
      return it->second;
    }
    py::list frames;
    for (auto& f : sc->frames) {
      py::dict frame;
      frame[filename_s] =
          py::reinterpret_borrow<py::object>(f.code->co_filename);
      frame[name_s] = py::reinterpret_borrow<py::object>(f.code->co_name);
      frame[line_s] = PyCode_Addr2Line(f.code, f.lasti);
      frames.append(std::move(frame));
    }
    cached_frames.insert({sc, frames});
    return frames;
  };

  const auto segmentInfoToDict = [&](const SegmentInfo& segmentInfo) {
    py::dict segmentDict;
    segmentDict[device_s] = segmentInfo.device;
    segmentDict[address_s] = segmentInfo.address;
    segmentDict[total_size_s] = segmentInfo.total_size;
    segmentDict[allocated_size_s] = segmentInfo.allocated_size;
    segmentDict[active_size_s] = segmentInfo.active_size;
    segmentDict[requested_size_s] = segmentInfo.requested_size;
    segmentDict[stream_s] = int64_t(segmentInfo.stream);
    segmentDict[segment_type_s] = (segmentInfo.is_large ? large_s : small_s);

    py::list blocks;
    for (const auto& blockInfo : segmentInfo.blocks) {
      py::dict blockDict;
      blockDict[size_s] = blockInfo.size;
      blockDict[requested_size_s] = blockInfo.requested_size;
      blockDict[state_s] =
          (blockInfo.allocated
               ? active_allocated_s
               : (blockInfo.active ? active_pending_free_s : inactive_s));
      if (blockInfo.history.size()) {
        py::list history;
        for (const History& h : blockInfo.history) {
          py::dict history_entry;
          history_entry[addr_s] = (int64_t)h.addr;
          history_entry[real_size_s] = h.real_size;
          if (h.context) {
            auto sc = (StackContext*)h.context.get();
            history_entry[frames_s] = get_frames(sc);
            if (!sc->cpp_frames.empty()) {
              history_entry[cpp_frames_s] = py::cast(sc->cpp_frames);
            }
          }
          history.append(std::move(history_entry));
        }
        blockDict[history_s] = std::move(history);
      }
      blocks.append(blockDict);
    }
    segmentDict[blocks_s] = blocks;

    return segmentDict;
  };

  const auto& snapshot = c10::musa::MUSACachingAllocator::GetMemorySnapshot();
  py::list segments;

  for (const auto& segmentInfo : snapshot.segments) {
    segments.append(segmentInfoToDict(segmentInfo));
  }

  py::list traces;
  py::str action_s = "action";
  py::str alloc_s = "alloc";
  py::str free_requested_s = "free_requested";
  py::str free_completed_s = "free_completed";
  py::str segment_alloc_s = "segment_alloc";
  py::str segment_free_s = "segment_free";
  py::str snapshot_s = "snapshot";
  py::str oom_s = "oom";
  py::str device_free_s = "device_free";

  using namespace c10::musa::MUSACachingAllocator;
  auto action_to_str = [&](TraceEntry::Action action) {
    switch (action) {
      case TraceEntry::ALLOC:
        return alloc_s;
      case TraceEntry::FREE_REQUESTED:
        return free_requested_s;
      case TraceEntry::FREE_COMPLETED:
        return free_completed_s;
      case TraceEntry::SEGMENT_ALLOC:
        return segment_alloc_s;
      case TraceEntry::SEGMENT_FREE:
        return segment_free_s;
      case TraceEntry::OOM:
        return oom_s;
      case TraceEntry::SNAPSHOT:
        return snapshot_s;
    }
    throw std::runtime_error("unreachable");
  };

  for (const auto& traceInfo : snapshot.device_traces) {
    py::list trace;
    for (const auto& te : traceInfo) {
      py::dict trace_entry;
      if (te.context_) {
        // without further compression frames can get really large on dump
        auto sc = (StackContext*)te.context_.get();
        trace_entry[frames_s] = get_frames(sc);
        if (!sc->cpp_frames.empty()) {
          trace_entry[cpp_frames_s] = py::cast(sc->cpp_frames);
        }
      }
      trace_entry[action_s] = action_to_str(te.action_);
      trace_entry[TraceEntry::OOM == te.action_ ? device_free_s : addr_s] =
          te.addr_;
      trace_entry[size_s] = te.size_;
      trace_entry[stream_s] = int64_t(te.stream_);
      trace.append(trace_entry);
    }
    traces.append(trace);
  }

  py::dict result;
  result["segments"] = segments;
  result["device_traces"] = traces;

  return result.release().ptr();
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaAttachOutOfMemoryObserver(
    PyObject* _unused,
    PyObject* observer) {
  HANDLE_TH_ERRORS
  Py_XINCREF(observer);
  auto obs = [observer](
                 int64_t device,
                 int64_t alloc,
                 int64_t device_allocated,
                 int64_t device_free) {
    py::gil_scoped_acquire g;
    PyObject* result = PyObject_CallFunction(
        observer, "LLLL", device, alloc, device_allocated, device_free);
    if (!result) {
      throw py::error_already_set();
    }
    Py_XDECREF(result);
  };
  at::musa::lazyInitMUSA();
  c10::musa::MUSACachingAllocator::AttachOutOfMemoryObserver(std::move(obs));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetMemoryFraction(PyObject* _unused, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* fraction_o = nullptr;
  PyObject* device_o = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &fraction_o, &device_o)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "set_memory_fraction",
        1,
        "(double fraction, int device);");
    return nullptr;
  }
  double fraction = PyFloat_AsDouble(fraction_o);
  int64_t device = PyLong_AsLongLong(device_o);

  c10::musa::MUSACachingAllocator::SetMemoryFraction(fraction, device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

PyObject* PyMusaresetPeakMemoryStats(PyObject* _unused, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to reset_peak_memory_stats");
  const int device = (int)THPUtils_unpackLong(arg);
  c10::musa::MUSACachingAllocator::ResetPeakStats(device);
  END_HANDLE_TH_ERRORS
  Py_RETURN_NONE;
}

static void RegisterMusaDeviceProperties(PyObject* module) {
  // Add _musaDeviceProperties class to torch_musa._MUSAC.
  auto m = py::handle(module).cast<py::module>();

  // Set musa version
  m.attr("_musa_version") = py::str(std::to_string(MUSA_VERSION));

  py::class_<musaDeviceProp>(m, "_MusaDeviceProperties")
      .def_readonly("name", &musaDeviceProp::name)
      .def_readonly("major", &musaDeviceProp::major)
      .def_readonly("minor", &musaDeviceProp::minor)
      .def_readonly("is_multi_gpu_board", &musaDeviceProp::isMultiGpuBoard)
      .def_readonly("is_integrated", &musaDeviceProp::integrated)
      .def_readonly(
          "multi_processor_count", &musaDeviceProp::multiProcessorCount)
      .def_readonly("total_memory", &musaDeviceProp::totalGlobalMem)
      .def("__repr__", [](const musaDeviceProp& prop) {
        std::ostringstream stream;
        stream << "_MusaDeviceProperties(name='" << prop.name << "', major='"
               << prop.major << ", minor=" << prop.minor
               << ", total_memory=" << prop.totalGlobalMem / (1024 * 1024)
               << "MB, multi_processor_count=" << prop.multiProcessorCount
               << ")";
        return stream.str();
      });
  m.def(
      "_musa_recordMemoryHistory",
      [](bool enabled,
         bool record_context,
         bool record_context_cpp,
         Py_ssize_t alloc_trace_max_entries,
         bool alloc_trace_record_context) {
        at::musa::lazyInitMUSA();
        c10::musa::MUSACachingAllocator::RecordHistory(
            enabled,
            record_context ? (record_context_cpp ? StackContext::gather_with_cpp
                                                 : StackContext::gather)
                           : nullptr,
            alloc_trace_max_entries,
            alloc_trace_record_context);
      });
}

static void BindGetDeviceProperties(PyObject* module) {
  // Add method to torch_musa
  auto m = py::handle(module).cast<py::module>();
  m.def(
      "_get_device_properties",
      [](int64_t device) -> musaDeviceProp* {
        return at::musa::getDeviceProperties(device);
      },
      py::return_value_policy::reference);
}

static PyObject* PyMusaInitExtension(
    PyObject* /* unused */,
    PyObject* /* unused */) {
#if C10_ASAN_ENABLED
  TORCH_WARN(
      "torch.musa: your pytorch binary has address sanitizer (asan) built in, "
      "asan is currently not compatible with torch.musa module, "
      "you might get unexpected behavior (eg. out of memory, crash, etc.), "
      "please rebuild pytorch without asan if you need to use this module");
#endif
  HANDLE_TH_ERRORS
  TORCH_INTERNAL_ASSERT(!in_bad_fork); // Handled at python level
  poison_fork();
  at::musa::lazyInitMUSA();

  auto m = THPObjectPtr(PyImport_ImportModule("torch_musa"));
  if (!m)
    throw python_error();

  bool has_half = true;

  auto set_module_attr = [&](const char* name, PyObject* v) {
    // PyObject_SetAttrString doesn't steal reference. So no need to incref.
    if (PyObject_SetAttrString(m, name, v) < 0) {
      throw python_error();
    }
  };

  set_module_attr("has_half", has_half ? Py_True : Py_False);

  const int64_t num_gpus = c10::musa::device_count();
  auto default_musa_generators = PyTuple_New(static_cast<Py_ssize_t>(num_gpus));
  for (const auto i : c10::irange(num_gpus)) {
    auto cast_gen = (THPGenerator*)THPGenerator_initDefaultGenerator(
        at::musa::detail::getDefaultMUSAGenerator(i));
    // This reference is meant to be given away, so no need to incref here.
    PyTuple_SetItem(default_musa_generators, i, (PyObject*)cast_gen);
  }
  set_module_attr("default_generators", default_musa_generators);
  BindGetDeviceProperties(m);

  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

#ifdef USE_MCCL
void AddMusaProcessGroupMethods(PyObject* module) {
  registerProcessGroupMCCL(module);
}
#endif

PyObject* PyMusaSetDevice(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(THPUtils_checkLong(arg), "invalid argument to setDevice");
  int64_t device = THPUtils_unpackLong(arg);
  torch::utils::musa_lazy_init();
  c10::musa::set_device(device);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDevice(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  torch::utils::musa_lazy_init();
  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::musa::current_device());
  return THPUtils_packInt32(device);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaCanDeviceAccessPeer(PyObject* /* unused */, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* arg1 = nullptr;
  PyObject* arg2 = nullptr;
  if (!PyArg_ParseTuple(args, "OO", &arg1, &arg2)) {
    THPUtils_invalidArguments(
        args,
        nullptr,
        "can_device_peer_access",
        1,
        "(int device, int peer_device);");
    return nullptr;
  }
  THPUtils_assert(
      THPUtils_checkLong(arg1), "invalid argument to canDeviceAccessPeer");
  THPUtils_assert(
      THPUtils_checkLong(arg2), "invalid argument to canDeviceAccessPeer");
  int64_t device = THPUtils_unpackLong(arg1);
  int64_t peer_device = THPUtils_unpackLong(arg2);
  torch::utils::musa_lazy_init();
  auto can_access = at::musa::canDeviceAccessPeer(device, peer_device);
  return PyBool_FromLong(can_access);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDeviceCount(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  poison_fork();
  return THPUtils_packUInt64(at::musa::device_count());
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaExchangeDevice(PyObject* /* unused */, PyObject* arg) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(arg), "invalid argument to exchangeDevice");
  int64_t device = THPUtils_unpackLong(arg);
  torch::utils::musa_lazy_init();
  auto exchanged_device = at::musa::exchangeDevice(device);
  return THPUtils_packInt32(exchanged_device);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSynchronize(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  c10::musa::Synchronize();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetDefaultStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getDefaultStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = at::musa::getDefaultMUSAStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetCurrentStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  auto stream = at::musa::getCurrentMUSAStream(device);
  PyObject* output_tuple = PyTuple_New(3);
  PyTuple_SetItem(
      output_tuple, 0, THPUtils_packInt64(static_cast<int64_t>(stream.id())));
  PyTuple_SetItem(
      output_tuple,
      1,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_index())));
  PyTuple_SetItem(
      output_tuple,
      2,
      THPUtils_packInt64(static_cast<int64_t>(stream.device_type())));
  return output_tuple;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaGetCurrentRawStream(
    PyObject* /* unused */,
    PyObject* device_index) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(device_index), "invalid argument to getCurrentStream");
  int64_t device = THPUtils_unpackLong(device_index);
  return PyLong_FromVoidPtr(at::musa::getCurrentMUSAStream(device).stream());
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetStream(
    PyObject* /* unused */,
    PyObject* args,
    PyObject* kwargs) {
  HANDLE_TH_ERRORS
  int64_t stream_id = 0;
  int64_t device_index = 0;
  int64_t device_type = 0;

  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  constexpr const char* kwlist[] = {
      "stream_id", "device_index", "device_type", nullptr};
  if (!PyArg_ParseTupleAndKeywords(
          args,
          kwargs,
          "|LLL",
          const_cast<char**>(kwlist),
          &stream_id,
          &device_index,
          &device_type)) {
  }

  auto stream = at::musa::MUSAStream::unpack3(
      stream_id, device_index, static_cast<c10::DeviceType>(device_type));

  // NOLINTNEXTLINE(bugprone-signed-char-misuse)
  auto device = static_cast<int>(c10::musa::current_device());
  if (device != stream.device_index()) {
    c10::musa::set_device(stream.device_index());
  }
  at::musa::setCurrentMUSAStream(stream);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaIPCCollect(PyObject* _unused, PyObject* noargs) {
  HANDLE_TH_ERRORS
  bool freed_memory = torch::musa::MusaIPCCollect();
  return PyBool_FromLong(freed_memory);
  END_HANDLE_TH_ERRORS
}

static PyObject* PyMusaIsInBadFork(PyObject* self, PyObject* noargs) {
  HANDLE_TH_ERRORS
  return PyBool_FromLong(in_bad_fork);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSleep(PyObject* /* unused */, PyObject* cycles) {
  HANDLE_TH_ERRORS
  THPUtils_assert(
      THPUtils_checkLong(cycles), "torch.musa._sleep(): expected 'int'");
  at::musa::sleep(THPUtils_unpackLong(cycles));
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaMudnnVersion(PyObject* /* unused */, PyObject* /* unused */) {
  HANDLE_TH_ERRORS
  return THPUtils_packInt64(MUDNN_VERSION);
  END_HANDLE_TH_ERRORS
}

PyObject* PyMusaSetDefaultDtype(PyObject* _unused, PyObject* type) {
  HANDLE_TH_ERRORS
  torch::musa::PySetDefaultDtype(type);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef MusaMemoryMethods[] = {
    {"_musa_emptyCache", PyMusaEmptyCache, METH_NOARGS, nullptr},
    {"_musa_memoryStats", PyMusaMemoryStats, METH_O, nullptr},
    {"_musa_resetPeakStats", PyMusaResetPeakStats, METH_NOARGS, nullptr},
    {"_musa_memorySnapshot", PyMusaMemorySnapshot, METH_NOARGS, nullptr},
    {"_musa_attach_out_of_memory_observer",
     PyMusaAttachOutOfMemoryObserver,
     METH_O,
     nullptr},
    {"_musa_setMemoryFraction", PyMusaSetMemoryFraction, METH_VARARGS, nullptr},
    {"_musa_resetPeakMemoryStats", PyMusaresetPeakMemoryStats, METH_O, nullptr},
    {nullptr}};

static PyMethodDef MusaStreamMethods[] = {
    {"_musa_getDefaultStream", PyMusaGetDefaultStream, METH_O, nullptr},
    {"_musa_getCurrentStream", PyMusaGetCurrentStream, METH_O, nullptr},
    {"_musa_getCurrentRawStream", PyMusaGetCurrentRawStream, METH_O, nullptr},
    {"_musa_setStream",
     castPyCFunctionWithKeywords(PyMusaSetStream),
     METH_VARARGS | METH_KEYWORDS,
     nullptr},
    {"_musa_sleep", PyMusaSleep, METH_O, nullptr},
    {nullptr}};

static PyMethodDef MusaDeviceMethods[] = {
    {"_musa_init", PyMusaInitExtension, METH_NOARGS, nullptr},
    {"_musa_getDevice", PyMusaGetDevice, METH_NOARGS, nullptr},
    {"_musa_getDeviceCount", PyMusaGetDeviceCount, METH_NOARGS, nullptr},
    {"_musa_setDevice", PyMusaSetDevice, METH_O, nullptr},
    {"_musa_exchangeDevice", PyMusaExchangeDevice, METH_O, nullptr},
    {"_musa_canDeviceAccessPeer",
     PyMusaCanDeviceAccessPeer,
     METH_VARARGS,
     nullptr},
    {"_musa_synchronize", PyMusaSynchronize, METH_NOARGS, nullptr},
    {"_mudnn_version", PyMusaMudnnVersion, METH_NOARGS, nullptr},
    {"_musa_ipc_collect", PyMusaIPCCollect, METH_NOARGS, nullptr},
    {"_musa_isInBadFork", PyMusaIsInBadFork, METH_NOARGS, nullptr},
    {nullptr}};

static PyMethodDef MusaDtypeMethods[] = {
    {"_set_default_dtype", PyMusaSetDefaultDtype, METH_O, nullptr},
    {nullptr}};

PyObject* module;
static std::vector<PyMethodDef> methods;

PyObject* InitMusaModule() {
  at::internal::lazy_init_num_threads();

  // Initialize some Python bindings.
  torch::musa::InitializePythonBindings();

  AddPyMethodDefs(methods, torch::musa::GetTensorMethods());
  AddPyMethodDefs(methods, GetStorageSharingMethods());
  AddPyMethodDefs(methods, MusaDeviceMethods);
  AddPyMethodDefs(methods, MusaStreamMethods);
  AddPyMethodDefs(methods, MusaMemoryMethods);
  AddPyMethodDefs(methods, MusaDtypeMethods);
  AddPyMethodDefs(methods, at::musa::autocast::GetAutocastMethods());
  AddPyMethodDefs(methods, at::musa::GetContextMethods());
  AddPyMethodDefs(methods, at::musa::GetStorageMethods());

  static struct PyModuleDef musa_module = {
      PyModuleDef_HEAD_INIT, "torch_musa._MUSAC", nullptr, -1, methods.data()};
  module = PyModule_Create(&musa_module);

  THMPStream_init(module);
  THMPEvent_init(module);
  torch::musa::python::InitCommMethods(module);

#ifdef USE_MCCL
  AddMusaProcessGroupMethods(module);
#endif
  // Register MUSA device properties
  RegisterMusaDeviceProperties(module);

  return module;
}
