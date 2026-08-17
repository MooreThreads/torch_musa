#include "torch_musa/csrc/distributed/Register.h"
#include <torch/csrc/distributed/c10d/Types.hpp>
#include "c10/util/Exception.h"
#include "pybind11/pybind11.h"
#include "torch/csrc/distributed/c10d/Types.hpp"
#include "torch_musa/csrc/distributed/ProcessGroupMCCL.h"

#include <c10/util/intrusive_ptr.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <pybind11/stl.h>
#include <optional>

namespace {

// This is a intrusive helper from pytorch.
template <typename T>
class IntrusivePtrNoGilDestructor {
 public:
  IntrusivePtrNoGilDestructor() = default;
  IntrusivePtrNoGilDestructor(const IntrusivePtrNoGilDestructor&) = default;
  IntrusivePtrNoGilDestructor(IntrusivePtrNoGilDestructor&&) = default;
  IntrusivePtrNoGilDestructor& operator=(const IntrusivePtrNoGilDestructor&) =
      default;
  IntrusivePtrNoGilDestructor& operator=(IntrusivePtrNoGilDestructor&&) =
      default;
  /* implicit */ IntrusivePtrNoGilDestructor(c10::intrusive_ptr<T> impl)
      : impl_(std::move(impl)) {}
  // This ctor is very important; see
  // https://github.com/pybind/pybind11/issues/2957
  explicit IntrusivePtrNoGilDestructor(T* impl)
      : impl_(c10::intrusive_ptr<T>::unsafe_steal_from_new(impl)) {}
  ~IntrusivePtrNoGilDestructor() {
    if (impl_) {
      if (PyGILState_Check()) {
        pybind11::gil_scoped_release release;
        impl_.reset();
      } else {
        impl_.reset();
      }
    }
  }
  T& operator*() const noexcept {
    return *impl_;
  }
  T* operator->() const noexcept {
    return impl_.get();
  }
  C10_NODISCARD T* get() const noexcept {
    return impl_.get();
  }
  void reset() noexcept {
    impl_.reset();
  }
  operator bool() const noexcept {
    return impl_;
  }

 private:
  c10::intrusive_ptr<T> impl_;
};

} // namespace

namespace py = pybind11;
PYBIND11_DECLARE_HOLDER_TYPE(T, IntrusivePtrNoGilDestructor<T>, true);
PYBIND11_DECLARE_HOLDER_TYPE(T, c10::intrusive_ptr<T>, true)

template <typename T>
using intrusive_ptr_class_ = py::class_<T, c10::intrusive_ptr<T>>;

template <typename T>
using intrusive_ptr_no_gil_destructor_class_ =
    py::class_<T, IntrusivePtrNoGilDestructor<T>>;
/*END OF COPY CODE!*/

void registerProcessGroupMCCL(PyObject* mod) {
  py::object module = py::module::import("torch.distributed");
  py::object register_backend = module.attr("Backend").attr("register_backend");
  register_backend(
      "mccl",
      py::cpp_function(
          [](const c10d::DistributedBackendOptions& dist_opts,
             py::object backend_opts_obj) {
            c10::intrusive_ptr<c10d::ProcessGroupMCCL::Options> backend_opts;
            if (!backend_opts_obj.is_none()) {
              backend_opts = backend_opts_obj.cast<
                  c10::intrusive_ptr<c10d::ProcessGroupMCCL::Options>>();
            }
            return c10d::ProcessGroupMCCL::MCCLcreator(dist_opts, backend_opts);
          },
          py::arg("dist_opts"),
          py::arg("backend_opts") = py::none()),
      /* extended_api */ true,
      "musa"); // returns a python ProcessGroupMCCL
  auto distributed_c10d = py::module::import("torch._C._distributed_c10d");
  auto backend = distributed_c10d.attr("Backend");

  if (!py::hasattr(distributed_c10d, "_dump_mccl_trace_json")) {
    distributed_c10d.def(
        "_dump_mccl_trace_json",
        [](std::optional<bool> includeCollectives,
           std::optional<bool> onlyActive) {
          return py::bytes(::c10d::dump_mccl_trace_json(
              includeCollectives.value_or(true), onlyActive.value_or(false)));
        },
        py::arg("includeCollectives") = std::optional<bool>(),
        py::arg("onlyActive") = std::optional<bool>(),
        R"(
        Arguments:
              includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
              onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
              Stringified json work traces.
              Default settings return everything.
      )");
  }

  if (!py::hasattr(distributed_c10d, "_dump_mccl_trace")) {
    distributed_c10d.def(
        "_dump_mccl_trace",
        [](std::optional<bool> includeCollectives,
           std::optional<bool> includeStackTraces,
           std::optional<bool> onlyActive) {
          return py::bytes(::c10d::dump_mccl_trace(
              includeCollectives.value_or(true),
              includeStackTraces.value_or(true),
              onlyActive.value_or(false)));
        },
        py::arg("includeCollectives") = std::optional<bool>(),
        py::arg("includeStackTraces") = std::optional<bool>(),
        py::arg("onlyActive") = std::optional<bool>(),
        R"(
        Arguments:
              includeCollectives(bool, optional): Whether to include collective work traces. Default is True.
              includeStackTraces(bool, optional): Whether to include stacktraces in the collective work traces. Default is True.
              onlyActive (bool, optional): Whether to only include active collective work traces. Default is False.
        Returns:
              Stringified pickle work traces.
              Default settings return everything.
      )");
  }

  if (!py::hasattr(distributed_c10d, "_reset_fr_recording_mccl")) {
    distributed_c10d.def(
        "_reset_fr_recording_mccl",
        []() { ::c10d::reset_mccl_trace(); },
        "API to reset MCCL Flight Recorder recording.");
  }

  auto processGroupMCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupMCCL>(
          module,
          "ProcessGroupMCCL",
          backend); // Define a python ProcessGroupMCCL
  processGroupMCCL
      .def( // Define the Init function of python ProcessGroupMCCL
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      c10::intrusive_ptr<::c10d::ProcessGroupMCCL::Options>
                          options) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            return c10::make_intrusive<::c10d::ProcessGroupMCCL>(
                store, rank, size, std::move(options));
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("options"),
          R"(Create a new ProcessGroupMCCL instance.)")
      .def(
          py::init([](const c10::intrusive_ptr<::c10d::Store>& store,
                      int rank,
                      int size,
                      const std::chrono::milliseconds& timeout) {
            // gil_scoped_release is not safe as a call_guard in init.
            // https://github.com/pybind/pybind11/issues/5473
            py::gil_scoped_release nogil{};

            auto options = ::c10d::ProcessGroupMCCL::Options::create();
            options->is_high_priority_stream = false;
            options->timeout = timeout;
            return c10::make_intrusive<::c10d::ProcessGroupMCCL>(
                store, rank, size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("size"),
          py::arg("timeout") = ::c10d::kProcessGroupMCCLDefaultTimeout,
          R"(Create a new ProcessGroupMCCL instance.)")
      .def(
          "_comm_ptr",
          &::c10d::ProcessGroupMCCL::getCommPtr,
          R"(
            Get the communicator of the current device.

            .. warning ::
                Unsafe to use. The collectives launched into the communicator
                externally outside ProcessGroupMCCL are not monitored by the
                watchdog. Please do not modify or free the communicator as the
                communicator is managed by the ProcessGroupMCCL. Please also
                check the readiness of the communicator before launching any
                collectives into the communicator.
            )")
      .def("_group_start", &::c10d::ProcessGroupMCCL::groupStart)
      .def("_group_end", &::c10d::ProcessGroupMCCL::groupEnd)
      .def("_start_time_estimate", &::c10d::ProcessGroupMCCL::startTimeEstimate)
      .def("_end_time_estimate", &::c10d::ProcessGroupMCCL::endTimeEstimate)
      .def("comm_split_count", &::c10d::ProcessGroupMCCL::getCommSplitCounter)
      .def(
          "_set_default_timeout",
          [](const c10::intrusive_ptr<::c10d::ProcessGroupMCCL>& self,
             std::chrono::milliseconds timeout) {
            self->getOptions()->timeout = timeout;
          },
          py::arg("timeout"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_add_ephemeral_timeout",
          [](const c10::intrusive_ptr<::c10d::ProcessGroupMCCL>& self,
             const std::chrono::milliseconds& timeout) {
            self->addEphemeralTimeout(timeout);
          },
          py::arg("timeout"))
      .def(
          "_verify_work_timeout",
          [](const c10::intrusive_ptr<::c10d::ProcessGroupMCCL>& self,
             const c10::intrusive_ptr<::c10d::Work>& work,
             const std::chrono::milliseconds& timeout) {
            return self->verifyWorkTimeoutForTest(work, timeout);
          },
          py::arg("work"),
          py::arg("timeout"))
      .def_property_readonly(
          "options",
          &::c10d::ProcessGroupMCCL::getOptions,
          R"(Return the options used to create this ProcessGroupMCCL instance.)")
      .def_property_readonly(
          "uid", &::c10d::ProcessGroupMCCL::getUid, R"(Return the uid.)")
      .def_property(
          "bound_device_id",
          &::c10d::ProcessGroupMCCL::getBoundDeviceId,
          &::c10d::ProcessGroupMCCL::setBoundDeviceId,
          R"(Return the bound device id.)")
      .def(
          "perform_nocolor_split",
          &::c10d::ProcessGroupMCCL::performNocolorSplit)
      .def(
          "register_mem_pool",
          &::c10d::ProcessGroupMCCL::registerMemPool,
          py::arg("pool"),
          py::arg("symm") = false)
      .def("deregister_mem_pool", &::c10d::ProcessGroupMCCL::deregisterMemPool)
      .def(
          "_is_initialized",
          &::c10d::ProcessGroupMCCL::isInitialized,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_error",
          &::c10d::ProcessGroupMCCL::getError,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "_set_enable_nan_check",
          [](const c10::intrusive_ptr<::c10d::ProcessGroupMCCL>& self,
             bool enable_nan_check) {
            self->setEnableNanCheck(enable_nan_check);
          },
          py::arg("enable_nan_check"),
          py::call_guard<py::gil_scoped_release>());

#ifdef MCCL_HAS_CTA_POLICY
  processGroupMCCL.def_property_readonly_static(
      "MCCL_CTA_POLICY_DEFAULT",
      [](const py::object&) { return MCCL_CTA_POLICY_DEFAULT; });
  processGroupMCCL.def_property_readonly_static(
      "MCCL_CTA_POLICY_EFFICIENCY",
      [](const py::object&) { return MCCL_CTA_POLICY_EFFICIENCY; });
  processGroupMCCL.def_property_readonly_static(
      "MCCL_CTA_POLICY_ZERO",
      [](const py::object&) { return MCCL_CTA_POLICY_ZERO; });
#endif // MCCL_HAS_CTA_POLICY

#ifdef MCCL_HAS_CONFIG
  py::class_<mcclConfig_t>(
      processGroupMCCL,
      "MCCLConfig",
      R"(mcclConfig_t data type for configuring MCCL communicators.)")
      .def(py::init([]() {
        mcclConfig_t defaultCfg = MCCL_CONFIG_INITIALIZER;
        return std::make_unique<mcclConfig_t>(defaultCfg);
      }))
      .def_readwrite("blocking", &mcclConfig_t::blocking)
      .def_readwrite("cga_cluster_size", &mcclConfig_t::cgaClusterSize)
      .def_readwrite("min_ctas", &mcclConfig_t::minCTAs)
      .def_readwrite("max_ctas", &mcclConfig_t::maxCTAs)
#ifdef MCCL_HAS_CTA_POLICY
      .def_readwrite("cta_policy", &mcclConfig_t::CTAPolicy)
#endif
      .def(
          "__copy__",
          [](const mcclConfig_t& self) { return mcclConfig_t(self); })
      .def(
          "__deepcopy__",
          [](const mcclConfig_t& self, const py::dict& memo) {
            return mcclConfig_t(self);
          },
          py::arg("memo"));
#endif // MCCL_HAS_CONFIG

  auto backendOptions = backend.attr("Options");

  intrusive_ptr_class_<::c10d::ProcessGroupMCCL::Options>(
      processGroupMCCL,
      "Options",
      backendOptions,
      R"(ProcessGroup options for the MCCL backend)")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
#ifdef MCCL_HAS_CONFIG
      .def_readwrite("config", &::c10d::ProcessGroupMCCL::Options::config)
#endif
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupMCCL::Options::is_high_priority_stream)
      .def_readwrite(
          "split_from", &::c10d::ProcessGroupMCCL::Options::split_from)
      .def_readwrite(
          "split_color", &::c10d::ProcessGroupMCCL::Options::split_color);
}
