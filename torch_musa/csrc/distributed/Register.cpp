#include "torch_musa/csrc/distributed/Register.h"
#include "torch_musa/csrc/distributed/ProcessGroupMCCL.h"

#include <c10/util/intrusive_ptr.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <torch/library.h>
#include <thread>
#include "mccl.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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
          &c10d::ProcessGroupMCCL::MCCLcreator,
          py::arg("store"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("timeout")),
      false,
      "musa"); // returns a python ProcessGroupMCCL
  auto backend =
      py::module::import("torch._C._distributed_c10d").attr("Backend");

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
      .def("_group_start", &::c10d::ProcessGroupMCCL::groupStart)
      .def("_group_end", &::c10d::ProcessGroupMCCL::groupEnd)
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
      // .def("register_mem_pool", &::c10d::ProcessGroupMCCL::registerMemPool)
      // .def(
      //     "deregister_mem_pool",
      //     &::c10d::ProcessGroupMCCL::deregisterMemPool)
      .def(
          "_is_initialized",
          &::c10d::ProcessGroupMCCL::isInitialized,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "get_error",
          &::c10d::ProcessGroupMCCL::getError,
          py::call_guard<py::gil_scoped_release>());

  auto backendOptions = backend.attr("Options");

  intrusive_ptr_class_<::c10d::ProcessGroupMCCL::Options>(
      processGroupMCCL,
      "Options",
      backendOptions,
      R"(ProcessGroup options for the MCCL backend)")
      .def(py::init<bool>(), py::arg("is_high_priority_stream") = false)
      // .def_readwrite("config", &::c10d::ProcessGroupNCCL::Options::config)
      .def_readwrite(
          "is_high_priority_stream",
          &::c10d::ProcessGroupMCCL::Options::is_high_priority_stream)
      .def_readwrite(
          "split_from", &::c10d::ProcessGroupMCCL::Options::split_from)
      .def_readwrite(
          "split_color", &::c10d::ProcessGroupMCCL::Options::split_color)
      .def_readwrite(
          "global_ranks_in_group",
          &::c10d::ProcessGroupMCCL::Options::global_ranks_in_group)
      .def_readwrite(
          "group_name", &::c10d::ProcessGroupMCCL::Options::group_name);
}
