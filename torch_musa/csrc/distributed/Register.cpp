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
  // auto backend = module.attr("Backend");

  auto processGroupMCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupMCCL>(
          module,
          "ProcessGroupMCCL",
          backend); // Define a python ProcessGroupMCCL
  processGroupMCCL
      .def( // Define the Init function of python ProcessGroupMCCL
          py::init([](c10d::PrefixStore& store,
                      int rank,
                      int world_size,
                      // std::chrono::duration<float>& timeout) {
                      std::chrono::milliseconds timeout) {
            auto options = c10d::ProcessGroupMCCL::Options::create();
            options->timeout = timeout;
            return c10::make_intrusive<::c10d::ProcessGroupMCCL>(
                store.getUnderlyingStore(), rank, world_size, options);
          }),
          py::arg("store"),
          py::arg("rank"),
          py::arg("world_size"),
          py::arg("timeout"))
      .def(
          "abort",
          [](const c10::intrusive_ptr<::c10d::ProcessGroupMCCL>& self) {
            return self->abort();
          },
          py::call_guard<py::gil_scoped_release>());
}
