#include "torch_musa/csrc/distributed/Register.h"
#include "torch_musa/csrc/distributed/ProcessGroupMCCL.h"

#include <c10/util/intrusive_ptr.h>
#include <pybind11/cast.h>
#include <pybind11/chrono.h>
#include <torch/library.h>
#include <thread>
#include "mccl.h"

namespace c10d {

namespace ops {

c10::intrusive_ptr<Work> send_musa(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t dstRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->send(tensor_vec, static_cast<int>(dstRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t srcRank,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->recv(tensor_vec, static_cast<int>(srcRank), static_cast<int>(tag));
}

c10::intrusive_ptr<Work> recv_any_source_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t tag) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->recvAnysource(tensor_vec, static_cast<int>(tag));
}

c10::intrusive_ptr<Work> reduce_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->reduce(
          tensor_vec,
          ReduceOptions{
              *reduce_op.get(),
              root_rank,
              root_tensor,
              std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> broadcast_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t root_tensor,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->broadcast(
              tensor_vec,
              BroadcastOptions{
                  root_rank, root_tensor, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> allreduce_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<c10d::ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  // c10d::ProcessGroupMCCL process_group;
  auto tensor_vec = tensors.vec();
  //->getBackend(c10::DeviceType::PrivateUse1)
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->allreduce(
              tensor_vec,
              AllreduceOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(tensor_vec), work);
}

c10::intrusive_ptr<Work> allreduce_coalesced_musa_(
    at::TensorList tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto tensor_vec = tensors.vec();
  AllreduceCoalescedOptions opts = AllreduceCoalescedOptions{};
  opts.reduceOp = *reduce_op.get();
  opts.timeout = std::chrono::milliseconds(timeout);

  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->allreduce_coalesced(tensor_vec, opts);
}

std::tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>
allgather_musa_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    at::TensorList input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->allgather(
              const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
              input_tensors_vec,
              AllgatherOptions{std::chrono::milliseconds(timeout)});

  // Copy output tensors (not storage) so that this can be used in a functional
  // manner
  return std::
      tuple<std::vector<std::vector<at::Tensor>>, c10::intrusive_ptr<Work>>(
          output_tensors, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _allgather_base_musa_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto work = process_group->getBackend(c10::DeviceType::PrivateUse1)
                  ->_allgather_base(output_tensor, input_tensor);

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

c10::intrusive_ptr<Work> allgather_coalesced_musa_(
    const std::vector<std::vector<at::Tensor>>& output_lists,
    const at::TensorList& input_list,
    const c10::intrusive_ptr<ProcessGroup>& process_group) {
  auto input_list_vec = input_list.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->allgather_coalesced(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_lists),
          input_list_vec);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>
reduce_scatter_musa_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->reduce_scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      output_tensors_vec, work);
}

std::tuple<at::Tensor, c10::intrusive_ptr<Work>> _reduce_scatter_base_musa_(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const c10::intrusive_ptr<ReduceOp>& reduce_op,
    int64_t timeout) {
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->_reduce_scatter_base(
              output_tensor,
              input_tensor,
              ReduceScatterOptions{
                  *reduce_op.get(), std::chrono::milliseconds(timeout)});

  return std::tuple<at::Tensor, c10::intrusive_ptr<Work>>(output_tensor, work);
}

c10::intrusive_ptr<Work> gather_musa_(
    const std::vector<std::vector<at::Tensor>>& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto input_tensors_vec = input_tensors.vec();
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->gather(
          const_cast<std::vector<std::vector<at::Tensor>>&>(output_tensors),
          input_tensors_vec,
          GatherOptions{root_rank, std::chrono::milliseconds(timeout)});
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> scatter_musa_(
    const at::TensorList& output_tensors,
    const std::vector<std::vector<at::Tensor>>& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t root_rank,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto work =
      process_group->getBackend(c10::DeviceType::PrivateUse1)
          ->scatter(
              output_tensors_vec,
              const_cast<std::vector<std::vector<at::Tensor>>&>(input_tensors),
              ScatterOptions{root_rank, std::chrono::milliseconds(timeout)});

  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>> alltoall_musa_(
    const at::TensorList& output_tensors,
    const at::TensorList& input_tensors,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    int64_t timeout) {
  auto output_tensors_vec = output_tensors.vec();
  auto input_tensors_vec = input_tensors.vec();
  auto work = process_group->getBackend(c10::DeviceType::PrivateUse1)
                  ->alltoall(
                      output_tensors_vec,
                      input_tensors_vec,
                      AllToAllOptions{std::chrono::milliseconds(timeout)});
  return std::tuple<std::vector<at::Tensor>, c10::intrusive_ptr<Work>>(
      std::move(output_tensors_vec), work);
}

c10::intrusive_ptr<Work> alltoall_base_musa_(
    at::Tensor& output,
    at::Tensor& input,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    std::vector<int64_t> output_split_sizes,
    std::vector<int64_t> input_split_sizes,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->alltoall_base(
          output,
          input,
          output_split_sizes,
          input_split_sizes,
          AllToAllOptions{std::chrono::milliseconds(timeout)});
}

c10::intrusive_ptr<Work> barrier_musa(
    at::Tensor /* unused */,
    const c10::intrusive_ptr<ProcessGroup>& process_group,
    const std::vector<int64_t>& device_ids,
    int64_t timeout) {
  return process_group->getBackend(c10::DeviceType::PrivateUse1)
      ->barrier(BarrierOptions{device_ids, std::chrono::milliseconds(timeout)});
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("send", &ops::send_musa);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("recv_", &ops::recv_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("allreduce_", &ops::allreduce_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("recv_any_source_", &ops::recv_any_source_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("reduce_", &ops::reduce_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("broadcast_", &ops::broadcast_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("allreduce_coalesced_", &ops::allreduce_coalesced_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("allgather_", &ops::allgather_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("_allgather_base_", &ops::_allgather_base_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("allgather_coalesced_", &ops::allgather_coalesced_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("reduce_scatter_", &ops::reduce_scatter_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("_reduce_scatter_base_", &ops::_reduce_scatter_base_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("gather_", &ops::gather_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("scatter_", &ops::scatter_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("alltoall_", &ops::alltoall_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("alltoall_base_", &ops::alltoall_base_musa_);
}

TORCH_LIBRARY_IMPL(c10d, AutogradPrivateUse1, m) {
  m.impl("barrier", &ops::barrier_musa);
}

} // namespace ops

} // namespace c10d

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
          py::arg("timeout"))); // returns a python ProcessGroupMCCL
  auto backend =
      py::module::import("torch._C._distributed_c10d").attr("Backend");
  // auto backend = module.attr("Backend");

  auto processGroupMCCL =
      intrusive_ptr_no_gil_destructor_class_<::c10d::ProcessGroupMCCL>(
          module,
          "ProcessGroupMCCL",
          backend); // Define a python ProcessGroupMCCL
  processGroupMCCL.def( // Define the Init function of python ProcessGroupMCCL
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
      py::arg("timeout"));
}
