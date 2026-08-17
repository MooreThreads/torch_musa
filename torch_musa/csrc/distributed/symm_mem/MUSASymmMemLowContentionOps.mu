#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <memory>
#include <unordered_map>

#include "c10/core/DeviceType.h"
#include "c10/core/TensorOptions.h"
#include "c10/util/Exception.h"

#include "c10/util/intrusive_ptr.h"
#include "torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp"
#include "torch/library.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAException.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"
#include "torch_musa/csrc/core/driver_api.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryTypes.hpp"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryUtils.hpp"

namespace {

using namespace c10d::symmetric_memory;

// low_contention_all_gather/reduce_scatter implementation start
struct ACECollCommunicator {
  enum collAlgo {
    ALL_GATHER_PULL_MODE = 0,
    REDUCE_SCATTER_PULL_MODE,
    REDUCE_SCATTER_PUSH_MODE,
    END
  };
  static constexpr uint32_t signal_numel_per_coll{
      max_musa_p2p_domain_size * 2}; // 2 means: ready and complete
  static constexpr uint32_t signal_size{
      signal_numel_per_coll * sizeof(uint64_t) * collAlgo::END};

  struct ACEGraphInfo {
    MUgraph graph;
    MUgraphExec graphExec;
    std::vector<std::vector<MUgraphNode>> nodes;
    MUcontext ctx;

    std::vector<MUSA_MEM_TRANSFER_NODE_PARAMS> copyParams;
    std::vector<MUSA_MEM_ATOMIC_NODE_PARAMS> atomicParams;
    std::vector<MUSA_MEM_WAIT_WRITE_NODE_PARAMS> waitParams;

    uint64_t seqNum{0};
  };

  ACECollCommunicator(
      int32_t rank,
      int32_t world_size,
      int device_idx,
      const std::string& group_name)
      : rank_(rank), world_size_(world_size) {
    auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
    // alloc an extra signal memory for simplity
    void* signal_ptr = allocator->alloc(
        ACECollCommunicator::signal_size, device_idx, std::nullopt);
    signal_symm_mem_ = std::move(allocator->rendezvous(signal_ptr, group_name));
  }

  int32_t rank_;
  int32_t world_size_;
  c10::intrusive_ptr<SymmetricMemory> signal_symm_mem_;

  struct ACEGraphInfo all_gather_pull_mode_info_;
  struct ACEGraphInfo reduce_scatter_pull_mode_info_;
  struct ACEGraphInfo reduce_scatter_push_mode_info_;
};

void pullModeAllGatherInit(ACECollCommunicator* communicator) {
  TORCH_CHECK(communicator != nullptr, "nullptr communicator was passed");

  int rank = communicator->rank_;
  int world_size = communicator->world_size_;
  auto signal_ptrs = communicator->signal_symm_mem_->get_buffer_ptrs();
  auto& graph_all_gather_info = communicator->all_gather_pull_mode_info_;
  auto* driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(
      driver_api->muCtxGetCurrent_(&(graph_all_gather_info.ctx)));
  C10_MUSA_DRIVER_CHECK(
      driver_api->muGraphCreate_(&(graph_all_gather_info.graph), 0));

  graph_all_gather_info.nodes.resize(world_size);
  // Keep params indexed by peer id so UpdateParams can safely use peer-based
  // indexing.
  graph_all_gather_info.waitParams.clear();
  graph_all_gather_info.copyParams.clear();
  graph_all_gather_info.waitParams.resize(world_size * 2);
  graph_all_gather_info.copyParams.resize(world_size);
  const int nsubnodes = 5;

  auto& graph = graph_all_gather_info.graph;
  auto& ctx = graph_all_gather_info.ctx;
  auto& nodes = graph_all_gather_info.nodes;

  const int signal_offset =
      ACECollCommunicator::collAlgo::ALL_GATHER_PULL_MODE *
      ACECollCommunicator::signal_numel_per_coll;
  uint64_t* local_ready_ptrs =
      static_cast<uint64_t*>(signal_ptrs[rank]) + signal_offset;
  uint64_t* local_complete_ptrs = local_ready_ptrs + max_musa_p2p_domain_size;

  for (int i = 0; i < world_size; i++) {
    int peer = (rank + i) % world_size;
    nodes[peer].resize(nsubnodes);

    uint64_t* peer_ready_ptrs =
        static_cast<uint64_t*>(signal_ptrs[peer]) + signal_offset;
    uint64_t* peer_complete_ptrs = peer_ready_ptrs + max_musa_p2p_domain_size;
    if (peer == rank) {
      // root
      MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS barrier_param{
          (MUdeviceptr)(peer_ready_ptrs + rank),
          1,
          MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
      C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
          &nodes[rank][0], graph, nullptr, 0, &barrier_param, ctx));
    }

    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_ready_param{
        (MUdeviceptr)(peer_ready_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][1], graph, &nodes[rank][0], 1, &wait_ready_param, ctx));

    MUSA_MEM_TRANSFER_NODE_PARAMS transfer_param{
        (MUdeviceptr)(local_ready_ptrs + peer),
        (MUdeviceptr)(peer_ready_ptrs + peer),
        sizeof(uint64_t)};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemTransferNode_(
        &nodes[peer][2], graph, &nodes[peer][1], 1, &transfer_param, ctx));

    MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS complete_signal_param{
        (MUdeviceptr)(peer_complete_ptrs + rank),
        1,
        MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
        &nodes[peer][3],
        graph,
        &nodes[peer][2],
        1,
        &complete_signal_param,
        ctx));

    // wait data read finish
    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_complete_param{
        (MUdeviceptr)(local_complete_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][4], graph, &nodes[peer][3], 1, &wait_complete_param, ctx));

    graph_all_gather_info.waitParams[peer * 2] = wait_ready_param;
    graph_all_gather_info.waitParams[peer * 2 + 1] = wait_complete_param;
    graph_all_gather_info.copyParams[peer] = transfer_param;
  }
  C10_MUSA_DRIVER_CHECK(driver_api->muGraphInstantiateWithFlags_(
      &graph_all_gather_info.graphExec, graph, 1));
}

void reduceScatterInit(ACECollCommunicator* communicator, bool pull_mode) {
  int rank = communicator->rank_;
  int world_size = communicator->world_size_;
  auto& graph_reduce_scatter_info = pull_mode
      ? communicator->reduce_scatter_pull_mode_info_
      : communicator->reduce_scatter_push_mode_info_;
  auto signal_ptrs = communicator->signal_symm_mem_->get_buffer_ptrs();
  auto* driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(
      driver_api->muCtxGetCurrent_(&(graph_reduce_scatter_info.ctx)));
  C10_MUSA_DRIVER_CHECK(
      driver_api->muGraphCreate_(&(graph_reduce_scatter_info.graph), 0));

  graph_reduce_scatter_info.nodes.resize(world_size);

  auto& graph = graph_reduce_scatter_info.graph;
  auto& ctx = graph_reduce_scatter_info.ctx;
  auto& nodes = graph_reduce_scatter_info.nodes;

  int signal_offset = pull_mode
      ? ACECollCommunicator::collAlgo::REDUCE_SCATTER_PULL_MODE *
          ACECollCommunicator::signal_numel_per_coll
      : ACECollCommunicator::collAlgo::REDUCE_SCATTER_PUSH_MODE *
          ACECollCommunicator::signal_numel_per_coll;
  uint64_t* local_ready_ptrs =
      static_cast<uint64_t*>(signal_ptrs[rank]) + signal_offset;
  uint64_t* local_complete_ptrs = local_ready_ptrs + max_musa_p2p_domain_size;

  const int nsubnodes = 5;
  for (int i = 0; i < world_size; i++) {
    nodes[i].resize(nsubnodes);
  }
  // root barrier param
  MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS barrier_param{
      (MUdeviceptr)(local_ready_ptrs + rank),
      1,
      MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
  C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
      &nodes[rank][0], graph, nullptr, 0, &barrier_param, ctx));
  for (int peer = 0; peer < world_size; peer++) {
    uint64_t* peer_ready_ptrs =
        static_cast<uint64_t*>(signal_ptrs[peer]) + signal_offset;
    uint64_t* peer_complete_ptrs = peer_ready_ptrs + max_musa_p2p_domain_size;

    // wait buffer (send_buf/recv_buf) ready
    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_ready_param{
        (MUdeviceptr)(peer_ready_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][1], graph, &nodes[rank][0], 1, &wait_ready_param, ctx));

    // atomic reducation
    MUSA_MEM_ATOMIC_NODE_PARAMS reduce_param{
        .dst = 0LL,
        .src = 0LL,
        .elementCount = 1,
        .operation = MU_ATOMIC_TYPE_ATOMIC_ADD64};
    if (pull_mode) {
      reduce_param.dst = (MUdeviceptr)(local_complete_ptrs + rank);
      reduce_param.src = (MUdeviceptr)(peer_complete_ptrs + rank);
    } else {
      reduce_param.dst = (MUdeviceptr)(peer_complete_ptrs + peer);
      reduce_param.src = (MUdeviceptr)(local_ready_ptrs + peer);
    }
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicNode_(
        &nodes[peer][2], graph, &nodes[peer][1], 1, &reduce_param, ctx));

    // reducation complete signal
    MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS complete_signal_param{
        (MUdeviceptr)(peer_complete_ptrs + rank),
        1,
        MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
        &nodes[peer][3],
        graph,
        &nodes[peer][2],
        1,
        &complete_signal_param,
        ctx));

    // wait complete signal
    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_complete_param{
        (MUdeviceptr)(local_complete_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][4], graph, &nodes[peer][3], 1, &wait_complete_param, ctx));

    graph_reduce_scatter_info.waitParams.push_back(wait_ready_param);
    graph_reduce_scatter_info.waitParams.push_back(wait_complete_param);
    graph_reduce_scatter_info.atomicParams.push_back(reduce_param);
  }
  C10_MUSA_DRIVER_CHECK(driver_api->muGraphInstantiateWithFlags_(
      &graph_reduce_scatter_info.graphExec, graph, 1));
}

void pullModeGraphAllGatherUpdateParams(
    ACECollCommunicator* communicator,
    std::vector<void*> src,
    void* dst,
    const size_t byte_count,
    const size_t offset) {
  TORCH_CHECK(communicator != nullptr, "invalid communicator");
  TORCH_CHECK(dst != nullptr, "invalid dst ptr");

  int rank = communicator->rank_;
  int world_size = communicator->world_size_;
  auto& graph_all_gather_info = communicator->all_gather_pull_mode_info_;
  auto* driver_api = c10::musa::DriverAPI::get();

  graph_all_gather_info.seqNum++;
  for (int i = 0; i < world_size; i++) {
    int peer = (rank + i) % world_size;
    TORCH_CHECK(src[peer] != nullptr, "invalid src ptr");
    auto& ready_param = graph_all_gather_info.waitParams[peer * 2];
    ready_param.value = graph_all_gather_info.seqNum;

    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_all_gather_info.graphExec,
        graph_all_gather_info.nodes[peer][1],
        &ready_param,
        graph_all_gather_info.ctx));

    graph_all_gather_info.copyParams[peer].dst =
        (MUdeviceptr)((char*)dst + byte_count * peer);
    // the src ptr is the base address of SymmetricMemory, but SymmetricMemory
    // might be managed by CachingAllocator, so an offset is needed here.
    graph_all_gather_info.copyParams[peer].src =
        (MUdeviceptr)((char*)(src[peer]) + offset);
    graph_all_gather_info.copyParams[peer].ByteCount = byte_count;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemTransferNodeSetParams_(
        graph_all_gather_info.graphExec,
        graph_all_gather_info.nodes[peer][2],
        &graph_all_gather_info.copyParams[peer],
        graph_all_gather_info.ctx));

    auto& finish_param = graph_all_gather_info.waitParams[peer * 2 + 1];
    finish_param.value = graph_all_gather_info.seqNum;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_all_gather_info.graphExec,
        graph_all_gather_info.nodes[peer][4],
        &finish_param,
        graph_all_gather_info.ctx));
  }
}

void graphReduceScatterUpdateParams(
    ACECollCommunicator* communicator,
    void** src,
    void** dst,
    const size_t element_count,
    const size_t element_size,
    MUatomicType atomic_type,
    const size_t offset_in_bytes,
    bool pull_mode) {
  TORCH_CHECK(communicator != nullptr, "invalid communicator");
  const size_t chunk_size_in_bytes = element_count * element_size;
  int rank = communicator->rank_;
  int world_size = communicator->world_size_;
  auto& graph_reduce_scatter_info = pull_mode
      ? communicator->reduce_scatter_pull_mode_info_
      : communicator->reduce_scatter_push_mode_info_;
  auto* driver_api = c10::musa::DriverAPI::get();
  graph_reduce_scatter_info.seqNum++;

  for (int i = 0; i < world_size; i++) {
    auto& ready_param = graph_reduce_scatter_info.waitParams[i * 2];
    ready_param.value = graph_reduce_scatter_info.seqNum;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_reduce_scatter_info.graphExec,
        graph_reduce_scatter_info.nodes[i][1],
        &ready_param,
        graph_reduce_scatter_info.ctx));

    auto& reduce_param = graph_reduce_scatter_info.atomicParams[i];
    // the dst ptr is the base address of SymmetricMemory (or src ptr is the
    // base address of SymmetricMemory, in pull mode), but SymmetricMemory might
    // be managed by CachingAllocator, so an offset is needed here.
    if (pull_mode) {
      reduce_param.dst = (MUdeviceptr)(dst[0]);
      reduce_param.src =
          (MUdeviceptr)((char*)src[i] + offset_in_bytes + chunk_size_in_bytes * rank);
    } else {
      reduce_param.dst = (MUdeviceptr)((char*)(dst[i]) + offset_in_bytes);
      reduce_param.src = (MUdeviceptr)((char*)src[0] + chunk_size_in_bytes * i);
    }
    reduce_param.elementCount = element_count;
    reduce_param.operation = atomic_type;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemAtomicNodeSetParams_(
        graph_reduce_scatter_info.graphExec,
        graph_reduce_scatter_info.nodes[i][2],
        &reduce_param,
        graph_reduce_scatter_info.ctx));

    auto& complete_param = graph_reduce_scatter_info.waitParams[i * 2 + 1];
    complete_param.value = graph_reduce_scatter_info.seqNum;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_reduce_scatter_info.graphExec,
        graph_reduce_scatter_info.nodes[i][4],
        &complete_param,
        graph_reduce_scatter_info.ctx));
  }
}

void graphCollectiveInit(ACECollCommunicator* communicator) {
  pullModeAllGatherInit(communicator);
  // Although the push‑mode reduce‑scatter has lower performance than the
  // pull‑mode implementation, I implemented it as well because it imposes fewer
  // constraints on SymmetricMemory. This variant should still be beneficial for
  // FSDP2.
  reduceScatterInit(communicator, /*pull_mode=*/false);
  reduceScatterInit(communicator, /*pull_mode=*/true);
}

static std::unordered_map<std::string, std::unique_ptr<ACECollCommunicator>>
    group_to_communicator;

void lowContentionAllGather(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const std::string& group_name) {
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> input_hdl =
      c10d::symmetric_memory::rendezvous(input_tensor, group_name);

  bool input_is_symm_mem = input_hdl == nullptr ? false : true;
  TORCH_CHECK(
      input_is_symm_mem, "input tensor must be allocated from SymmetricMemory");

  auto it = group_to_communicator.find(group_name);
  auto stream = at::musa::getCurrentMUSAStream();
  size_t input_bytes_ = input_tensor.numel() * input_tensor.element_size();

  auto rank = input_hdl->get_rank();
  auto world_size = input_hdl->get_world_size();
  if (it == group_to_communicator.end()) {
    // init once
    TORCH_CHECK(
        input_hdl->world_within_direct_access(),
        "only support intra node communication");
    it = group_to_communicator
             .emplace(
                 group_name,
                 std::make_unique<ACECollCommunicator>(
                     rank, world_size, input_tensor.get_device(), group_name))
             .first;
    graphCollectiveInit((it->second).get());
  }
  void* dst_ptr = output_tensor.data_ptr();
  auto* communicator = (it->second).get();
  pullModeGraphAllGatherUpdateParams(
      communicator,
      input_hdl->get_buffer_ptrs(),
      dst_ptr,
      input_bytes_,
      input_hdl->get_offset());
  C10_MUSA_CHECK(musaGraphLaunch(
      communicator->all_gather_pull_mode_info_.graphExec, stream));
}

MUatomicType getMUAtomicType(at::ScalarType datatype) {
  if (datatype == at::ScalarType::Float) {
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD_F32;
  } else if (datatype == at::ScalarType::BFloat16) {
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD_BF16;
  } else if (datatype == at::ScalarType::Half) {
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD_HF16;
  } else {
    TORCH_CHECK(
        false,
        "Dtype of output_tensor only support flaot, bfloat16 and float16");
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD64;
  }
}

void lowContentionReduceScatter(
    at::Tensor& output_tensor,
    at::Tensor& input_tensor,
    const std::string& reduce_op,
    const std::string& group_name) {
  TORCH_CHECK(
      reduce_op == "sum" || reduce_op == "avg",
      "Only sum and avg reduce are supported, but got: ",
      reduce_op);

  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> output_hdl =
      c10d::symmetric_memory::rendezvous(output_tensor, group_name);
  c10::intrusive_ptr<c10d::symmetric_memory::SymmetricMemory> input_hdl =
      c10d::symmetric_memory::rendezvous(input_tensor, group_name);
  bool input_is_symm_mem = input_hdl == nullptr ? false : true;
  bool output_is_symm_mem = output_hdl == nullptr ? false : true;

  TORCH_CHECK(
      input_is_symm_mem || output_is_symm_mem,
      "Tensors not within any SymmetricMemory allocation, ",
      "is the tensor allocated from SymmetricMemory?");

  auto it = group_to_communicator.find(group_name);
  auto stream = at::musa::getCurrentMUSAStream();

  const size_t output_size = output_tensor.numel();
  const size_t element_size = output_tensor.element_size();
  const size_t output_bytes = output_size * element_size;
  const size_t input_bytes = input_tensor.numel() * input_tensor.element_size();

  bool use_pull_mode = !output_is_symm_mem || input_is_symm_mem;
  auto rank = use_pull_mode ? input_hdl->get_rank() : output_hdl->get_rank();
  auto world_size = use_pull_mode ? input_hdl->get_world_size()
                                  : output_hdl->get_world_size();

  // do some check
  TORCH_CHECK(input_tensor.numel() == output_tensor.numel() * world_size);
  TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type());
  TORCH_CHECK(input_tensor.is_contiguous() && output_tensor.is_contiguous());
  auto input_iptr = reinterpret_cast<uintptr_t>(input_tensor.data_ptr());
  auto output_iptr = reinterpret_cast<uintptr_t>(output_tensor.data_ptr());
  TORCH_CHECK(
      !(output_iptr >= input_iptr &&
        ((output_iptr - input_iptr) < input_bytes)),
      "inplace reduce_scatter is not supported");

  if (it == group_to_communicator.end()) {
    TORCH_CHECK(
        use_pull_mode ? input_hdl->world_within_direct_access()
                      : output_hdl->world_within_direct_access(),
        "only support intra node communication");
    it = group_to_communicator
             .emplace(
                 group_name,
                 std::make_unique<ACECollCommunicator>(
                     rank, world_size, output_tensor.get_device(), group_name))
             .first;
    graphCollectiveInit((it->second).get());
  }
  auto* communicator = (it->second).get();

  std::vector<void*> src_ptrs = use_pull_mode
      ? input_hdl->get_buffer_ptrs()
      : std::vector<void*>{input_tensor.data_ptr()};
  std::vector<void*> dst_ptrs = use_pull_mode
      ? std::vector<void*>{output_tensor.data_ptr()}
      : output_hdl->get_buffer_ptrs();

  C10_MUSA_CHECK(
      musaMemsetAsync(output_tensor.data_ptr(), 0, output_bytes, stream));
  graphReduceScatterUpdateParams(
      communicator,
      src_ptrs.data(),
      dst_ptrs.data(),
      output_size,
      element_size,
      getMUAtomicType(input_tensor.scalar_type()),
      use_pull_mode ? input_hdl->get_offset() : output_hdl->get_offset(),
      use_pull_mode);
  if (use_pull_mode) {
    C10_MUSA_CHECK(musaGraphLaunch(
        communicator->reduce_scatter_pull_mode_info_.graphExec, stream));
  } else {
    C10_MUSA_CHECK(musaGraphLaunch(
        communicator->reduce_scatter_push_mode_info_.graphExec, stream));
  }

  // final reducation
  c10::musa::MUSAStreamGuard guard(stream);
  if (reduce_op == "avg") {
    float factor = 1.0f / static_cast<float>(world_size);
    output_tensor.mul_(factor);
  }
}

// low contention_all_gather/reduce_scatter implementation end

} // anonymous namespace

TORCH_LIBRARY_FRAGMENT(symm_mem, m) {
  // PyTorch already define `_low_contention_all_gather` at python frontend, and
  // implement it using multi stream copy, since our low_contention_all_gather
  // implementation is different from PyTorch, to distinguish from it, we define
  // another schema here.
  m.def(
      "low_contention_all_gather(Tensor output_tensor, Tensor input_tensor, str group_name) -> ()");
  m.def(
      "low_contention_reduce_scatter(Tensor output_tensor, Tensor input_tensor, str reduce_op, str group_name) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, PrivateUse1, m) {
  m.impl("low_contention_all_gather", lowContentionAllGather);
  m.impl("low_contention_reduce_scatter", lowContentionReduceScatter);
}
