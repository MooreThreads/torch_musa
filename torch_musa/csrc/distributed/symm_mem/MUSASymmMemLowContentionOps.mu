#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <cstdint>
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

// low-contention collective implementations start
struct ACECollCommunicator {
  enum collAlgo {
    ALL_GATHER_PULL_MODE = 0,
    REDUCE_SCATTER_PULL_MODE,
    REDUCE_SCATTER_PUSH_MODE,
    ALL_REDUCE,
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
  struct ACEGraphInfo all_reduce_info_;
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

void allReduceInit(ACECollCommunicator* communicator) {
  TORCH_CHECK(communicator != nullptr, "nullptr communicator was passed");

  const int rank = communicator->rank_;
  const int world_size = communicator->world_size_;
  auto signal_ptrs = communicator->signal_symm_mem_->get_buffer_ptrs();
  auto& graph_info = communicator->all_reduce_info_;
  auto* driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(driver_api->muCtxGetCurrent_(&graph_info.ctx));
  C10_MUSA_DRIVER_CHECK(driver_api->muGraphCreate_(&graph_info.graph, 0));

  graph_info.nodes.resize(world_size);
  graph_info.waitParams.resize(world_size * 3);
  graph_info.copyParams.resize(world_size);
  graph_info.atomicParams.resize(world_size);

  auto& graph = graph_info.graph;
  auto& ctx = graph_info.ctx;
  auto& nodes = graph_info.nodes;

  const int signal_offset = ACECollCommunicator::collAlgo::ALL_REDUCE *
      ACECollCommunicator::signal_numel_per_coll;
  uint64_t* local_ready_ptrs =
      static_cast<uint64_t*>(signal_ptrs[rank]) + signal_offset;
  uint64_t* local_complete_ptrs = local_ready_ptrs + max_musa_p2p_domain_size;

  constexpr int nsubnodes = 8;
  for (int peer = 0; peer < world_size; peer++) {
    nodes[peer].resize(nsubnodes);
    uint64_t* peer_ready_ptrs =
        static_cast<uint64_t*>(signal_ptrs[peer]) + signal_offset;
    uint64_t* peer_complete_ptrs = peer_ready_ptrs + max_musa_p2p_domain_size;

    // Signal each peer that this rank's input is ready.
    MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS barrier_param{
        (MUdeviceptr)(peer_ready_ptrs + rank),
        1,
        MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
        &nodes[peer][0], graph, nullptr, 0, &barrier_param, ctx));

    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_ready_param{
        (MUdeviceptr)(local_ready_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][1], graph, &nodes[peer][0], 1, &wait_ready_param, ctx));
    graph_info.waitParams[peer * 3] = wait_ready_param;

    if (peer == rank) {
      continue;
    }

    // Reduce peer's rank-local slice into this rank's slice.
    MUSA_MEM_ATOMIC_NODE_PARAMS reduce_param{
        .dst = (MUdeviceptr)(local_complete_ptrs + rank),
        .src = (MUdeviceptr)(peer_complete_ptrs + rank),
        .elementCount = 1,
        .operation = MU_ATOMIC_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicNode_(
        &nodes[peer][2], graph, &nodes[peer][1], 1, &reduce_param, ctx));

    MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS reduce_complete_param{
        (MUdeviceptr)(local_complete_ptrs + rank),
        1,
        MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
        &nodes[peer][3],
        graph,
        &nodes[peer][2],
        1,
        &reduce_complete_param,
        ctx));

    // Wait until peer has reduced its own slice.
    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_reduce_param{
        (MUdeviceptr)(peer_complete_ptrs + peer),
        static_cast<uint64_t>(world_size - 1),
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][4], graph, &nodes[peer][3], 1, &wait_reduce_param, ctx));

    MUSA_MEM_TRANSFER_NODE_PARAMS gather_param{
        (MUdeviceptr)(local_ready_ptrs + peer),
        (MUdeviceptr)(peer_ready_ptrs + peer),
        sizeof(uint64_t)};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemTransferNode_(
        &nodes[peer][5], graph, &nodes[peer][4], 1, &gather_param, ctx));

    // Tell peer that its reduced slice has been consumed.
    MUSA_MEM_ATOMIC_VALUE_NODE_PARAMS gather_complete_param{
        (MUdeviceptr)(peer_complete_ptrs + rank),
        1,
        MU_ATOMIC_VALUE_TYPE_ATOMIC_ADD64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemAtomicValueNode_(
        &nodes[peer][6],
        graph,
        &nodes[peer][5],
        1,
        &gather_complete_param,
        ctx));

    MUSA_MEM_WAIT_WRITE_NODE_PARAMS wait_complete_param{
        (MUdeviceptr)(local_complete_ptrs + peer),
        0,
        MU_STREAM_WAIT_VALUE_EQ,
        MU_STREAM_MEM_OP_WAIT_VALUE_64};
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphAddMemWaitWriteNode_(
        &nodes[peer][7], graph, &nodes[peer][6], 1, &wait_complete_param, ctx));

    graph_info.waitParams[peer * 3 + 1] = wait_reduce_param;
    graph_info.waitParams[peer * 3 + 2] = wait_complete_param;
    graph_info.copyParams[peer] = gather_param;
    graph_info.atomicParams[peer] = reduce_param;
  }

  C10_MUSA_DRIVER_CHECK(driver_api->muGraphInstantiateWithFlags_(
      &graph_info.graphExec, graph, 1));
}

void pullModeGraphAllGatherUpdateParams(
    ACECollCommunicator* communicator,
    std::vector<void*> src,
    void* dst,
    const size_t byte_count,
    const size_t src_base_offset,
    bool src_offset_by_peer) {
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
    // Non-inplace reads from the input offset. Standard inplace reads peer p's
    // rank-local chunk from output_base + p * chunk_bytes.
    const size_t src_offset =
        src_base_offset + (src_offset_by_peer ? byte_count * peer : 0);
    graph_all_gather_info.copyParams[peer].src =
        (MUdeviceptr)((char*)(src[peer]) + src_offset);
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

void graphAllReduceUpdateParams(
    ACECollCommunicator* communicator,
    const std::vector<void*>& buffer_ptrs,
    void* input_ptr,
    const size_t element_count,
    const size_t element_size,
    MUatomicType atomic_type,
    const size_t input_offset) {
  const int rank = communicator->rank_;
  const int world_size = communicator->world_size_;

  const size_t chunk_numel = element_count / world_size;
  const size_t chunk_bytes = chunk_numel * element_size;
  auto& graph_info = communicator->all_reduce_info_;
  auto* driver_api = c10::musa::DriverAPI::get();
  graph_info.seqNum++;

  auto* local_input = static_cast<char*>(input_ptr);
  void* local_reduce_ptr = local_input + chunk_bytes * rank;
  for (int peer = 0; peer < world_size; peer++) {
    auto& ready_param = graph_info.waitParams[peer * 3];
    ready_param.value = graph_info.seqNum;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_info.graphExec,
        graph_info.nodes[peer][1],
        &ready_param,
        graph_info.ctx));

    if (peer == rank) {
      continue;
    }

    auto& reduce_param = graph_info.atomicParams[peer];
    reduce_param.dst = (MUdeviceptr)local_reduce_ptr;
    reduce_param.src =
        (MUdeviceptr)(static_cast<char*>(buffer_ptrs[peer]) + input_offset + chunk_bytes * rank);
    reduce_param.elementCount = chunk_numel;
    reduce_param.operation = atomic_type;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemAtomicNodeSetParams_(
        graph_info.graphExec,
        graph_info.nodes[peer][2],
        &reduce_param,
        graph_info.ctx));

    auto& reduce_complete_param = graph_info.waitParams[peer * 3 + 1];
    reduce_complete_param.value =
        graph_info.seqNum * static_cast<uint64_t>(world_size - 1);
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_info.graphExec,
        graph_info.nodes[peer][4],
        &reduce_complete_param,
        graph_info.ctx));

    auto& gather_param = graph_info.copyParams[peer];
    gather_param.dst = (MUdeviceptr)(local_input + chunk_bytes * peer);
    gather_param.src =
        (MUdeviceptr)(static_cast<char*>(buffer_ptrs[peer]) + input_offset + chunk_bytes * peer);
    gather_param.ByteCount = chunk_bytes;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemTransferNodeSetParams_(
        graph_info.graphExec,
        graph_info.nodes[peer][5],
        &gather_param,
        graph_info.ctx));

    auto& finish_param = graph_info.waitParams[peer * 3 + 2];
    finish_param.value = graph_info.seqNum;
    C10_MUSA_DRIVER_CHECK(driver_api->muGraphExecMemWaitWriteNodeSetParams_(
        graph_info.graphExec,
        graph_info.nodes[peer][7],
        &finish_param,
        graph_info.ctx));
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
  allReduceInit(communicator);
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
  // rendezvous(tensor) uses tensor.storage().data_ptr() to build the handle.
  // For a view tensor, get_offset() points to the storage base, not the view's
  // data_ptr(), so add storage_offset to recover the actual input offset.
  const size_t input_offset = input_hdl->get_offset() +
      input_tensor.storage_offset() * input_tensor.element_size();
  const size_t output_bytes =
      output_tensor.numel() * output_tensor.element_size();
  const auto input_iptr = reinterpret_cast<uintptr_t>(input_tensor.data_ptr());
  const auto output_iptr =
      reinterpret_cast<uintptr_t>(output_tensor.data_ptr());
  const bool has_overlap = input_iptr < output_iptr + output_bytes &&
      output_iptr < input_iptr + input_bytes_;
  const bool standard_inplace =
      input_iptr == output_iptr + input_bytes_ * static_cast<size_t>(rank);
  TORCH_CHECK(
      !has_overlap || standard_inplace,
      "low_contention_all_gather only supports standard inplace all-gather "
      "when input and output overlap: input must be "
      "output.chunk(world_size)[rank]");
  const size_t src_base_offset = standard_inplace
      ? input_offset - input_bytes_ * static_cast<size_t>(rank)
      : input_offset;
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
      src_base_offset,
      standard_inplace);
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
  const size_t symm_mem_offset = use_pull_mode ? input_hdl->get_offset() +
          input_tensor.storage_offset() * input_tensor.element_size()
                                               : output_hdl->get_offset() +
          output_tensor.storage_offset() * output_tensor.element_size();

  C10_MUSA_CHECK(
      musaMemsetAsync(output_tensor.data_ptr(), 0, output_bytes, stream));
  graphReduceScatterUpdateParams(
      communicator,
      src_ptrs.data(),
      dst_ptrs.data(),
      output_size,
      element_size,
      getMUAtomicType(input_tensor.scalar_type()),
      symm_mem_offset,
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

void lowContentionAllReduce(
    at::Tensor& input_tensor,
    const std::string& reduce_op,
    const std::string& group_name) {
  TORCH_CHECK(
      reduce_op == "sum" || reduce_op == "avg",
      "Only sum and avg reduce are supported, but got: ",
      reduce_op);
  TORCH_CHECK(
      input_tensor.is_contiguous(),
      "low_contention_allreduce: input must be contiguous");

  auto input_hdl = c10d::symmetric_memory::rendezvous(input_tensor, group_name);
  TORCH_CHECK(
      input_hdl != nullptr,
      "low_contention_allreduce: input must be allocated from SymmetricMemory");

  const int rank = input_hdl->get_rank();
  const int world_size = input_hdl->get_world_size();
  TORCH_CHECK(
      input_tensor.numel() % world_size == 0,
      "low_contention_allreduce: input numel must be divisible by world size");
  const auto atomic_type = getMUAtomicType(input_tensor.scalar_type());
  if (input_tensor.numel() == 0) {
    return;
  }

  auto it = group_to_communicator.find(group_name);
  if (it == group_to_communicator.end()) {
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

  auto* communicator = (it->second).get();
  TORCH_CHECK(
      communicator->rank_ == rank && communicator->world_size_ == world_size,
      "low_contention_allreduce: process group metadata changed");
  const size_t input_offset = input_hdl->get_offset() +
      input_tensor.storage_offset() * input_tensor.element_size();
  graphAllReduceUpdateParams(
      communicator,
      input_hdl->get_buffer_ptrs(),
      input_tensor.data_ptr(),
      input_tensor.numel(),
      input_tensor.element_size(),
      atomic_type,
      input_offset);

  auto stream = at::musa::getCurrentMUSAStream();
  C10_MUSA_CHECK(
      musaGraphLaunch(communicator->all_reduce_info_.graphExec, stream));
  c10::musa::MUSAStreamGuard guard(stream);
  if (reduce_op == "avg") {
    input_tensor.mul_(1.0f / static_cast<float>(world_size));
  }
}

// low-contention collective implementations end

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
  m.def(
      "low_contention_allreduce(Tensor(a!) input_tensor, str reduce_op, str group_name) -> ()");
}

TORCH_LIBRARY_IMPL(symm_mem, PrivateUse1, m) {
  m.impl("low_contention_all_gather", lowContentionAllGather);
  m.impl("low_contention_reduce_scatter", lowContentionReduceScatter);
  m.impl("low_contention_allreduce", lowContentionAllReduce);
}
