#pragma once

#include <array>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10d::musa_intra_node_comm {

using namespace c10d::symmetric_memory;

constexpr size_t kMaxDevices = 8;

using MtLinkMesh = std::array<std::array<size_t, kMaxDevices>, kMaxDevices>;

enum class Topology : uint8_t {
  UNKNOWN = 0,
  FULLY_CONNECTED = 1,
};

enum class AllReduceAlgo : uint8_t {
  NONE = 0,
  ONE_SHOT = 1,
  TWO_SHOT = 2,
  LOW_CONTENTION = 3,
};

class IntraNodeComm : public c10::intrusive_ptr_target {
 public:
  IntraNodeComm(
      c10::intrusive_ptr<c10d::Store> store,
      size_t rank,
      size_t worldSize,
      std::string processGroupName,
      std::optional<size_t> bufferSize = std::nullopt);

  ~IntraNodeComm() override;

  static bool isEnabled();

  bool rendezvous();

  c10::DeviceIndex deviceIndex() const;

  void setProcessGroupName(const std::string& processGroupName);

  enum class CollectiveType : uint8_t {
    ALL_REDUCE = 0,
    ALL_GATHER = 1,
    REDUCE_SCATTER = 2,
  };

  AllReduceAlgo selectAllReduceAlgo(
      const at::Tensor& input,
      const char* reduceOp = "sum");

  at::Tensor allReduce(
      const at::Tensor& input,
      AllReduceAlgo algo,
      const char* reduceOp = "sum");

  bool canUseAllGather(const at::Tensor& input, const at::Tensor& output) const;

  at::Tensor allGather(const at::Tensor& output, const at::Tensor& input);

  bool canUseReduceScatter(const at::Tensor& input, const at::Tensor& output)
      const;

  at::Tensor reduceScatter(
      const at::Tensor& output,
      const at::Tensor& input,
      const std::string& reduceOp);

 private:
  struct SymmetricMemoryWorkspace {
    void* symmetricMemoryPtr = nullptr;
    c10::intrusive_ptr<SymmetricMemory> symmetricMemory = nullptr;
    size_t size = 0;
  };

  SymmetricMemoryWorkspace& ensureSymmetricMemory(
      CollectiveType collectiveType,
      size_t size);

  void releaseSymmetricMemory(SymmetricMemoryWorkspace& workspace);

  at::Tensor oneShotAllReduce(
      const at::Tensor& input,
      c10::musa::MUSAStream& stream);

  at::Tensor twoShotAllReduce(
      const at::Tensor& input,
      c10::musa::MUSAStream& stream);

  at::Tensor lowContentionAllReduce(
      const at::Tensor& input,
      const char* reduceOp,
      c10::musa::MUSAStream& stream);

  c10::intrusive_ptr<Store> store_;
  size_t rank_;
  size_t worldSize_;
  size_t initialBufferSize_;
  // Private symmetric-memory group for internally staged workspaces.
  std::string groupName_;
  // Atomic immutable snapshot used by user-created symmetric-memory tensors.
  std::shared_ptr<const std::string> processGroupName_;

  std::shared_ptr<const std::string> getProcessGroupName() const;

  bool isInitialized_ = false;
  int deviceIdx_{0};
  Topology topology_ = Topology::UNKNOWN;
  std::array<SymmetricMemoryWorkspace, 3> symmetricMemoryWorkspaces_;
  std::array<std::mutex, 3> collectiveMutexes_;
};

int64_t getIntraNodeCommUsageCounter();

bool isIntraNodeMemoryOptimModeEnabled();

bool isIntraNodeCommSupported();

} // namespace c10d::musa_intra_node_comm
