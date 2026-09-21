#include "torch_musa/csrc/distributed/symm_mem/intra_node_comm.hpp"

#include <limits.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdlib>

#include <cstring>
#include <sstream>
#include <unordered_set>

#include <c10/util/Logging.h>
#include <c10/util/env.h>
#include <torch/csrc/distributed/c10d/PrefixStore.hpp>
#include <torch/csrc/distributed/c10d/Utils.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/DMAConnectivity.hpp>

#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace c10d::musa_intra_node_comm {

static std::vector<std::string> ENABLE_INTRA_NODE_COMM = {
    "ENABLE_INTRA_NODE_COMM"};
static std::vector<std::string> INTRA_NODE_MEMORY_OPTIM_MODE = {
    "INTRA_NODE_MEMORY_OPTIM_MODE"};
// Forces detectTopology() to return Topology::FULLY_CONNECTED, so
// IntraNodeComm can be used even without MTLink connection. This is only used
// for testing purposes.
static std::vector<std::string> TEST_INTRA_NODE_COMM = {"TEST_INTRA_NODE_COMM"};

static int intraNodeCommIdx = 0;
constexpr const char* kMtLinkConnectionType = "mtlink";

static MtLinkMesh getMtLinkMesh(const std::vector<int>& rankToDeviceIdx) {
  auto connectivity = detect_dma_connectivity(
      c10::DeviceType::PrivateUse1, kMtLinkConnectionType);
  MtLinkMesh mtlinkMesh = {};
  for (size_t srcRank = 0; srcRank < kMaxDevices; ++srcRank) {
    for (size_t dstRank = 0; dstRank < kMaxDevices; ++dstRank) {
      if (srcRank < rankToDeviceIdx.size() &&
          dstRank < rankToDeviceIdx.size()) {
        mtlinkMesh[srcRank][dstRank] =
            connectivity
                ->matrix[rankToDeviceIdx[srcRank]][rankToDeviceIdx[dstRank]];
      }
    }
  }
  return mtlinkMesh;
}

static Topology detectTopology(const MtLinkMesh& mtlinkMesh, size_t worldSize) {
  if (getCvarBool(TEST_INTRA_NODE_COMM, false)) {
    return Topology::FULLY_CONNECTED;
  }

  bool fullyConnected = true;
  for (size_t i = 0; i < worldSize - 1; ++i) {
    for (size_t j = i + 1; j < worldSize; ++j) {
      if (mtlinkMesh[i][j] == 0 || mtlinkMesh[j][i] == 0) {
        fullyConnected = false;
      }
    }
  }
  if (fullyConnected) {
    LOG(INFO) << "MUSA IntraNodeComm: Topology::FULLY_CONNECTED";
    return Topology::FULLY_CONNECTED;
  }
  LOG(INFO) << "MUSA IntraNodeComm: Topology::UNKNOWN";
  return Topology::UNKNOWN;
}

IntraNodeComm::IntraNodeComm(
    c10::intrusive_ptr<c10d::Store> store,
    size_t rank,
    size_t worldSize,
    std::string processGroupName,
    std::optional<size_t> bufferSize)
    : store_(std::move(store)),
      rank_(rank),
      worldSize_(worldSize),
      initialBufferSize_(bufferSize.has_value() ? *bufferSize : 0),
      processGroupName_(
          std::make_shared<const std::string>(std::move(processGroupName))) {}

IntraNodeComm::~IntraNodeComm() {
  for (auto& workspace : symmetricMemoryWorkspaces_) {
    releaseSymmetricMemory(workspace);
  }
}

void IntraNodeComm::setProcessGroupName(const std::string& processGroupName) {
  auto current =
      std::atomic_load_explicit(&processGroupName_, std::memory_order_acquire);
  if (current != nullptr && *current == processGroupName) {
    return;
  }
  std::atomic_store_explicit(
      &processGroupName_,
      std::make_shared<const std::string>(processGroupName),
      std::memory_order_release);
}

std::shared_ptr<const std::string> IntraNodeComm::getProcessGroupName() const {
  return std::atomic_load_explicit(
      &processGroupName_, std::memory_order_acquire);
}

bool IntraNodeComm::isEnabled() {
  return getCvarBool(ENABLE_INTRA_NODE_COMM, false);
}

c10::DeviceIndex IntraNodeComm::deviceIndex() const {
  TORCH_CHECK(
      isInitialized_,
      "MUSA IntraNodeComm deviceIndex called before rendezvous");
  return deviceIdx_;
}

bool isIntraNodeMemoryOptimModeEnabled() {
  return getCvarBool(INTRA_NODE_MEMORY_OPTIM_MODE, false);
}

static c10::DeviceIndex maybeSetDeviceFromLocalRank() {
  const char* localRankEnv = std::getenv("LOCAL_RANK");
  if (localRankEnv == nullptr || localRankEnv[0] == '\0') {
    return c10::musa::current_device();
  }

  char* end = nullptr;
  errno = 0;
  const long localRank = std::strtol(localRankEnv, &end, 10);
  TORCH_CHECK(
      errno != ERANGE && end != localRankEnv && *end == '\0',
      "Invalid LOCAL_RANK value: ",
      localRankEnv);
  TORCH_CHECK(
      localRank >= 0, "LOCAL_RANK must be non-negative, got ", localRankEnv);

  const auto deviceCount = c10::musa::device_count();
  TORCH_CHECK(
      localRank < deviceCount,
      "LOCAL_RANK=",
      localRank,
      " is outside available MUSA device range [0, ",
      deviceCount,
      ")");
  const auto deviceIdx = static_cast<c10::DeviceIndex>(localRank);
  c10::musa::set_device(deviceIdx);
  return deviceIdx;
}

template <typename T>
static std::vector<T> storeAllGather(
    const c10::intrusive_ptr<c10d::Store>& store,
    const std::string& prefix,
    size_t rank,
    size_t worldSize,
    T val) {
  static_assert(std::is_trivially_copyable_v<T>);

  std::vector<std::string> peerKeys;
  peerKeys.reserve(worldSize);
  for (size_t r = 0; r < worldSize; ++r) {
    std::ostringstream oss;
    oss << prefix << '-' << r;
    peerKeys.push_back(oss.str());
  }

  {
    std::vector<uint8_t> payload(
        reinterpret_cast<uint8_t*>(&val),
        reinterpret_cast<uint8_t*>(&val) + sizeof(T));
    store->set(peerKeys[rank], payload);
  }

  std::vector<T> peerVals;
  peerVals.reserve(worldSize);
  for (size_t r = 0; r < worldSize; ++r) {
    if (r == rank) {
      peerVals.push_back(val);
      continue;
    }
    store->wait({peerKeys[r]});
    auto payload = store->get(peerKeys[r]);
    TORCH_CHECK(payload.size() == sizeof(T));
    T peerVal{};
    std::memcpy(&peerVal, payload.data(), sizeof(T));
    peerVals.push_back(peerVal);
  }
  return peerVals;
}

bool IntraNodeComm::rendezvous() {
  if (isInitialized_) {
    return true;
  }
  if (!isIntraNodeCommSupported() || worldSize_ < 2 ||
      worldSize_ > kMaxDevices) {
    return false;
  }

  deviceIdx_ = maybeSetDeviceFromLocalRank();

  struct DevInfo {
    char hostname[HOST_NAME_MAX + 1];
    int deviceIdx;
  };

  DevInfo devInfo{};
  gethostname(devInfo.hostname, sizeof(devInfo.hostname));
  devInfo.deviceIdx = deviceIdx_;

  auto peerDevInfos =
      storeAllGather(store_, "handshake-0", rank_, worldSize_, devInfo);

  std::vector<int> rankToDeviceIdx;
  rankToDeviceIdx.reserve(worldSize_);
  for (const auto& info : peerDevInfos) {
    if (strcmp(info.hostname, peerDevInfos.front().hostname) != 0) {
      LOG(WARNING)
          << "Aborting MUSA IntraNodeComm::rendezvous because participants "
             "are not on the same host ("
          << info.hostname << ", " << devInfo.hostname << ')';
      return false;
    }
    rankToDeviceIdx.emplace_back(info.deviceIdx);
  }

  {
    std::unordered_set<int> uniqueDeviceIdxs(
        rankToDeviceIdx.begin(), rankToDeviceIdx.end());
    if (uniqueDeviceIdxs.size() != worldSize_) {
      LOG(WARNING)
          << "Skipping MUSA IntraNodeComm::rendezvous() because participants "
             "have overlapping devices. Call torch.musa.set_device() before "
             "init_process_group() or launch with a correct LOCAL_RANK "
             "environment.";
      return false;
    }
  }

  auto mtlinkMesh = getMtLinkMesh(rankToDeviceIdx);
  topology_ = detectTopology(mtlinkMesh, worldSize_);
  if (topology_ != Topology::FULLY_CONNECTED) {
    return false;
  }

  groupName_ = "MUSAIntraNodeComm" + std::to_string(intraNodeCommIdx++);
  set_group_info(
      groupName_,
      static_cast<int>(rank_),
      static_cast<int>(worldSize_),
      store_);
  isInitialized_ = true;
  return true;
}

IntraNodeComm::SymmetricMemoryWorkspace& IntraNodeComm::ensureSymmetricMemory(
    CollectiveType collectiveType,
    size_t size) {
  TORCH_CHECK(isInitialized_, "MUSA IntraNodeComm called before rendezvous");
  TORCH_CHECK(size > 0, "MUSA IntraNodeComm requires non-empty tensor");

  // TODO(tiening.ma): The workspace may be reused cross diff collective ops to
  // save device memory.
  auto& workspace =
      symmetricMemoryWorkspaces_[static_cast<size_t>(collectiveType)];

  if (workspace.symmetricMemoryPtr != nullptr) {
    LOG(INFO) << "MUSA IntraNodeComm: existing symmetric memory workspace "
              << "size=" << static_cast<double>(workspace.size) / (1024 * 1024)
              << " MB, required size="
              << static_cast<double>(size) / (1024 * 1024) << " MB";
  }

  if (workspace.symmetricMemoryPtr != nullptr && workspace.size >= size) {
    LOG(INFO)
        << "MUSA IntraNodeComm: reusing previous symmetric memory workspace.";
    return workspace;
  }

  if (workspace.symmetricMemoryPtr != nullptr) {
    c10::musa::MUSAGuard guard(deviceIdx_);
    C10_MUSA_CHECK(musaDeviceSynchronize());
    LOG(INFO) << "MUSA IntraNodeComm: releasing symmetric memory workspace "
              << "before reallocation.";
    releaseSymmetricMemory(workspace);
  }

  LOG(INFO) << "MUSA IntraNodeComm: allocating symmetric memory workspace.";
  auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  const size_t allocSize = std::max(size, initialBufferSize_);
  void* ptr =
      allocator->alloc(allocSize, deviceIdx_, std::make_optional(groupName_));
  c10::intrusive_ptr<SymmetricMemory> symmetricMemory;
  try {
    symmetricMemory = allocator->rendezvous(ptr, std::nullopt);
    LOG(INFO)
        << "MUSA IntraNodeComm: symmetric memory workspace rendezvous completed.";
  } catch (...) {
    LOG(INFO)
        << "MUSA IntraNodeComm: symmetric memory workspace rendezvous failed.";
    allocator->free(ptr);
    throw;
  }

  workspace.symmetricMemoryPtr = ptr;
  workspace.symmetricMemory = std::move(symmetricMemory);
  workspace.size = allocSize;
  return workspace;
}

void IntraNodeComm::releaseSymmetricMemory(
    SymmetricMemoryWorkspace& workspace) {
  if (workspace.symmetricMemoryPtr == nullptr) {
    return;
  }
  workspace.symmetricMemory = nullptr;
  auto allocator = get_allocator(c10::DeviceType::PrivateUse1);
  allocator->free(workspace.symmetricMemoryPtr);
  workspace.symmetricMemoryPtr = nullptr;
  workspace.size = 0;
}

} // namespace c10d::musa_intra_node_comm
