#ifndef TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_
#define TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_

#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>

#include <ATen/ATen.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <mccl.h>
#include <torch/csrc/distributed/c10d/FlightRecorder.hpp>

#include "torch_musa/csrc/core/MUSAEvent.h"

namespace c10d {

extern template struct FlightRecorder<at::musa::MUSAEvent>;

} // namespace c10d

#if defined(MCCL_CONFIG_INITIALIZER)
#define MCCL_HAS_CONFIG
#endif

#if defined(MCCL_VERSION_CODE) && defined(MCCL_VERSION) && \
    MCCL_VERSION_CODE >= MCCL_VERSION(2, 14, 0)
#define MCCL_HAS_COMM_NONBLOCKING
#endif

#if defined(MCCL_SPLIT_NOCOLOR)
#define MCCL_HAS_COMM_SPLIT
#endif

#if !defined(MCCL_BF16_SUPPORTED)
#if defined(MARCH_TYPE) && (MARCH_TYPE >= 220)
#define MCCL_BF16_SUPPORTED 1
#else
#define MCCL_BF16_SUPPORTED 0
#endif
#endif

#if !defined(MCCL_FP8_SUPPORTED)
#if defined(MARCH_TYPE) && (MARCH_TYPE >= 310)
#define MCCL_FP8_SUPPORTED 1
#else
#define MCCL_FP8_SUPPORTED 0
#endif
#endif

constexpr int64_t kMcclCommInitBusyWaitMillis = 2;

// since MCCL 2.28
#if defined(MCCL_VERSION_CODE) && defined(MCCL_VERSION) && \
    MCCL_VERSION_CODE >= MCCL_VERSION(2, 28, 9)
#define MCCL_HAS_CONFIG
#define MCCL_HAS_CTA_POLICY
#define MCCL_ALLTOALL_SUPPORTED
#define MCCL_ALLTOALLV_SUPPORTED
#define MCCL_HAS_MEM_ALLOC
#define MCCL_HAS_COMM_REGISTER
#define MCCL_HAS_COMM_WINDOW_REGISTER
#endif

// Macro to throw on a non-successful MCCL return value.
#define C10D_MCCL_CHECK(cmd, failureReason)                                   \
  do {                                                                        \
    mcclResult_t result = cmd;                                                \
    if (result != mcclSuccess) {                                              \
      std::string err = "MCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + mcclGetErrorWithVersion(result) + \
          "\n" + getMcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK(false, err);                                                \
    }                                                                         \
  } while (0)

// Macro to throw on a non-successful MCCL return value for NONBLOCKING calls.
#define C10D_MCCL_CHECK_NONBLOCKING(cmd, failureReason)                       \
  do {                                                                        \
    mcclResult_t result = cmd;                                                \
    if (result != mcclSuccess && result != mcclInProgress) {                  \
      std::string err = "MCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + mcclGetErrorWithVersion(result) + \
          "\n" + getMcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK(false, err);                                                \
    }                                                                         \
  } while (0)

// Error out if (current time - startTime) is greater than timeout (sec).
#define C10D_MCCL_CHECK_TIMEOUT_ELAPSED(startTime, timeout)                 \
  do {                                                                      \
    auto currentTime = std::chrono::steady_clock::now();                    \
    auto timeElapsed = std::chrono::duration_cast<std::chrono::seconds>(    \
                           currentTime - startTime)                         \
                           .count();                                        \
    if (timeElapsed > timeout) {                                            \
      std::string err = "MCCL timeout in: " + std::string(__FILE__) + ":" + \
          std::to_string(__LINE__);                                         \
      TORCH_CHECK(false, err);                                              \
    }                                                                       \
  } while (0)

// Macro to throw on a non-successful MCCL return value, non-blocking.
#define C10D_MCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, yield_fn)      \
  do {                                                                        \
    mcclResult_t result = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                   \
    auto timeout = c10d::mccl_nonblocking_timeout();                          \
    while (result == mcclInProgress) {                                        \
      C10D_MCCL_CHECK_TIMEOUT_ELAPSED(startTimepoint, timeout);               \
      yield_fn;                                                               \
      mcclCommGetAsyncError(comm, &result);                                   \
    }                                                                         \
    if (result != mcclSuccess) {                                              \
      std::string err = "MCCL error in: " + std::string(__FILE__) + ":" +     \
          std::to_string(__LINE__) + ", " + mcclGetErrorWithVersion(result) + \
          "\n" + getMcclErrorDetailStr(result, failureReason);                \
      TORCH_CHECK(false, err);                                                \
    }                                                                         \
  } while (0)

// Sleep for kMcclCommInitBusyWaitMillis milliseconds.
#define C10D_MCCL_SCHED_SLEEP() \
  std::this_thread::sleep_for(  \
      std::chrono::milliseconds(kMcclCommInitBusyWaitMillis))

// Macro to throw exception on a non-successful MCCL return value or timeout.
// This macro uses sched_yield() to yield the CPU and is suitable for
// collectives that should complete quickly.
#define C10D_MCCL_CHECK_TIMEOUT(cmd, comm, failureReason) \
  C10D_MCCL_CHECK_TIMEOUT_BASE(cmd, comm, failureReason, sched_yield())

// Macro to throw exception on a non-successful MCCL return value or timeout.
// This macro uses sleep to yield the CPU and is suitable for longer calls.
#define C10D_MCCL_CHECK_TIMEOUT_SLEEP(cmd, comm, failureReason) \
  C10D_MCCL_CHECK_TIMEOUT_BASE(                                 \
      cmd, comm, failureReason, C10D_MCCL_SCHED_SLEEP())

#define C10D_MCCL_CHECK_TIMEOUT_GROUPEND(cmd, comm, failureReason)           \
  do {                                                                       \
    mcclResult_t state = cmd;                                                \
    auto startTimepoint = std::chrono::steady_clock::now();                  \
    auto timeout = c10d::mccl_nonblocking_timeout();                         \
    if (state == mcclInProgress) {                                           \
      do {                                                                   \
        C10D_MCCL_CHECK_TIMEOUT_ELAPSED(startTimepoint, timeout);            \
        sched_yield();                                                       \
        mcclCommGetAsyncError(comm->getMcclComm(), &state);                  \
      } while (state == mcclInProgress);                                     \
    }                                                                        \
    if (state != mcclSuccess) {                                              \
      std::string err = "MCCL error in: " + std::string(__FILE__) + ":" +    \
          std::to_string(__LINE__) + ", " + mcclGetErrorWithVersion(state) + \
          "\n" + getMcclErrorDetailStr(state, failureReason);                \
      TORCH_CHECK(false, err);                                               \
    }                                                                        \
  } while (0)

// Macro to print and abort on a non-successful MCCL return value.
#define C10D_MCCL_ASSERT(cmd)                            \
  do {                                                   \
    mcclResult_t result = cmd;                           \
    if (result != mcclSuccess) {                         \
      std::string err = mcclGetErrorWithVersion(result); \
      fprintf(                                           \
          stderr,                                        \
          "[Error] MCCL error in: %s:%d, %s\n",          \
          __FILE__,                                      \
          __LINE__,                                      \
          err.c_str());                                  \
      abort();                                           \
    }                                                    \
  } while (0)

namespace c10d {

// MCCL type mapping
static std::map<at::ScalarType, mcclDataType_t> mcclDataType = {
    {at::kFloat, mcclFloat},
    {at::kInt, mcclInt32},
    {at::kChar, mcclInt8},
    {at::kByte, mcclUint8},
    {at::kLong, mcclInt64},
    {at::kHalf, mcclFloat16},
    {at::kDouble, mcclFloat64},
    {at::kBool, mcclUint8},
#if MCCL_BF16_SUPPORTED
    {at::kBFloat16, mcclBfloat16},
#endif
#if MCCL_FP8_SUPPORTED
    {at::kFloat8_e5m2, mcclFp8E5M2},
    {at::kFloat8_e4m3fn, mcclFp8E4M3},
#endif
};

std::string getMcclVersion();
std::string mcclGetErrorWithVersion(mcclResult_t error);
int mccl_nonblocking_timeout();

std::string getMcclErrorDetailStr(
    mcclResult_t error,
    c10::optional<std::string> processGroupFailureReason = c10::nullopt);

size_t hashTensors(const std::vector<at::Tensor>& tensors);

mcclDataType_t getMcclDataType(at::ScalarType type);

// RAII helper class to manage MCCL group API lifetime.
// The destructor is allowed to throw since this helper only manages group
// lifetime around MCCL API calls.
struct AutoMcclGroup {
  AutoMcclGroup() : comm_(nullptr), comm_nonblocking_(false) {
    C10D_MCCL_ASSERT(mcclGroupStart());
  }

  AutoMcclGroup(mcclComm_t comm, bool comm_nonblocking)
      : comm_(comm), comm_nonblocking_(comm_nonblocking) {
    C10D_MCCL_ASSERT(mcclGroupStart());
  }

  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~AutoMcclGroup() noexcept(false) {
#ifdef MCCL_HAS_COMM_NONBLOCKING
    if (comm_nonblocking_) {
      C10D_MCCL_CHECK_TIMEOUT(mcclGroupEnd(), comm_, std::nullopt);
    } else
#endif // MCCL_HAS_COMM_NONBLOCKING
    {
      C10D_MCCL_ASSERT(mcclGroupEnd());
    }
  }

  mcclComm_t comm_;
  bool comm_nonblocking_;
};

// RAII wrapper for MCCL communicator
class MCCLComm {
  using MutexType = std::recursive_mutex;
  using LockType = std::unique_lock<MutexType>;

 public:
  explicit MCCLComm(mcclComm_t mcclComm);

  MCCLComm() = default;

  ~MCCLComm() noexcept;

#ifdef MCCL_HAS_CONFIG
  static std::shared_ptr<MCCLComm> create(
      int numRanks,
      int rank,
      mcclUniqueId commId,
      at::DeviceIndex deviceIndex,
      mcclConfig_t& config);
#endif

  static std::shared_ptr<MCCLComm> create(
      int numRanks,
      int rank,
      mcclUniqueId commId,
      at::DeviceIndex deviceIndex);

  std::unordered_map<std::string, std::string> mcclCommDump();

  void setUniqueHash(mcclUniqueId mcclId);
  void setUniqueHash(std::string hash);
  std::string getUniqueHash();

  at::DeviceIndex getDeviceIndex();

  // Must not be copyable
  MCCLComm(const MCCLComm&) = delete;
  MCCLComm& operator=(const MCCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  MCCLComm& operator=(MCCLComm&& other) = delete;

  // Move constructable
  // NOLINTNEXTLINE(*-noexcept-move-*)
  MCCLComm(MCCLComm&& other);

  mcclComm_t getMcclComm();

  void waitReady(bool longInterval);

  c10::optional<std::string> getMcclCommFailureReason() const;

  void abort(std::optional<std::string> commFailureReason = std::nullopt);

  void finalize();

  // Destroy a communicator. This is a blocking function.
  void destroy();

  bool isInitialized() const;

  bool isAborted() const;

  uint64_t getCommSplitCounter() const;

  mcclResult_t checkForMcclError();

  mcclResult_t registerSegment(
      void* ptr,
      size_t size,
      bool errorOnRereg = true,
      bool window = false);

  mcclResult_t deregisterSegment(void* ptr, bool window = false);

  std::string repr() const;

  friend class ProcessGroupMCCL;

 protected:
  // Unique hash for this communicator.
  std::string uniqueHash_;
  bool aborted_{false};
  uint64_t mcclCommSplitCounter_{0};
  mcclResult_t mcclAsyncErr_{mcclSuccess};
  mutable MutexType mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
  // Optional reason for communicator failure, provided by ProcessGroupMCCL for
  // better error messaging.
  std::optional<std::string> commFailureReason_{};
  bool initialized_{false};
  // Whether this communicator is using nonblocking mode. Recorded during comm
  // creation or split. For safety, we give a default value of true (more
  // protection).
  bool nonBlocking_{true};
  // Device index for which the MCCL comm is created
  at::DeviceIndex deviceIndex_{-1};

 private:
  mcclComm_t mcclComm_{nullptr};
#if defined(MCCL_HAS_COMM_REGISTER)
  // Stores handlers for tensors registered by MCCL
  std::unordered_map<void*, void*> registeredSegmentHandles_;
#endif
};

// Helper that automatically cleans up premul sums.
struct mcclRedOpRAII {
  mcclRedOpRAII() = default;
  mcclRedOpRAII(mcclRedOp_t op) : op_(op) {}
  mcclRedOpRAII(mcclRedOp_t op, mcclComm_t comm)
      : op_(op), comm_(comm), premul_sum_(true) {}
  mcclRedOpRAII(const mcclRedOpRAII&) = delete;
  mcclRedOpRAII& operator=(const mcclRedOpRAII&) = delete;
  mcclRedOpRAII(mcclRedOpRAII&& tmp) : mcclRedOpRAII() {
    std::swap(tmp.op_, this->op_);
    std::swap(tmp.comm_, this->comm_);
    std::swap(tmp.premul_sum_, this->premul_sum_);
  }
  ~mcclRedOpRAII() {
    if (premul_sum_) {
      mcclRedOpDestroy(op_, comm_);
    }
  }
  operator mcclRedOp_t() const {
    return op_;
  }
  mcclRedOp_t op_;
  mcclComm_t comm_;
  bool premul_sum_ = false;
};

} // namespace c10d

#endif // TORCH_MUSA_CSRC_DISTRIBUTED_MCCLUTILS_H_
