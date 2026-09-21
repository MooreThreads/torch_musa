#include "torch_musa/csrc/distributed/symm_mem/intra_node_comm.hpp"

#include <algorithm>
#include <atomic>
#include <cstring>

#include <c10/util/Logging.h>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemory-inl.h"

namespace c10d::musa_intra_node_comm {

static constexpr size_t kOneShotThreshBytes = 256 * 1024;
static constexpr size_t kTwoShotThreshBytes = 10 * 1024 * 1024;

static void checkInput(const at::Tensor& input, int deviceIdx) {
  TORCH_CHECK(
      input.dtype() == at::kBFloat16 || input.dtype() == at::kFloat,
      "MUSA IntraNodeComm allReduce only supports float and bf16 for now");
  TORCH_CHECK(
      input.is_contiguous(),
      "MUSA IntraNodeComm allReduce expects contiguous input");
  TORCH_CHECK(input.is_musa());
  TORCH_CHECK(
      input.get_device() == deviceIdx,
      "MUSA IntraNodeComm: expect input to be on device ",
      deviceIdx,
      ", got device ",
      input.get_device());
}

static void checkLowContentionInput(const at::Tensor& input, int deviceIdx) {
  TORCH_CHECK(
      input.dtype() == at::kBFloat16 || input.dtype() == at::kFloat ||
          input.dtype() == at::kHalf,
      "MUSA IntraNodeComm low_contention_allreduce only supports "
      "float, bf16, and fp16");
  TORCH_CHECK(
      input.is_contiguous(),
      "MUSA IntraNodeComm low_contention_allreduce expects contiguous input");
  TORCH_CHECK(input.is_musa());
  TORCH_CHECK(
      input.get_device() == deviceIdx,
      "MUSA IntraNodeComm: expect input to be on device ",
      deviceIdx,
      ", got device ",
      input.get_device());
}

static bool isLowContentionReduceOp(const char* reduceOp) {
  return reduceOp != nullptr &&
      (std::strcmp(reduceOp, "sum") == 0 || std::strcmp(reduceOp, "avg") == 0);
}

static size_t tensorBytes(const at::Tensor& tensor) {
  return static_cast<size_t>(tensor.numel() * tensor.element_size());
}

static bool isSymmetricMemoryTensor(
    const at::Tensor& tensor,
    const std::string& groupName) {
  if (groupName.empty()) {
    return false;
  }
  return c10d::symmetric_memory::rendezvous(tensor, groupName) != nullptr;
}

static void logInputMemcpyDecision(
    const char* op,
    bool inputIsSymmMem,
    bool needsMemcpy) {
  LOG(INFO)
      << "MUSA IntraNodeComm " << op << ": input tensor "
      << (inputIsSymmMem ? "is" : "is not")
      << " already a symmetric memory tensor; "
      << (needsMemcpy
              ? "memcpy to internal symmetric memory workspace is needed."
              : "memcpy to internal symmetric memory workspace is not needed.");
}

static void logInputOutputMemcpyDecision(
    bool inputIsSymmMem,
    bool outputIsSymmMem,
    bool stageInput,
    bool stageOutput) {
  LOG(INFO)
      << "MUSA IntraNodeComm reduceScatter: input tensor "
      << (inputIsSymmMem ? "is" : "is not")
      << " already a symmetric memory tensor, output tensor "
      << (outputIsSymmMem ? "is" : "is not")
      << " already a symmetric memory tensor; "
      << (stageInput
              ? "staging input in the internal symmetric memory "
                "workspace."
              : (stageOutput
                     ? "staging output in the internal symmetric memory "
                       "workspace."
                     : "no internal symmetric memory staging is needed."));
}

bool isIntraNodeCommSupported() {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ < 310)
  return false;
#else
  return true;
#endif
}

at::Tensor IntraNodeComm::oneShotAllReduce(
    const at::Tensor& input,
    c10::musa::MUSAStream& stream) {
  checkInput(input, deviceIdx_);

  const auto processGroupName = getProcessGroupName();
  const bool inputIsSymmMem = isSymmetricMemoryTensor(input, *processGroupName);
  logInputMemcpyDecision("oneShotAllReduce", inputIsSymmMem, !inputIsSymmMem);
  if (inputIsSymmMem) {
    // one-shot out requires non-aliased input/output; use the in-place op.
    auto op = c10::Dispatcher::singleton()
                  .findSchemaOrThrow("symm_mem::two_shot_all_reduce_", "")
                  .typed<at::Tensor(at::Tensor, std::string, std::string)>();
    op.call(input, "sum", *processGroupName);
    return input;
  }

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::one_shot_all_reduce_out", "")
                .typed<at::Tensor(
                    const at::Tensor&, std::string, std::string, at::Tensor)>();

  auto& workspace =
      ensureSymmetricMemory(CollectiveType::ALL_REDUCE, tensorBytes(input));
  auto symmMemTensor = workspace.symmetricMemory->get_buffer(
      static_cast<int>(rank_), input.sizes(), input.scalar_type(), 0);

  C10_MUSA_CHECK(musaMemcpyAsync(
      symmMemTensor.data_ptr(),
      input.data_ptr(),
      tensorBytes(input),
      musaMemcpyDeviceToDevice,
      stream));
  op.call(symmMemTensor, "sum", groupName_, input);
  return input;
}

at::Tensor IntraNodeComm::lowContentionAllReduce(
    const at::Tensor& input,
    const char* reduceOp,
    c10::musa::MUSAStream& stream) {
  checkLowContentionInput(input, deviceIdx_);
  TORCH_CHECK(
      isLowContentionReduceOp(reduceOp),
      "MUSA IntraNodeComm low_contention_allreduce only supports sum and avg");

  auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("symm_mem::low_contention_allreduce", "")
          .typed<void(at::Tensor&, const std::string&, const std::string&)>();

  auto inputArg = input;
  const std::string* groupName = &groupName_;
  const auto processGroupName = getProcessGroupName();
  const bool inputIsSymmMem = isSymmetricMemoryTensor(input, *processGroupName);
  logInputMemcpyDecision(
      "lowContentionAllReduce", inputIsSymmMem, !inputIsSymmMem);
  if (inputIsSymmMem) {
    // The low-contention op can operate directly on a user-provided
    // symmetric-memory tensor, including a non-zero storage offset.
    groupName = processGroupName.get();
  } else {
    auto& workspace =
        ensureSymmetricMemory(CollectiveType::ALL_REDUCE, tensorBytes(input));
    inputArg = workspace.symmetricMemory->get_buffer(
        static_cast<int>(rank_), input.sizes(), input.scalar_type(), 0);

    C10_MUSA_CHECK(musaMemcpyAsync(
        inputArg.data_ptr(),
        input.data_ptr(),
        tensorBytes(input),
        musaMemcpyDeviceToDevice,
        stream));
  }
  op.call(inputArg, reduceOp, *groupName);
  if (!inputIsSymmMem) {
    C10_MUSA_CHECK(musaMemcpyAsync(
        input.data_ptr(),
        inputArg.data_ptr(),
        tensorBytes(input),
        musaMemcpyDeviceToDevice,
        stream));
  }
  return input;
}

at::Tensor IntraNodeComm::twoShotAllReduce(
    const at::Tensor& input,
    c10::musa::MUSAStream& stream) {
  checkInput(input, deviceIdx_);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::two_shot_all_reduce_", "")
                .typed<at::Tensor(at::Tensor, std::string, std::string)>();

  const auto processGroupName = getProcessGroupName();
  const bool inputIsSymmMem = isSymmetricMemoryTensor(input, *processGroupName);
  logInputMemcpyDecision("twoShotAllReduce", inputIsSymmMem, !inputIsSymmMem);
  if (inputIsSymmMem) {
    op.call(input, "sum", *processGroupName);
    return input;
  }

  auto& workspace =
      ensureSymmetricMemory(CollectiveType::ALL_REDUCE, tensorBytes(input));
  auto symmMemTensor = workspace.symmetricMemory->get_buffer(
      static_cast<int>(rank_), input.sizes(), input.scalar_type(), 0);

  C10_MUSA_CHECK(musaMemcpyAsync(
      symmMemTensor.data_ptr(),
      input.data_ptr(),
      tensorBytes(input),
      musaMemcpyDeviceToDevice,
      stream));
  op.call(symmMemTensor, "sum", groupName_);
  input.copy_(symmMemTensor);
  return input;
}

AllReduceAlgo IntraNodeComm::selectAllReduceAlgo(
    const at::Tensor& input,
    const char* reduceOp) {
  if (!isInitialized_ || topology_ != Topology::FULLY_CONNECTED ||
      !input.is_musa() || input.get_device() != deviceIdx_ ||
      input.numel() == 0 || !input.is_contiguous()) {
    return AllReduceAlgo::NONE;
  }

  if (isLowContentionReduceOp(reduceOp) &&
      (input.dtype() == at::kBFloat16 || input.dtype() == at::kFloat ||
       input.dtype() == at::kHalf) &&
      input.numel() % worldSize_ == 0) {
    return AllReduceAlgo::LOW_CONTENTION;
  }

  // The legacy one/two-shot kernels only support SUM and float/bfloat16.
  if (reduceOp == nullptr || std::strcmp(reduceOp, "sum") != 0 ||
      (input.dtype() != at::kBFloat16 && input.dtype() != at::kFloat)) {
    return AllReduceAlgo::NONE;
  }

  const auto inputSize = tensorBytes(input);
  const size_t ptrAlignment = c10d::symmetric_memory::get_alignment(
      static_cast<size_t>(input.storage_offset() * input.element_size()));
  const size_t sizeAlignment = c10d::symmetric_memory::get_alignment(inputSize);
  const size_t alignment = std::min(ptrAlignment, sizeAlignment);

  // TODO: The conditions for one-shot and two-shot need to be rechecked and
  // confirmed.
  if (topology_ == Topology::FULLY_CONNECTED) {
    if (alignment >= 4 && inputSize <= kOneShotThreshBytes) {
      return AllReduceAlgo::ONE_SHOT;
    }
    if (alignment >= 4 && inputSize <= kTwoShotThreshBytes) {
      return AllReduceAlgo::TWO_SHOT;
    }
  }
  return AllReduceAlgo::NONE;
}

static std::atomic<int64_t> usageCounter{0};

bool IntraNodeComm::canUseAllGather(
    const at::Tensor& input,
    const at::Tensor& output) const {
  if (!isInitialized_ || topology_ != Topology::FULLY_CONNECTED ||
      input.numel() == 0) {
    return false;
  }
  if (!input.is_musa() || !output.is_musa()) {
    return false;
  }
  if (input.get_device() != deviceIdx_ || output.get_device() != deviceIdx_) {
    return false;
  }
  if (!input.is_contiguous() || !output.is_contiguous()) {
    return false;
  }
  if (input.scalar_type() != output.scalar_type()) {
    return false;
  }
  if (output.numel() != input.numel() * static_cast<int64_t>(worldSize_)) {
    return false;
  }
  return true;
}

at::Tensor IntraNodeComm::allGather(
    const at::Tensor& output,
    const at::Tensor& input) {
  TORCH_CHECK(
      isInitialized_, "MUSA IntraNodeComm allGather called before rendezvous");
  usageCounter.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(
      collectiveMutexes_[static_cast<size_t>(CollectiveType::ALL_GATHER)]);

  auto op = c10::Dispatcher::singleton()
                .findSchemaOrThrow("symm_mem::low_contention_all_gather", "")
                .typed<void(at::Tensor&, at::Tensor&, const std::string&)>();

  auto inputArg = input;
  const std::string* groupName = &groupName_;
  const auto processGroupName = getProcessGroupName();
  const bool inputIsSymmMem = isSymmetricMemoryTensor(input, *processGroupName);
  logInputMemcpyDecision("allGather", inputIsSymmMem, !inputIsSymmMem);
  if (!inputIsSymmMem) {
    const auto inputBytes = tensorBytes(input);
    auto& workspace =
        ensureSymmetricMemory(CollectiveType::ALL_GATHER, inputBytes);
    inputArg = workspace.symmetricMemory->get_buffer(
        static_cast<int>(rank_), input.sizes(), input.scalar_type(), 0);
    auto stream = at::musa::getCurrentMUSAStream();
    C10_MUSA_CHECK(musaMemcpyAsync(
        inputArg.data_ptr(),
        input.data_ptr(),
        inputBytes,
        musaMemcpyDeviceToDevice,
        stream));
  } else {
    groupName = processGroupName.get();
  }
  auto outputArg = output;
  op.call(outputArg, inputArg, *groupName);
  return output;
}

bool IntraNodeComm::canUseReduceScatter(
    const at::Tensor& input,
    const at::Tensor& output) const {
  if (!isInitialized_ || topology_ != Topology::FULLY_CONNECTED ||
      input.numel() == 0) {
    return false;
  }
  if (!input.is_contiguous() || !output.is_contiguous()) {
    return false;
  }
  if (input.scalar_type() != at::kBFloat16 &&
      input.scalar_type() != at::kFloat && input.scalar_type() != at::kHalf) {
    return false;
  }
  return true;
}

at::Tensor IntraNodeComm::reduceScatter(
    const at::Tensor& output,
    const at::Tensor& input,
    const std::string& reduceOp) {
  TORCH_CHECK(
      isInitialized_,
      "MUSA IntraNodeComm reduceScatter called before rendezvous");
  usageCounter.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(
      collectiveMutexes_[static_cast<size_t>(CollectiveType::REDUCE_SCATTER)]);

  auto op =
      c10::Dispatcher::singleton()
          .findSchemaOrThrow("symm_mem::low_contention_reduce_scatter", "")
          .typed<void(
              at::Tensor&,
              at::Tensor&,
              const std::string&,
              const std::string&)>();

  auto inputArg = input;
  const std::string* groupName = &groupName_;
  const auto processGroupName = getProcessGroupName();
  const bool inputIsSymmMem = isSymmetricMemoryTensor(input, *processGroupName);
  const bool outputIsSymmMem = inputIsSymmMem
      ? false
      : isSymmetricMemoryTensor(output, *processGroupName);
  const bool memoryOptimModeEnabled = isIntraNodeMemoryOptimModeEnabled();
  LOG(INFO) << "MUSA IntraNodeComm reduceScatter: "
            << "INTRA_NODE_MEMORY_OPTIM_MODE="
            << (memoryOptimModeEnabled ? "1" : "0");
  const bool stageOutput =
      !inputIsSymmMem && !outputIsSymmMem && memoryOptimModeEnabled;
  const bool stageInput = !inputIsSymmMem && !outputIsSymmMem && !stageOutput;
  logInputOutputMemcpyDecision(
      inputIsSymmMem, outputIsSymmMem, stageInput, stageOutput);

  auto outputArg = output;
  if (stageInput) {
    const auto inputBytes = tensorBytes(input);
    auto& workspace =
        ensureSymmetricMemory(CollectiveType::REDUCE_SCATTER, inputBytes);
    inputArg = workspace.symmetricMemory->get_buffer(
        static_cast<int>(rank_), input.sizes(), input.scalar_type(), 0);
    auto stream = at::musa::getCurrentMUSAStream();
    C10_MUSA_CHECK(musaMemcpyAsync(
        inputArg.data_ptr(),
        input.data_ptr(),
        inputBytes,
        musaMemcpyDeviceToDevice,
        stream));
  } else if (stageOutput) {
    const auto outputBytes = tensorBytes(output);
    auto& workspace =
        ensureSymmetricMemory(CollectiveType::REDUCE_SCATTER, outputBytes);
    // A regular input and symmetric output make the low-contention op select
    // push mode (use_pull_mode=false).
    outputArg = workspace.symmetricMemory->get_buffer(
        static_cast<int>(rank_), output.sizes(), output.scalar_type(), 0);
  } else {
    groupName = processGroupName.get();
  }
  op.call(outputArg, inputArg, reduceOp, *groupName);
  if (stageOutput) {
    auto stream = at::musa::getCurrentMUSAStream();
    C10_MUSA_CHECK(musaMemcpyAsync(
        output.data_ptr(),
        outputArg.data_ptr(),
        tensorBytes(output),
        musaMemcpyDeviceToDevice,
        stream));
  }
  return output;
}

at::Tensor IntraNodeComm::allReduce(
    const at::Tensor& input,
    AllReduceAlgo algo,
    const char* reduceOp) {
  usageCounter.fetch_add(1, std::memory_order_relaxed);
  std::lock_guard<std::mutex> lock(
      collectiveMutexes_[static_cast<size_t>(CollectiveType::ALL_REDUCE)]);
  auto stream = at::musa::getCurrentMUSAStream();
  switch (algo) {
    case AllReduceAlgo::LOW_CONTENTION:
      LOG(INFO) << "MUSA IntraNodeComm allReduce: using LOW_CONTENTION branch";
      return lowContentionAllReduce(input, reduceOp, stream);
    case AllReduceAlgo::ONE_SHOT:
      LOG(INFO) << "MUSA IntraNodeComm allReduce: using ONE_SHOT branch";
      return oneShotAllReduce(input, stream);
    case AllReduceAlgo::TWO_SHOT:
      LOG(INFO) << "MUSA IntraNodeComm allReduce: using TWO_SHOT branch";
      return twoShotAllReduce(input, stream);
    default:
      LOG(INFO) << "MUSA IntraNodeComm allReduce: using INVALID_ALGO branch";
      C10_THROW_ERROR(ValueError, "MUSA IntraNodeComm: invalid algo");
  }
}

int64_t getIntraNodeCommUsageCounter() {
  return usageCounter.load(std::memory_order_relaxed);
}

} // namespace c10d::musa_intra_node_comm
