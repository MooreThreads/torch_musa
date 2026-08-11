#include <torch/csrc/distributed/c10d/GroupRegistry.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include <limits.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <c10/core/DeviceType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/Exception.h>
#include <c10/util/intrusive_ptr.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <atomic>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/driver_api.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemory.hpp"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryTypes.hpp"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryUtils.hpp"

namespace c10d {
namespace symmetric_memory {

/* Start of MUSASymmetricMemory implementation */

static StoreExchange storeExchange("MUSASymmetricMemory");

MUSAAllocationRef::MUSAAllocationRef(
    void* ptr,
    HandleType handle,
    size_t block_size,
    int device_idx,
    bool is_multicast)
    : ptr(ptr),
      handle(handle),
      block_size(block_size),
      device_idx(device_idx),
      is_multicast(is_multicast) {}

MUSAAllocationRef::~MUSAAllocationRef() {
  if (is_finalizing()) {
    return;
  }
  c10::musa::MUSAGuard guard(device_idx);
  C10_MUSA_CHECK(musaDeviceSynchronize());
  auto driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(
      driver_api->muMemUnmap_(reinterpret_cast<MUdeviceptr>(ptr), block_size));
  C10_MUSA_DRIVER_CHECK(driver_api->muMemAddressFree_(
      reinterpret_cast<MUdeviceptr>(ptr), block_size));
  C10_MUSA_DRIVER_CHECK(driver_api->muMemRelease_(handle));
};

MUSAPeerAllocInfo::MUSAPeerAllocInfo(
    std::vector<c10::intrusive_ptr<MUSAAllocationRef>> alloc_refs,
    std::vector<void*> buffers,
    std::vector<void*> signal_pads,
    HandleType mc_handle,
    void* mc_addr,
    size_t buffer_size,
    int local_device_idx,
    int rank,
    int world_size)
    : alloc_refs_(std::move(alloc_refs)),
      buffers_(std::move(buffers)),
      signal_pads_(std::move(signal_pads)),
      mc_handle_(mc_handle),
      mc_addr_(mc_addr),
      buffer_size_(buffer_size),
      local_device_idx_(local_device_idx),
      rank_(rank),
      world_size_(world_size) {
  const size_t arr_size = sizeof(void*) * world_size_;
  buffers_dev_ = reinterpret_cast<void**>(
      c10::musa::MUSACachingAllocator::raw_alloc(arr_size));
  signal_pads_dev_ = reinterpret_cast<void**>(
      c10::musa::MUSACachingAllocator::raw_alloc(arr_size));

  c10::musa::MUSAGuard guard(local_device_idx);
  AT_MUSA_CHECK(musaMemcpy(
      buffers_dev_, buffers_.data(), arr_size, musaMemcpyHostToDevice));
  AT_MUSA_CHECK(musaMemcpy(
      signal_pads_dev_, signal_pads_.data(), arr_size, musaMemcpyHostToDevice));
}

/* Start of MUSASymmetricMemory */

MUSASymmetricMemory::MUSASymmetricMemory(
    const c10::intrusive_ptr<MUSAPeerAllocInfo>& pai,
    size_t offset)
    : local_device_idx_(pai->local_device_idx_),
      rank_(pai->rank_),
      world_size_(pai->world_size_),
      pai_(pai),
      offset_(offset) {
  // offset is specific per symm_mem handle
  TORCH_INTERNAL_ASSERT(offset_ < pai_->buffer_size_, "offset out of range");
}

std::vector<void*> MUSASymmetricMemory::get_buffer_ptrs() {
  return pai_->buffers_;
}

std::vector<void*> MUSASymmetricMemory::get_signal_pad_ptrs() {
  return pai_->signal_pads_;
}

void** MUSASymmetricMemory::get_buffer_ptrs_dev() {
  return pai_->buffers_dev_;
}

void** MUSASymmetricMemory::get_signal_pad_ptrs_dev() {
  return pai_->signal_pads_dev_;
}

size_t MUSASymmetricMemory::get_buffer_size() {
  return pai_->buffer_size_;
}

size_t MUSASymmetricMemory::get_signal_pad_size() {
  return signal_pad_size;
}

bool MUSASymmetricMemory::has_multicast_support() {
  return pai_->mc_addr_ != nullptr;
}

void* MUSASymmetricMemory::get_multicast_ptr() {
  if (!has_multicast_support()) {
    return nullptr;
  }
  return static_cast<char*>(pai_->mc_addr_) + offset_;
}

size_t MUSASymmetricMemory::get_offset() {
  return offset_;
}

void check_channel(int channel, int world_size) {
  TORCH_CHECK(
      channel >= 0,
      "channel for barrier(), put_signal() and wait_signal() ",
      "must be greater than or equal to 0 (got ",
      channel,
      ")");
  const size_t num_channels =
      signal_pad_size / sizeof(uint32_t) / static_cast<size_t>(world_size);
  TORCH_CHECK(
      static_cast<size_t>(channel) < num_channels,
      "The maximum supported channel for barrier(), put_signal() and "
      "wait_signal() is ",
      num_channels - 1,
      " (got ",
      channel,
      ")");
}

// TODO(mingyuan.wang): move these kernel into a header file
namespace {

#define MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL(assertion, fmt, ...) \
  do {                                                        \
    printf(fmt, __VA_ARGS__);                                 \
    __assert_fail(assertion, __FILE__, __LINE__, __func__);   \
  } while (0)

__device__ __forceinline__ bool try_put_signal(
    uint32_t* addr,
    size_t timeout_ms,
    uint64_t clock_rate_khz) {
  const auto deadline = timeout_ms == 0 ? 0
                                        : static_cast<uint64_t>(clock64()) +
          static_cast<uint64_t>(timeout_ms) * clock_rate_khz;
  // TODO(mingyuan.wang): Revisit this issue later. Use __threadfence_system()
  // + atomicCAS() instead of PyTorch's cas wrapper for now, since unit tests
  // exposed precision issues and program hangs.
  __threadfence_system();
  while (atomicCAS(addr, 0U, 1U) != 0U) {
    if (timeout_ms != 0 && static_cast<uint64_t>(clock64()) > deadline) {
      return false;
    }
  }
  return true;
}

__device__ __forceinline__ bool try_wait_signal(
    uint32_t* addr,
    size_t timeout_ms,
    uint64_t clock_rate_khz) {
  const auto deadline = timeout_ms == 0 ? 0
                                        : static_cast<uint64_t>(clock64()) +
          static_cast<uint64_t>(timeout_ms) * clock_rate_khz;
  while (atomicCAS(addr, 1U, 0U) != 1U) {
    if (timeout_ms != 0 && static_cast<uint64_t>(clock64()) > deadline) {
      return false;
    }
  }
  __threadfence_system();
  return true;
}

__global__ void barrier_kernel(
    uint32_t** signal_pads,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    uint64_t clock_rate_khz) {
  if (threadIdx.x < world_size) {
    auto target_rank = static_cast<int>(threadIdx.x);
    if (target_rank == rank) {
      return;
    }
    auto put_success = try_put_signal(
        signal_pads[target_rank] + world_size * channel + rank,
        timeout_ms,
        clock_rate_khz);
    if (!put_success) {
      // Use assert instead of trap because real training occasionally failed
      // to exit after the program finished. It is unclear why trap may related
      // to this issue, but assert is sufficient for this fatal error path.
      MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL(
          "MUSASymmetricMemory::barrier failed to send signal",
          "[FATAL] MUSASymmetricMemory::barrier: rank %d failed to send "
          "signal to rank %d on channel %d after %lu milliseconds\n",
          rank,
          target_rank,
          channel,
          timeout_ms);
    }
    auto wait_success = try_wait_signal(
        signal_pads[rank] + world_size * channel + target_rank,
        timeout_ms,
        clock_rate_khz);
    if (!wait_success) {
      // Use assert instead of trap because real training occasionally failed
      // to exit after the program finished. It is unclear why trap may related
      // to this issue, but assert is sufficient for this fatal error path.
      MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL(
          "MUSASymmetricMemory::barrier failed to receive signal",
          "[FATAL] MUSASymmetricMemory::barrier: rank %d failed to receive "
          "signal from rank %d on channel %d after %lu milliseconds\n",
          rank,
          target_rank,
          channel,
          timeout_ms);
    }
  }
}

__global__ void put_signal_kernel(
    uint32_t** signal_pads,
    int dst_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    uint64_t clock_rate_khz) {
  if (threadIdx.x == 0) {
    bool success = try_put_signal(
        signal_pads[dst_rank] + world_size * channel + rank,
        timeout_ms,
        clock_rate_khz);
    if (!success) {
      // Use assert instead of trap because real training occasionally failed
      // to exit after the program finished. It is unclear why trap may related
      // to this issue, but assert is sufficient for this fatal error path.
      MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL(
          "MUSASymmetricMemory::put_signal failed to send signal",
          "[FATAL] MUSASymmetricMemory::put_signal: rank %d failed to send "
          "signal to rank %d on channel %d after %lu milliseconds\n",
          rank,
          dst_rank,
          channel,
          timeout_ms);
    }
  }
}

__global__ void wait_signal_kernel(
    uint32_t** signal_pads,
    int src_rank,
    int channel,
    int rank,
    int world_size,
    size_t timeout_ms,
    uint64_t clock_rate_khz) {
  if (threadIdx.x == 0) {
    bool success = try_wait_signal(
        signal_pads[rank] + world_size * channel + src_rank,
        timeout_ms,
        clock_rate_khz);
    if (!success) {
      // Use assert instead of trap because real training occasionally failed
      // to exit after the program finished. It is unclear why trap may related
      // to this issue, but assert is sufficient for this fatal error path.
      MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL(
          "MUSASymmetricMemory::wait_signal failed to receive signal",
          "[FATAL] MUSASymmetricMemory::wait_signal: rank %d failed to "
          "receive signal from rank %d on channel %d after %lu milliseconds\n",
          rank,
          src_rank,
          channel,
          timeout_ms);
    }
  }
  __threadfence_system();
}

#undef MUSA_SYMM_MEM_KERNEL_ASSERT_FAIL

uint64_t get_current_device_clock_rate_khz() {
  const int clock_rate_khz = at::musa::getCurrentDeviceProperties()->clockRate;
  TORCH_CHECK(
      clock_rate_khz > 0,
      "MUSASymmetricMemory: invalid device clock rate ",
      clock_rate_khz);
  return static_cast<uint64_t>(clock_rate_khz);
}

} // anonymous namespace

void MUSASymmetricMemory::barrier(int channel, size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::musa::MUSAGuard guard(local_device_idx_);
  const uint64_t clock_rate_khz = get_current_device_clock_rate_khz();
  barrier_kernel<<<
      1,
      std::max(at::musa::warp_size(), world_size_),
      0,
      at::musa::getCurrentMUSAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      channel,
      rank_,
      world_size_,
      timeout_ms,
      clock_rate_khz);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

void MUSASymmetricMemory::put_signal(
    int dst_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::musa::MUSAGuard guard(local_device_idx_);
  const uint64_t clock_rate_khz = get_current_device_clock_rate_khz();
  put_signal_kernel<<<
      1,
      at::musa::warp_size(),
      0,
      at::musa::getCurrentMUSAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      dst_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms,
      clock_rate_khz);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

void MUSASymmetricMemory::wait_signal(
    int src_rank,
    int channel,
    size_t timeout_ms) {
  check_channel(channel, world_size_);
  c10::musa::MUSAGuard guard(local_device_idx_);
  const uint64_t clock_rate_khz = get_current_device_clock_rate_khz();
  wait_signal_kernel<<<
      1,
      at::musa::warp_size(),
      0,
      at::musa::getCurrentMUSAStream()>>>(
      reinterpret_cast<uint32_t**>(pai_->signal_pads_dev_),
      src_rank,
      channel,
      rank_,
      world_size_,
      timeout_ms,
      clock_rate_khz);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

int MUSASymmetricMemory::get_rank() {
  return rank_;
}

int MUSASymmetricMemory::get_world_size() {
  return world_size_;
}

c10::Device MUSASymmetricMemory::get_device() {
  return c10::Device(c10::DeviceType::PrivateUse1, local_device_idx_);
}

bool MUSASymmetricMemory::world_within_direct_access() {
  return true;
}

/* End of MUSASymmetricMemory */

Block::Block(
    c10::intrusive_ptr<MUSAAllocationRef> alloc_ref,
    int device_idx,
    size_t block_size,
    size_t buffer_size,
    size_t signal_pad_offset,
    const std::optional<std::string>& group_name)
    : alloc_ref(std::move(alloc_ref)),
      device_idx(device_idx),
      block_size(block_size),
      buffer_size(buffer_size),
      signal_pad_offset(signal_pad_offset),
      default_group_name(group_name) {}

namespace {
using Expandable_Segments_Handle_Type =
    c10::musa::MUSACachingAllocator::Expandable_Segments_Handle_Type;

inline size_t round_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}
} // anonymous namespace

void* MUSASymmetricMemoryAllocator::alloc(
    size_t size,
    int device_idx,
    const std::optional<std::string>& group_name) {
  size_t signal_pad_offset = round_up(size, 16UL);
  size_t block_size = signal_pad_offset + signal_pad_size;

  c10::musa::MUSAGuard guard(device_idx);

  handle_type_ = Expandable_Segments_Handle_Type::POSIX_FD;
  MUmemAllocationProp prop{};
  prop.type = MU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device_idx;
  prop.requestedHandleTypes = MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity = 0;
  auto driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(driver_api->muMemGetAllocationGranularity_(
      &granularity, &prop, MU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
  block_size = round_up(block_size, granularity);

  HandleType handle{};
  C10_MUSA_DRIVER_CHECK(
      driver_api->muMemCreate_(&handle, block_size, &prop, 0));

  void* ptr = nullptr;
  // allocate virtual address space and map the physical memory to the virtual
  // address
  map_block(&ptr, handle, block_size, device_idx);
  C10_MUSA_CHECK(musaMemset(ptr, 0, block_size));

  auto alloc_ref = c10::make_intrusive<MUSAAllocationRef>(
      ptr, handle, block_size, device_idx);
  auto block = c10::make_intrusive<Block>(
      std::move(alloc_ref),
      device_idx,
      block_size,
      size,
      signal_pad_offset,
      group_name);
  {
    std::unique_lock lock(mutex_);
    ptr_to_block_.emplace(ptr, std::move(block));
  }
  return ptr;
}

void MUSASymmetricMemoryAllocator::free(void* ptr) {
  std::unique_lock lock(mutex_);
  ptr_to_block_.erase(ptr);
}

size_t MUSASymmetricMemoryAllocator::get_alloc_size(void* ptr) {
  auto block = find_block(ptr);
  TORCH_CHECK(
      block != nullptr,
      "MUSASymmetricMemoryAllocator::get_alloc_size: input must be "
      "allocated via MUSASymmetricMemoryAllocator::alloc");
  return block->buffer_size;
}

struct RendezvousRequest {
  int device_idx;
  int pid;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  bool has_multicast_support;
  char hostname[HOST_NAME_MAX + 1];
};

void validate_rendezvous_requests(
    const std::vector<RendezvousRequest>& reqs,
    int world_size) {
  TORCH_CHECK(reqs.size() == static_cast<size_t>(world_size));

  // For SuperPod systems, multiple blocks will have same device_idx
  // but they are on different hosts.
  // Use (hostname, device_idx) pair to uniquely identify each allocation.
  std::set<std::pair<std::string, int>> device_host_pairs;
  for (auto& req : reqs) {
    device_host_pairs.insert(
        std::make_pair(std::string(req.hostname), req.device_idx));
  }
  if (device_host_pairs.size() < (size_t)world_size) {
    TORCH_CHECK(
        false,
        "MUSASymmetricMemoryAllocator::rendezvous: ",
        "detected allocations from overlapping devices ",
        "from different ranks.");
  }

  for (int r = 1; r < world_size; ++r) {
    TORCH_CHECK(reqs[r].block_size == reqs[0].block_size);
    TORCH_CHECK(reqs[r].buffer_size == reqs[0].buffer_size);
    TORCH_CHECK(reqs[r].signal_pad_offset == reqs[0].signal_pad_offset);
  }
}

namespace {
template <bool use_fabric_handle>
c10::intrusive_ptr<MUSAPeerAllocInfo> make_peer_alloc_info(
    void* ptr,
    c10::intrusive_ptr<Block> block,
    const std::string& group_name) {
  static_assert(!use_fabric_handle, "fabric handle is not supported");
  using BlockHandleType =
      std::conditional_t<use_fabric_handle, MUmemFabricHandle, int>;

  BlockHandleType block_handle;
  c10::musa::MUSAGuard guard(block->device_idx);
  if constexpr (use_fabric_handle) {
    LOG(INFO) << "using fabric handle to import symmetric memory handles.";
  } else {
    LOG(INFO) << "using posix fd to import symmetric memory handles.";
  }

  auto group = resolve_process_group(group_name);
  auto rank = group->getRank();
  auto world_size = group->getSize();
  auto store = group->getStore();

  // Note: don't move ipc_channel construction closer to the use
  // there needs to be a barrier between constructor and first use,
  // and this barrier is provided when we are exchanging rendezvous requests
  using IpcChannelType = std::conditional_t<use_fabric_handle, int, IpcChannel>;
  IpcChannelType ipc_channel;

  auto driver_api = c10::musa::DriverAPI::get();
  C10_MUSA_DRIVER_CHECK(driver_api->muMemExportToShareableHandle_(
      &block_handle,
      block->alloc_ref->handle,
      use_fabric_handle ? MU_MEM_HANDLE_TYPE_FABRIC
                        : MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
      0));
  RendezvousRequest local_req = RendezvousRequest{
      .device_idx = block->device_idx,
      .pid = getpid(),
      .block_size = block->block_size,
      .buffer_size = block->buffer_size,
      .signal_pad_offset = block->signal_pad_offset,
      .has_multicast_support = false};

  gethostname(local_req.hostname, sizeof(local_req.hostname));
  auto reqs = storeExchange.all_gather(store, rank, world_size, local_req);
  validate_rendezvous_requests(reqs, world_size);

  std::vector<int> pids(world_size);
  for (int r = 0; r < world_size; r++) {
    pids[r] = reqs[r].pid;
  }

  std::vector<BlockHandleType> imported_handles;
  if constexpr (!use_fabric_handle) {
    imported_handles = ipc_channel.all_gather_fds(rank, pids, block_handle);
  } else {
    imported_handles =
        storeExchange.all_gather(store, rank, world_size, block_handle);
  }

  std::vector<HandleType> handles(world_size);
  std::vector<void*> buffers(world_size, nullptr);
  std::vector<void*> signal_pads(world_size, nullptr);

  for (int r = 0; r < world_size; r++) {
    if (r == rank) {
      handles[r] = block->alloc_ref->handle;
      buffers[r] = ptr;
      signal_pads[r] = (void*)((uintptr_t)ptr + block->signal_pad_offset);
      continue;
    }
    if constexpr (!use_fabric_handle) {
      C10_MUSA_DRIVER_CHECK(driver_api->muMemImportFromShareableHandle_(
          &handles[r],
          (void*)(uintptr_t)imported_handles[r],
          MU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
    } else {
      C10_MUSA_DRIVER_CHECK(driver_api->muMemImportFromShareableHandle_(
          &handles[r],
          (void*)&(imported_handles[r]),
          MU_MEM_HANDLE_TYPE_FABRIC));
    }

    map_block(&buffers[r], handles[r], block->block_size, block->device_idx);
    signal_pads[r] = (void*)((uintptr_t)buffers[r] + block->signal_pad_offset);
    if constexpr (!use_fabric_handle) {
      close(imported_handles[r]);
    }
  }
  storeExchange.barrier(store, rank, world_size);
  if constexpr (!use_fabric_handle) {
    close(block_handle);
  }

  // multicast not supported
  HandleType mc_handle{};
  void* mc_addr = nullptr;

  std::vector<c10::intrusive_ptr<MUSAAllocationRef>> alloc_refs;
  for (int r = 0; r < world_size; r++) {
    if (r == rank) {
      alloc_refs.emplace_back(block->alloc_ref);
      continue;
    }
    alloc_refs.push_back(c10::make_intrusive<MUSAAllocationRef>(
        buffers[r], handles[r], block->block_size, block->device_idx));
  }

  return c10::make_intrusive<MUSAPeerAllocInfo>(
      std::move(alloc_refs),
      std::move(buffers),
      std::move(signal_pads),
      mc_handle,
      mc_addr,
      block->buffer_size,
      block->device_idx,
      rank,
      world_size);
}

} // anonymous namespace

c10::intrusive_ptr<SymmetricMemory> MUSASymmetricMemoryAllocator::rendezvous(
    void* ptr,
    const std::optional<std::string>& group_name) {
  size_t offset = 0;
  auto block = find_block_covering(ptr, offset);
  if (block == nullptr) {
    return nullptr;
  }

  std::string group_name_;
  if (group_name.has_value() && *group_name != "") {
    group_name_ = *group_name;
  } else {
    TORCH_CHECK(
        block->default_group_name.has_value(),
        "MUSASymmetricMemory::rendezvous: `group_name` is neither "
        "specified during allocation nor passed to rendezvous().");
    group_name_ = *block->default_group_name;
  }

  auto it = block->symm_mems.find(group_name_);
  if (it == block->symm_mems.end()) {
    TORCH_INTERNAL_ASSERT(
        handle_type_ != Expandable_Segments_Handle_Type::UNSPECIFIED);
    auto pai = make_peer_alloc_info<false>(ptr, block, group_name_);
    it = block->symm_mems.emplace(group_name_, pai).first;
  }
  return c10::make_intrusive<MUSASymmetricMemory>(it->second, offset);
}

c10::intrusive_ptr<Block> MUSASymmetricMemoryAllocator::find_block(void* ptr) {
  std::shared_lock lock(mutex_);
  auto it = ptr_to_block_.find(ptr);
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

c10::intrusive_ptr<Block> MUSASymmetricMemoryAllocator::find_block_covering(
    void* ptr,
    size_t& offset) {
  std::shared_lock lock(mutex_);
  auto it = std::find_if(
      ptr_to_block_.begin(), ptr_to_block_.end(), [&](const auto& item) {
        const auto& block = item.second;
        auto base_ptr = reinterpret_cast<uintptr_t>(block->alloc_ref->ptr);
        auto ptr_int = reinterpret_cast<uintptr_t>(ptr);
        // Modify offset so that it is returned
        offset = ptr_int - base_ptr;
        return ptr_int >= base_ptr && offset < block->buffer_size;
      });
  if (it == ptr_to_block_.end()) {
    return nullptr;
  }
  return it->second;
}

bool MUSASymmetricMemoryAllocator::has_multicast_support(int device_idx) {
  return device_has_multicast_support(device_idx);
}

c10::DeviceType MUSASymmetricMemoryAllocator::supported_device_type() {
  return c10::DeviceType::PrivateUse1;
}

std::string MUSASymmetricMemoryAllocator::name() {
  return "MUSA";
}

struct RegisterMUSASymmetricMemoryAllocator {
  RegisterMUSASymmetricMemoryAllocator() {
    auto allocator = c10::make_intrusive<MUSASymmetricMemoryAllocator>();
    register_allocator(c10::DeviceType::PrivateUse1, allocator);
  }
};

static RegisterMUSASymmetricMemoryAllocator register_allocator_;

} // namespace symmetric_memory
} // namespace c10d
