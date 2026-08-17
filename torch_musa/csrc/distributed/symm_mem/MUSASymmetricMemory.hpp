#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/Store.hpp>
#include <torch/csrc/distributed/c10d/symm_mem/SymmetricMemory.hpp>

#include "torch_musa/csrc/core/MUSAAllocatorConfig.h"
#include "torch_musa/csrc/core/driver_api.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemoryTypes.hpp"


namespace c10d {
namespace symmetric_memory {

// Resource wrapper that owns a (vaddr, allocation handle) pair. Upon
// destruction, it unmaps the vaddr and releases the allocation handle.
struct MUSAAllocationRef : public c10::intrusive_ptr_target {
  void* ptr;
  MUmemGenericAllocationHandle handle;
  size_t block_size;
  int device_idx;
  bool is_multicast;

  MUSAAllocationRef(
      void* ptr,
      MUmemGenericAllocationHandle handle,
      size_t block_size,
      int device_idx,
      bool is_multicast = false);

  ~MUSAAllocationRef();
};

// Forward declaration of MUSAPeerAllocInfo
class MUSAPeerAllocInfo;

class MUSASymmetricMemory : public SymmetricMemory {
 public:
  // This is mostly a shallow copy that shares the pointer to
  // `MUSAPeerAllocInfo` which corresponds to the base Block. The
  // MUSASymmetricMemory handle is specified by the offset to the base ptr.
  MUSASymmetricMemory(
      const c10::intrusive_ptr<MUSAPeerAllocInfo>& pai,
      size_t offset);

  ~MUSASymmetricMemory() override {};

  std::vector<void*> get_buffer_ptrs() override;
  std::vector<void*> get_signal_pad_ptrs() override;
  void** get_buffer_ptrs_dev() override;
  void** get_signal_pad_ptrs_dev() override;
  size_t get_buffer_size() override;
  size_t get_offset() override;

  bool has_multicast_support() override;
  void* get_multicast_ptr() override;

  void barrier(int channel, size_t timeout_ms) override;
  void put_signal(int dst_rank, int channel, size_t timeout_ms) override;
  void wait_signal(int src_rank, int channel, size_t timeout_ms) override;

  int get_rank() override;
  int get_world_size() override;
  c10::Device get_device() override;
  bool world_within_direct_access() override;

 private:
  int local_device_idx_;
  int rank_;
  int world_size_;
  c10::intrusive_ptr<MUSAPeerAllocInfo> pai_;
  size_t offset_{0}; // in byte
};

// A class to hold the base pointers and signal pad pointers for a group of
// peers. One `MUSAPeerAllocInfo` object can be shared by multiple
// `MUSASymmetricMemory` objects when latter reside on the same allocation
// and rendezvous over the same group. (The `MUSASymmetricMemory` objects may
// have different offsets compared to the base address.)
class MUSAPeerAllocInfo : public c10::intrusive_ptr_target {
 public:
  MUSAPeerAllocInfo(
      std::vector<c10::intrusive_ptr<MUSAAllocationRef>> alloc_refs,
      std::vector<void*> buffers,
      std::vector<void*> signal_pads,
      MUmemGenericAllocationHandle mc_handle,
      void* mc_addr,
      size_t buffer_size,
      int local_device_idx,
      int rank,
      int world_size);

 private:
  std::vector<c10::intrusive_ptr<MUSAAllocationRef>> alloc_refs_;
  std::vector<void*> buffers_;
  std::vector<void*> signal_pads_;
  MUmemGenericAllocationHandle mc_handle_;
  void* mc_addr_;
  size_t buffer_size_;
  int local_device_idx_;
  int rank_;
  int world_size_;
  void** buffers_dev_;
  void** signal_pads_dev_;

  friend class MUSASymmetricMemory;
};

// Metadata associated with each allocation performed by
// `MUSASymmetricMemoryAllocator`.
struct Block : public c10::intrusive_ptr_target {
  c10::intrusive_ptr<MUSAAllocationRef> alloc_ref;
  int device_idx;
  size_t block_size;
  size_t buffer_size;
  size_t signal_pad_offset;
  std::optional<std::string> default_group_name;
  std::map<std::string, c10::intrusive_ptr<MUSAPeerAllocInfo>> symm_mems;

  Block(
      c10::intrusive_ptr<MUSAAllocationRef> alloc_ref,
      int device_idx,
      size_t block_size,
      size_t buffer_size,
      size_t signal_pad_offset,
      const std::optional<std::string>& group_name);
};

class MUSASymmetricMemoryAllocator : public SymmetricMemoryAllocator {
 public:
  void* alloc(
      size_t size,
      int device_idx,
      const std::optional<std::string>& group_name) override;

  void free(void* ptr) override;
  size_t get_alloc_size(void* ptr) override;
  c10::intrusive_ptr<SymmetricMemory> rendezvous(
      void* ptr,
      const std::optional<std::string>& group_name) override;
  bool has_multicast_support(int device_idx) override;
  c10::DeviceType supported_device_type() override;
  std::string name() override;

 private:
  c10::intrusive_ptr<Block> find_block(void* ptr);
  c10::intrusive_ptr<Block> find_block_covering(void* ptr, size_t& offset);

  std::shared_mutex mutex_;
  std::unordered_map<void*, c10::intrusive_ptr<Block>> ptr_to_block_;
  c10::musa::MUSACachingAllocator::Expandable_Segments_Handle_Type
      handle_type_ = c10::musa::MUSACachingAllocator::
          Expandable_Segments_Handle_Type::UNSPECIFIED;
};

} // namespace symmetric_memory
} // namespace c10d
