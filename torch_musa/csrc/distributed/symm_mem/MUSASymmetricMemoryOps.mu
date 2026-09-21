#include <ATen/ATen.h>
#include <ATen/ceil_div.h>
#include <torch/library.h>

#include <vector>

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"
#include "torch_musa/csrc/core/MUSAEvent.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/core/driver_api.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_like.h>
#endif

#include "torch_musa/csrc/distributed/symm_mem/MUSAAsyncMM.hpp"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemory-inl.h"
#include "torch_musa/csrc/distributed/symm_mem/MUSASymmetricMemory.hpp"

#define INT_SWITCH_CASE(name, val, ...) \
  case val: {                           \
    constexpr int name = val;           \
    __VA_ARGS__();                      \
    break;                              \
  }

#define DISPATCH_WORLD_SIZES_NO_DEFAULT(world_size, ...)                 \
  switch (world_size) {                                                  \
    INT_SWITCH_CASE(k_world_size, 8, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 4, __VA_ARGS__);                       \
    INT_SWITCH_CASE(k_world_size, 2, __VA_ARGS__);                       \
    default: {                                                           \
      TORCH_CHECK(false, "Not implemented for world_size=", world_size); \
    }                                                                    \
  }

#define DISPATCH_ALIGNMENTS_16_8_4(alignment, ...)                     \
  switch (alignment) {                                                 \
    INT_SWITCH_CASE(k_alignment, 16, __VA_ARGS__);                     \
    INT_SWITCH_CASE(k_alignment, 8, __VA_ARGS__);                      \
    INT_SWITCH_CASE(k_alignment, 4, __VA_ARGS__);                      \
    default: {                                                         \
      TORCH_CHECK(false, "Not implemented for alignment=", alignment); \
    }                                                                  \
  }

#define AT_DISPATCH_FLOAT_AND_BFLOAT16(scalar_type, name, ...)         \
  AT_DISPATCH_SWITCH(                                                  \
      scalar_type, name, AT_DISPATCH_CASE(at::kBFloat16, __VA_ARGS__); \
      AT_DISPATCH_CASE(at::kFloat, __VA_ARGS__));

namespace {

using namespace c10d::symmetric_memory;

size_t get_and_verify_alignment(
    const at::Tensor& input,
    const char* op_name,
    size_t base_offset_bytes = 0) {
  const size_t min_alignment = std::max(4l, input.element_size());
  // Only check the offset since the multicast address is always at least
  // 128-bit aligned
  const size_t input_offset_bytes = base_offset_bytes +
      static_cast<size_t>(input.storage_offset()) * input.element_size();
  const size_t ptr_alignment =
      at::native::memory::get_alignment(input_offset_bytes);
  TORCH_CHECK(
      ptr_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input ptr + offset must be at least ",
      min_alignment,
      "-byte aligned.");

  const size_t size_alignment = at::native::memory::get_alignment(
      static_cast<size_t>(input.numel() * input.element_size()));
  TORCH_CHECK(
      size_alignment >= min_alignment,
      op_name,
      "<",
      input.scalar_type(),
      ">: input size must be at least ",
      min_alignment,
      "-byte aligned.");
  return std::min(ptr_alignment, size_alignment);
}

size_t get_symmetric_offset_bytes(
    const at::Tensor& input,
    const c10::intrusive_ptr<SymmetricMemory>& symm_mem) {
  return symm_mem->get_offset() +
      static_cast<size_t>(input.storage_offset()) * input.element_size();
}

size_t get_symmetric_offset_elements(
    const at::Tensor& input,
    const c10::intrusive_ptr<SymmetricMemory>& symm_mem,
    const char* op_name) {
  const size_t offset_bytes = get_symmetric_offset_bytes(input, symm_mem);
  TORCH_CHECK(
      offset_bytes % input.element_size() == 0,
      op_name,
      ": symmetric memory offset must be divisible by element size.");
  return offset_bytes / input.element_size();
}

MUatomicType get_memory_atomic_add_type(
    c10::ScalarType scalar_type,
    const char* op_name) {
  if (scalar_type == c10::ScalarType::Float) {
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD_F32;
  }
  if (scalar_type == c10::ScalarType::BFloat16) {
    return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD_BF16;
  }
  TORCH_CHECK(
      false,
      op_name,
      ": only float and bfloat16 are supported for memory atomic.");
  return MUatomicType::MU_ATOMIC_TYPE_ATOMIC_ADD64;
}

template <typename GetSrcPtr>
void launch_memory_atomic_for_peers(
    void* dst,
    size_t element_count,
    MUatomicType atomic_type,
    int rank,
    int world_size,
    const at::musa::MUSAStream& stream,
    GetSrcPtr get_src_ptr) {
  auto* driver_api = c10::musa::DriverAPI::get();

  TORCH_CHECK(
      world_size <= 8, "memory atomic collectives support world_size <= 8");

  if (world_size <= 1) {
    C10_MUSA_DRIVER_CHECK(driver_api->muMemoryAtomicAsync_(
        (MUdeviceptr)dst,
        (MUdeviceptr)get_src_ptr(rank),
        element_count,
        atomic_type,
        (MUstream)stream.stream()));
    return;
  }

  at::musa::MUSAEvent atomic_ready(musaEventDisableTiming);
  atomic_ready.record(stream);

  std::vector<at::musa::MUSAEvent> atomic_done_events;
  atomic_done_events.reserve(world_size);
  const int stream_priority = stream.priority();
  for (int step = 0; step < world_size; ++step) {
    const int peer = (rank + step) % world_size;
    auto peer_stream =
        at::musa::getStreamFromPool(stream_priority, stream.device_index());
    atomic_ready.block(peer_stream);

    C10_MUSA_DRIVER_CHECK(driver_api->muMemoryAtomicAsync_(
        (MUdeviceptr)dst,
        (MUdeviceptr)get_src_ptr(peer),
        element_count,
        atomic_type,
        (MUstream)peer_stream.stream()));

    atomic_done_events.emplace_back(musaEventDisableTiming);
    atomic_done_events.back().record(peer_stream);
  }

  for (auto& event : atomic_done_events) {
    event.block(stream);
  }
}

void reduce_scatter_out_memory_atomic(
    const at::Tensor& input,
    const c10::intrusive_ptr<SymmetricMemory>& symm_mem,
    at::Tensor& output) {
  const int rank = symm_mem->get_rank();
  const int world_size = symm_mem->get_world_size();
  const size_t output_numel = static_cast<size_t>(output.numel());
  const size_t output_bytes = output_numel * output.element_size();
  const size_t input_offset_bytes = get_symmetric_offset_bytes(input, symm_mem);
  const auto atomic_type =
      get_memory_atomic_add_type(input.scalar_type(), "reduce_scatter");
  const auto stream = at::musa::getCurrentMUSAStream();
  const auto peer_ptrs = symm_mem->get_buffer_ptrs();

  // Match the kernel-side entry/exit barriers: all peer inputs must be ready
  // before CE reads them, and all CE reads must finish before inputs are
  // reused.
  symm_mem->barrier(0, 0);
  C10_MUSA_CHECK(musaMemsetAsync(output.data_ptr(), 0, output_bytes, stream));
  launch_memory_atomic_for_peers(
      output.data_ptr(),
      output_numel,
      atomic_type,
      rank,
      world_size,
      stream,
      [&](int peer) {
        TORCH_CHECK(
            peer_ptrs[peer] != nullptr,
            "reduce_scatter: peer buffer is not directly accessible, peer: ",
            peer);
        return static_cast<char*>(peer_ptrs[peer]) + input_offset_bytes +
            rank * output_bytes;
      });
  symm_mem->barrier(0, 0);
}

void all_reduce_out_memory_atomic(
    const at::Tensor& input,
    const c10::intrusive_ptr<SymmetricMemory>& symm_mem,
    at::Tensor& output,
    const std::optional<at::Tensor>& local_input,
    const char* op_name) {
  const int rank = symm_mem->get_rank();
  const int world_size = symm_mem->get_world_size();
  const size_t input_numel = static_cast<size_t>(input.numel());
  const size_t input_bytes = input_numel * input.element_size();
  const size_t input_offset_bytes = get_symmetric_offset_bytes(input, symm_mem);
  const auto atomic_type =
      get_memory_atomic_add_type(input.scalar_type(), op_name);
  const auto stream = at::musa::getCurrentMUSAStream();
  const auto peer_ptrs = symm_mem->get_buffer_ptrs();

  TORCH_CHECK(
      output.data_ptr() != input.data_ptr(),
      op_name,
      ": memory atomic all-reduce output must not alias input; use "
      "two_shot_all_reduce_ for in-place all-reduce.");
  TORCH_CHECK(
      peer_ptrs[rank] != nullptr,
      op_name,
      ": local symmetric buffer is not directly accessible, rank: ",
      rank);
  if (local_input.has_value()) {
    TORCH_CHECK(
        local_input->numel() == input.numel(),
        op_name,
        ": local input size must match symm buffer size.");
    auto local_dst = static_cast<char*>(peer_ptrs[rank]) + input_offset_bytes;
    C10_MUSA_CHECK(musaMemcpyAsync(
        local_dst,
        local_input->data_ptr(),
        input_bytes,
        musaMemcpyDeviceToDevice,
        stream));
  }

  // Ensure optional local copies are visible before CE reads peer buffers, and
  // keep peers from reusing input until all atomic reads have completed.
  symm_mem->barrier(0, 0);
  C10_MUSA_CHECK(musaMemsetAsync(output.data_ptr(), 0, input_bytes, stream));
  launch_memory_atomic_for_peers(
      output.data_ptr(),
      input_numel,
      atomic_type,
      rank,
      world_size,
      stream,
      [&](int peer) {
        TORCH_CHECK(
            peer_ptrs[peer] != nullptr,
            op_name,
            ": peer buffer is not directly accessible, peer: ",
            peer);
        return static_cast<char*>(peer_ptrs[peer]) + input_offset_bytes;
      });
  symm_mem->barrier(0, 0);
}

void init_elementwise_launch_config(
    size_t numel,
    size_t element_size,
    size_t alignment,
    size_t splits,
    size_t max_num_blocks,
    size_t max_num_threads,
    int& num_blocks,
    int& num_threads,
    int world_size) {
  // Align to preserve alignment in each split
  const size_t aligned_numel = at::round_up(numel, alignment * splits);
  const size_t numel_per_split = aligned_numel / splits;
  const size_t numel_per_thread = alignment / element_size;

  if (numel_per_split <= max_num_threads * numel_per_thread) {
    num_blocks = 1;
    num_threads = at::ceil_div(numel_per_split, numel_per_thread);
    // `sync_remote_blocks` maps threads to peers, so we need to make sure there
    // are enough threads
    num_threads = max(num_threads, world_size);
    num_threads = at::round_up(num_threads, at::musa::warp_size());
  } else {
    num_blocks = std::min(
        at::ceil_div(numel_per_split, max_num_threads * numel_per_thread),
        max_num_blocks);
    num_threads = max_num_threads;
  }
}

at::Tensor one_shot_all_reduce_out_impl(
    const at::Tensor& input,
    const std::optional<at::Tensor>& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  TORCH_CHECK(
      input.is_contiguous(), "one_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      out.is_contiguous(), "one_shot_all_reduce: output must be contiguous.");
  TORCH_CHECK(
      out.scalar_type() == input.scalar_type(),
      "one_shot_all_reduce: input/output dtype mismatch, input.scalar_type(): ",
      input.scalar_type(),
      ", output.scalar_type(): ",
      out.scalar_type());
  TORCH_CHECK(
      out.sizes() == input.sizes(),
      "one_shot_all_reduce: input/output size mismatch, input.sizes(): ",
      input.sizes(),
      ", output.sizes(): ",
      out.sizes());
  TORCH_CHECK(
      reduce_op == "sum",
      "one_shot_all_reduce: only sum is supported for now.");
  if (local_input.has_value()) {
    TORCH_CHECK(
        local_input->is_contiguous(),
        "one_shot_all_reduce: local input must be contiguous.");
    TORCH_CHECK(
        local_input->scalar_type() == input.scalar_type(),
        "one_shot_all_reduce: local input dtype must match symm buffer dtype.");
    TORCH_CHECK(
        local_input->numel() == input.numel(),
        "one_shot_all_reduce: local input size must match symm buffer size.");
  }
  if (input.numel() == 0) {
    return out;
  }
  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "one_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment = get_and_verify_alignment(
      input, "one_shot_all_reduce", symm_mem->get_offset());
  if (local_input.has_value()) {
    const size_t local_alignment =
        get_and_verify_alignment(*local_input, "one_shot_all_reduce");
    TORCH_CHECK(
        alignment == local_alignment,
        "one_shot_all_reduce: local input and symm buffer must have the same alignment.");
  }

  all_reduce_out_memory_atomic(
      input, symm_mem, out, local_input, "one_shot_all_reduce");
  return out;
}

at::Tensor one_shot_all_reduce_out(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, std::nullopt, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce_copy_out(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor out) {
  return one_shot_all_reduce_out_impl(
      input, local_input, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce(
    const at::Tensor& input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(input);
  return one_shot_all_reduce_out_impl(
      input, std::nullopt, reduce_op, group_name, out);
}

at::Tensor one_shot_all_reduce_copy(
    const at::Tensor& input,
    const at::Tensor& local_input,
    std::string reduce_op,
    std::string group_name) {
  auto out = at::empty_like(local_input);
  return one_shot_all_reduce_out_impl(
      input, local_input, reduce_op, group_name, out);
}

constexpr size_t reduce_scatter_max_num_blocks = 24;
constexpr size_t reduce_scatter_max_num_threads = 1024;
template <
    typename T,
    int alignment,
    int k_world_size,
    bool split_last_dim = false>
static __launch_bounds__(reduce_scatter_max_num_threads) __global__
    void reduce_scatter_kernel(
        T** input_ptrs,
        T* output_ptr,
        size_t input_offset,
        size_t numel,
        uint32_t** signal_pads,
        size_t rank,
        size_t world_size,
        size_t last_dim_size = 0) {
  static_assert(alignment % sizeof(T) == 0);
  constexpr size_t numel_per_thread = alignment / sizeof(T);
  int32_t N_last_dim =
      last_dim_size / world_size; // used only for split_last_dim reduce_scatter
  sync_remote_blocks<false, true>(signal_pads, rank, world_size);
  __SYNCTHREADS;

  const size_t numel_per_rank =
      at::round_up(numel, numel_per_thread * world_size) / world_size;
  const size_t start = split_last_dim ? last_dim_size / world_size * rank
                                      : numel_per_rank * rank;

  auto offset = (blockDim.x * blockIdx.x + threadIdx.x) * numel_per_thread;
  auto stride = blockDim.x * gridDim.x * numel_per_thread;
  for (size_t i = offset; i < numel_per_rank; i += stride) {
    size_t idx = i;
    if constexpr (split_last_dim) {
      idx = i / N_last_dim * last_dim_size + i % N_last_dim;
    }
    auto vec = load_and_reduce<T, alignment, k_world_size>(
        input_ptrs, rank, world_size, input_offset + start + idx);
    at::native::memory::st_vec<alignment>(output_ptr + i, vec);
  }

  __SYNCTHREADS;
  sync_remote_blocks<true, true>(signal_pads, rank, world_size);
}

at::Tensor two_shot_all_reduce_impl(
    at::Tensor input,
    std::optional<at::Tensor> output,
    std::string reduce_op,
    std::string group_name) {
  TORCH_CHECK(
      input.is_contiguous(), "two_shot_all_reduce: input must be contiguous.");
  TORCH_CHECK(
      reduce_op == "sum",
      "two_shot_all_reduce: only sum is supported for now.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "two_shot_all_reduce: input must be allocated with empty_strided_p2p().");

  const size_t alignment = get_and_verify_alignment(
      input, "two_shot_all_reduce", symm_mem->get_offset());

  if (output.has_value()) {
    TORCH_CHECK(
        output->is_contiguous(),
        "two_shot_all_reduce: output must be contiguous.");
    const size_t output_alignment =
        get_and_verify_alignment(*output, "two_shot_all_reduce");
    TORCH_CHECK(
        alignment <= output_alignment,
        "two_shot_all_reduce: output alignment must be equal to or larger than input.");
    TORCH_CHECK(
        output->sizes() == input.sizes(),
        "two_shot_all_reduce: input/output size mismatch, input.sizes(): ",
        input.sizes(),
        ", output.sizes(): ",
        output->sizes());
    TORCH_CHECK(
        output->scalar_type() == input.scalar_type(),
        "two_shot_all_reduce: input/output dtype mismatch, input.scalar_type(): ",
        input.scalar_type(),
        ", output.scalar_type(): ",
        output->scalar_type());
    if (input.numel() == 0) {
      return *output;
    }
  } else {
    if (input.numel() == 0) {
      return input;
    }
  }

  if (!output.has_value()) {
    auto tmp = at::empty_like(input);
    all_reduce_out_memory_atomic(
        input, symm_mem, tmp, std::nullopt, "two_shot_all_reduce");
    const auto stream = at::musa::getCurrentMUSAStream();
    C10_MUSA_CHECK(musaMemcpyAsync(
        input.data_ptr(),
        tmp.data_ptr(),
        static_cast<size_t>(input.numel()) * input.element_size(),
        musaMemcpyDeviceToDevice,
        stream));
    symm_mem->barrier(0, 0);
    return input;
  } else {
    all_reduce_out_memory_atomic(
        input, symm_mem, *output, std::nullopt, "two_shot_all_reduce");
    return *output;
  }
}

at::Tensor two_shot_all_reduce_(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name) {
  return two_shot_all_reduce_impl(input, std::nullopt, reduce_op, group_name);
}

at::Tensor two_shot_all_reduce_out(
    at::Tensor input,
    std::string reduce_op,
    std::string group_name,
    at::Tensor output) {
  return two_shot_all_reduce_impl(input, output, reduce_op, group_name);
}

at::Tensor reduce_scatter_out(
    at::Tensor input,
    std::string group_name,
    bool split_last_dim,
    at::Tensor output) {
  TORCH_CHECK(
      input.is_contiguous(), "reduce_scatter: input must be contiguous.");
  TORCH_CHECK(
      output.is_contiguous(), "reduce_scatter: output must be contiguous.");

  auto symm_mem = c10d::symmetric_memory::rendezvous(input, group_name);
  TORCH_CHECK(
      symm_mem != nullptr,
      "reduce_scatter: input must be allocated with empty_strided_p2p().");

  const size_t alignment =
      get_and_verify_alignment(input, "reduce_scatter", symm_mem->get_offset());

  const size_t output_alignment =
      get_and_verify_alignment(output, "reduce_scatter");

  TORCH_CHECK(
      input.numel() %
              (symm_mem->get_world_size() *
               (alignment / input.element_size())) ==
          0,
      "expected number of elements to be divisible by world_size * alignment, number of elements ",
      input.numel(),
      " world size ",
      symm_mem->get_world_size(),
      "alignment ",
      alignment);

  if (split_last_dim) {
    TORCH_CHECK(input.dim() == output.dim());
    bool are_equal_except_last = std::equal(
        input.sizes().begin(), input.sizes().end() - 1, output.sizes().begin());
    TORCH_CHECK(
        are_equal_except_last,
        "reduce_scatter expected input and output to have same sizes except in the last dimension");
    TORCH_CHECK(
        output.size(-1) == input.size(-1) / symm_mem->get_world_size(),
        "reduce_scatter expected output last dim size to be input last dim size / world_size");

    TORCH_CHECK(
        input.size(-1) %
                (symm_mem->get_world_size() *
                 (alignment / input.element_size())) ==
            0,
        "expected last dimension to be divisible by world_size * alignment, last dimension ",
        input.size(-1),
        " world size ",
        symm_mem->get_world_size(),
        "alignment ",
        alignment);
  } else {
    TORCH_CHECK(input.dim() == 1, "reduce_scatter expected 1D input");
    TORCH_CHECK(output.dim() == 1, "reduce_scatter expected 1D output");
    TORCH_CHECK(output.numel() == input.numel() / symm_mem->get_world_size());
  }
  TORCH_CHECK(
      output.scalar_type() == input.scalar_type(),
      "reduce_scatter: input/output dtype mismatch, input.scalar_type(): ",
      input.scalar_type(),
      ", output.scalar_type(): ",
      output.scalar_type());
  if (input.numel() == 0) {
    return output;
  }

  TORCH_CHECK(
      output_alignment >= alignment,
      "reduce_scatter: output alignment should be not smaller than input alignment");

  int num_blocks = 0, num_threads = 0;
  init_elementwise_launch_config(
      input.numel(),
      input.element_size(),
      alignment,
      symm_mem->get_world_size(),
      reduce_scatter_max_num_blocks,
      reduce_scatter_max_num_threads,
      num_blocks,
      num_threads,
      symm_mem->get_world_size());
  if (split_last_dim) {
    AT_DISPATCH_FLOAT_AND_BFLOAT16(
        input.scalar_type(), "reduce_scatter", [&]() {
          DISPATCH_ALIGNMENTS_16_8_4(alignment, [&]() {
            DISPATCH_WORLD_SIZES_NO_DEFAULT(symm_mem->get_world_size(), [&]() {
              reduce_scatter_kernel<scalar_t, k_alignment, k_world_size, true>
                  <<<num_blocks,
                     num_threads,
                     0,
                     at::musa::getCurrentMUSAStream()>>>(
                      reinterpret_cast<scalar_t**>(
                          symm_mem->get_buffer_ptrs_dev()),
                      output.data_ptr<scalar_t>(),
                      get_symmetric_offset_elements(
                          input, symm_mem, "reduce_scatter"),
                      input.numel(),
                      reinterpret_cast<uint32_t**>(
                          symm_mem->get_signal_pad_ptrs_dev()),
                      symm_mem->get_rank(),
                      symm_mem->get_world_size(),
                      input.size(-1));
              C10_MUSA_KERNEL_LAUNCH_CHECK();
            });
          });
        });
  } else {
    reduce_scatter_out_memory_atomic(input, symm_mem, output);
  }
  return output;
}
} // namespace

namespace {

at::Tensor memset32_(
    at::Tensor& input,
    int64_t offset,
    int64_t val,
    int64_t count) {
  TORCH_CHECK(
      input.dim() == 1 && input.is_contiguous() &&
          input.scalar_type() == c10::ScalarType::UInt32,
      "symm_mem::memset32_: input must be a flat, contiguous uint32 tensor.");

  TORCH_CHECK(
      offset >= 0,
      "symm_mem::memset32_: offset must be greater than or equal to 0 (got ",
      offset,
      ")");

  TORCH_CHECK(
      count > 0,
      "symm_mem::memset32_: count must be a positive integer (got ",
      count,
      ")");

  TORCH_CHECK(
      val >= 0 &&
          static_cast<size_t>(val) <= std::numeric_limits<uint32_t>::max(),
      "symm_mem::memset32_: val must be in the range of "
      "[0, 4294967295] (uint32_t).");

  TORCH_CHECK(
      offset + count <= input.numel(),
      "symm_mem::memset32_: offset + count (",
      offset + count,
      ") exceeded the numel of the input (",
      input.numel(),
      ")");

  auto addr = reinterpret_cast<uint32_t*>(input.data_ptr()) + offset;
  c10::musa::MUSAGuard guard(input.device());
  auto* driver_api = c10::musa::DriverAPI::get();

  C10_MUSA_DRIVER_CHECK(driver_api->muMemsetD32Async_(
      reinterpret_cast<MUdeviceptr>(addr),
      val,
      count,
      at::musa::getCurrentMUSAStream()));
  return input;
}

at::Tensor stream_write_value32_(
    at::Tensor& input,
    int64_t offset,
    int64_t val) {
  TORCH_CHECK(
      input.dim() == 1 && input.is_contiguous() &&
          input.scalar_type() == c10::ScalarType::UInt32,
      "symm_mem::stream_write_value32_: input must be a flat, contiguous "
      "uint32 tensor.");

  TORCH_CHECK(
      offset >= 0,
      "symm_mem::stream_write_value32_: offset must be greater than or "
      "equal to 0 (got ",
      offset,
      ")");

  TORCH_CHECK(
      val >= 0 &&
          static_cast<size_t>(val) <= std::numeric_limits<uint32_t>::max(),
      "symm_mem::stream_write_value32_: "
      "val must be in the range of [0, 4294967295] (uint32_t).");

  TORCH_CHECK(
      offset < input.numel(),
      "symm_mem::stream_write_value32_: offset (",
      offset,
      ") exceeded the numel of the input (",
      input.numel(),
      ")");

  auto addr = reinterpret_cast<uint32_t*>(input.data_ptr()) + offset;
  c10::musa::MUSAGuard guard(input.device());
  auto* driver_api = c10::musa::DriverAPI::get();

  // According to the documentation of MUstreamWriteValue_flags,
  // muStreamWriteValue32 will provide a memory fence before the write, which
  // has similar semantics to __threadfence_system() but is scoped to the
  // stream rather than a MUSA thread.
  C10_MUSA_DRIVER_CHECK(driver_api->muStreamWriteValue32_(
      at::musa::getCurrentMUSAStream(),
      reinterpret_cast<MUdeviceptr>(addr),
      val,
      0));
  return input;
}

} // namespace

TORCH_LIBRARY_IMPL(symm_mem, PrivateUse1, m) {
  m.impl("one_shot_all_reduce", ::one_shot_all_reduce);
  m.impl("one_shot_all_reduce_out", ::one_shot_all_reduce_out);
  m.impl("one_shot_all_reduce_copy", ::one_shot_all_reduce_copy);
  m.impl("one_shot_all_reduce_copy_out", ::one_shot_all_reduce_copy_out);
  m.impl("two_shot_all_reduce_", ::two_shot_all_reduce_);
  m.impl("two_shot_all_reduce_out", ::two_shot_all_reduce_out);
  m.impl("reduce_scatter_out", ::reduce_scatter_out);
  // TODO: The current MUSA _async_input_mm implementation validates the async
  // input contract but falls back to at::mm_out, so a_chunk_signals and
  // a_chunk_pivot do not yet drive chunk readiness or compute order. This is
  // correct only when the full A matrix is already available; it cannot overlap
  // all-gather/copy with GEMM the way the CUDA CUTLASS async-input scheduler
  // does. Replace this with a MUTLASS kernel that waits on chunk signals before
  // consuming each A chunk and pivots the M-tile rasterization accordingly.
  m.impl("_async_input_mm", c10d::musa::detail::async_input_mm);
  m.impl("stream_write_value32_", ::stream_write_value32_);
  m.impl("memset32_", ::memset32_);
}
