#ifndef TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGENERATORIMPL_H_
#define TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGENERATORIMPL_H_

#include <ATen/Context.h>
#include <ATen/core/Generator.h>
#include <atomic>
#include <limits>

#include <ATen/musa/PhiloxCudaState.h>

namespace at {
/**
 * Note [MUSA Graph-safe RNG states]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Strategy:
 * ~~~~~~~~~
 * (It helps to look at
 * pytorch/aten/src/ATen/cuda/CUDAGeneratorImpl.h)
 *
 *
 * Usage:
 * ~~~~~~
 * PhiloxMusaState in this file, and unpack() in
 * musa/MUSAGraphsUtils.muh allow non-divergent use of
 * MUSAGeneratorImpl whether graph capture is underway or not.
 *
 * Each PhiloxMusaState instance should be used for one and only one
 * consumer kernel.
 *
 * Example (see e.g. native/musa/Dropout.mu):
 *
 * #include <torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h>
 * #include <torch_musa/csrc/aten/musa/MUSAGraphsUtils.cuh>
 *
 * __global__ void kernel(..., PhiloxMusaState philox_args) {
 *   auto seeds = at::musa::philox::unpack(philox_args);
 *   IndexType idx = blockIdx.x * blockDim.x + threadIdx.x;
 *   murand_state_philox4_32_10_t state;
 *   murand_init(std::get<0>(seeds), // seed
 *               idx,                // per-thread subsequence
 *               std::get<1>(seeds), // offset in subsequence
 *               &state);
 *   ...
 * }
 *
 * host_caller(...) {
 *   PhiloxMusaState rng_engine_inputs;
 *   {
 *     // See Note [Acquire lock when using random generators]
 *     std::lock_guard<std::mutex> lock(gen->mutex_);
 *
 *     // gen could be HostState or DevState here! No divergent code needed!
 *     rng_engine_inputs = gen->philox_musa_state(offset_increment);
 *   }
 *   kernel<<<...>>>(..., rng_engine_inputs);
 * }
 *
 */

struct MUSAGeneratorImpl : public c10::GeneratorImpl {
  // Constructors
  MUSAGeneratorImpl(DeviceIndex device_index = -1);
  ~MUSAGeneratorImpl() override = default;

  // MUSAGeneratorImpl methods
  std::shared_ptr<MUSAGeneratorImpl> clone() const;
  void set_current_seed(uint64_t seed) override;
  uint64_t current_seed() const override;
  uint64_t seed() override;
  void set_state(const c10::TensorImpl& new_state) override;
  c10::intrusive_ptr<c10::TensorImpl> get_state() const override;
  void set_offset(uint64_t offset) override;
  uint64_t get_offset() const override;
  void set_philox_offset_per_thread(uint64_t offset);
  uint64_t philox_offset_per_thread() const;
  void capture_prologue(int64_t* seed_extragraph, int64_t* offset_extragraph);
  uint64_t capture_epilogue();
  PhiloxMusaState philox_musa_state(uint64_t increment);

  bool reset_rnn_state() {
    return !no_reset_rnn_state_.test_and_set();
  }

  // Temporarily accommodates call sites that use philox_engine_inputs.
  // Allows incremental refactor of call sites to use philox_musa_state.
  std::pair<uint64_t, uint64_t> philox_engine_inputs(uint64_t increment);

  static DeviceType device_type();

 private:
  MUSAGeneratorImpl* clone_impl() const override;
  uint64_t seed_ = default_rng_seed_val;
  uint64_t philox_offset_per_thread_ = 0;
  int64_t* seed_extragraph_{};
  int64_t* offset_extragraph_{};
  uint32_t offset_intragraph_ = 0;
  bool graph_expects_this_gen_ = false;
  std::atomic_flag no_reset_rnn_state_;
};

namespace musa {
namespace detail {

const Generator& getDefaultMUSAGenerator(DeviceIndex device_index = -1);
Generator createMUSAGenerator(DeviceIndex device_index = -1);

} // namespace detail
} // namespace musa
} // namespace at
#endif // TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGENERATORIMPL_H_
