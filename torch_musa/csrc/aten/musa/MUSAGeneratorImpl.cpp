#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/musa/MUSAGraphsUtils.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"

namespace at {
namespace musa {
namespace detail {

namespace {

// Ensures we only call musaGetDeviceCount only once.
static c10::once_flag num_gpu_init_flag;

// Total number of gpus in the system.
static int64_t num_gpus;

// Ensures default_gens_musa is initialized once.
static std::deque<c10::once_flag> musa_gens_init_flag;

// Default, global MUSA generators, one per GPU.
static std::vector<Generator> default_gens_musa;

/*
 * Populates the global variables related to MUSA generators
 * Warning: this function must only be called once!
 */
static void initMUSAGenVector() {
  num_gpus = c10::musa::device_count();
  musa_gens_init_flag.resize(num_gpus);
  default_gens_musa.resize(num_gpus);
}

} // anonymous namespace

/**
 * PyTorch maintains a collection of default generators that get
 * initialized once. The purpose of these default generators is to
 * maintain a global running state of the pseudo random number generation,
 * when a user does not explicitly mention any generator.
 * getDefaultMUSAGenerator gets the default generator for a particular
 * musa device.
 */
const Generator& getDefaultMUSAGenerator(DeviceIndex device_index) {
  c10::call_once(num_gpu_init_flag, initMUSAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::musa::current_device();
  } else {
    TORCH_CHECK(idx >= 0 && idx < num_gpus);
  }
  c10::call_once(musa_gens_init_flag[idx], [&] {
    default_gens_musa[idx] = make_generator<MUSAGeneratorImpl>(idx);
    default_gens_musa[idx].seed();
  });
  return default_gens_musa[idx];
}

/**
 * Utility to create a MUSAGeneratorImpl. Returns a shared_ptr
 */
Generator createMUSAGenerator(DeviceIndex device_index) {
  c10::call_once(num_gpu_init_flag, initMUSAGenVector);
  DeviceIndex idx = device_index;
  if (idx == -1) {
    idx = c10::musa::current_device();
  }
  TORCH_CHECK(idx >= 0 && idx < num_gpus, "The device_index is invalid.");
  auto gen = make_generator<MUSAGeneratorImpl>(idx);
  auto musa_gen = check_generator<MUSAGeneratorImpl>(gen);
  musa_gen->set_current_seed(default_rng_seed_val);
  musa_gen->set_philox_offset_per_thread(0);
  return gen;
}

} // namespace detail
} // namespace musa

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_offset(uint64_t offset) {
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::set_offset");
  philox_offset_per_thread_ = offset;
  no_reset_rnn_state_.clear();
}

/**
 * Gets the current offset of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::get_offset() const {
  // Debatable if get_offset() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::get_offset");
  return philox_offset_per_thread_;
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * (It helps to look at pytorch/aten/src/ATen/cuda/CUDAGeneratorImpl.cpp)
 * The "Philox" constructor computes offset/4 (a uint64_t division) to locate
 * its internal start in its virtual bitstream viewed as 128-bit chunks, then,
 * when called in a thread, returns one 32-bit chunk at a time from that start
 * in the bitstream. In other words, if the incoming offset is not a multiple of
 * 4, each thread might repeat some previously-generated 32-bit values in the
 * bitstream. See https://github.com/pytorch/pytorch/pull/50169.
 */

/**
 * MUSAGeneratorImpl class implementation
 */
MUSAGeneratorImpl::MUSAGeneratorImpl(DeviceIndex device_index)
    : c10::GeneratorImpl{
          Device(at::musa::kMUSA, device_index),
          DispatchKeySet(at::musa::kMUSAKey)} {
  at::musa::assertNotCapturing("Cannot construct a new MUSAGeneratorImpl");
  no_reset_rnn_state_.clear();
}

/**
 * Sets the seed to be used by murandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_current_seed(uint64_t seed) {
  at::musa::assertNotCapturing(
      "Cannot call MUSAGeneratorImpl::set_current_seed");
  seed_ = seed;
  philox_offset_per_thread_ = 0;
  no_reset_rnn_state_.clear();
}

#define CAPTURE_DEFAULT_GENS_MSG                                                    \
  "In regions captured by MUSA graphs, you may only use the default MUSA RNG "      \
  "generator on the device that's current when capture begins. "                    \
  "If you need a non-default (user-supplied) generator, or a generator on another " \
  "device, please file an issue."

/**
 * Gets the current seed of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::current_seed() const {
  // Debatable if current_seed() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::current_seed");
  return seed_;
}

/**
 * Gets a nondeterministic random number from /dev/urandom or time,
 * seeds the CPUGeneratorImpl with it and then returns that number.
 *
 * FIXME: You can move this function to Generator.cpp if the algorithm
 * in getNonDeterministicRandom is unified for both CPU and MUSA
 */
uint64_t MUSAGeneratorImpl::seed() {
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::seed");
  auto random = c10::detail::getNonDeterministicRandom(true);
  this->set_current_seed(random);
  return random;
}

/**
 * Gets the current internal state of MUSAGeneratorImpl. The internal
 * state is returned as a CPU byte tensor.
 */
c10::intrusive_ptr<c10::TensorImpl> MUSAGeneratorImpl::get_state() const {
  // The RNG state comprises the seed, and an offset used for Philox.
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  auto state_tensor = at::detail::empty_cpu(
      {(int64_t)total_size},
      ScalarType::Byte,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt,
      c10::nullopt);
  auto rng_state = state_tensor.data_ptr<uint8_t>();
  auto current_seed = this->current_seed();
  auto offset = static_cast<int64_t>(
      this->philox_offset_per_thread()); // Note that old THCGeneratorState had
                                         // offset as std::atomic<int64_t>
  memcpy(rng_state, &current_seed, seed_size);
  memcpy(rng_state + seed_size, &offset, offset_size);

  return state_tensor.getIntrusivePtr();
}

/**
 * Sets the internal state of MUSAGeneratorImpl. The new internal state
 * must be a strided CPU byte tensor and have appropriate size. See
 * comments of MUSAGeneratorImpl::state for information about the layout
 * and size of the internal state.
 */
void MUSAGeneratorImpl::set_state(const c10::TensorImpl& new_state) {
  static const size_t seed_size = sizeof(uint64_t);
  static const size_t offset_size = sizeof(int64_t);
  static const size_t total_size = seed_size + offset_size;

  detail::check_rng_state(new_state);

  bool no_philox_seed = false;
  auto new_state_size = new_state.numel();
  if (new_state_size == total_size - offset_size) {
    no_philox_seed = true;
  } else {
    TORCH_CHECK(new_state_size == total_size, "RNG state is wrong size");
  }

  uint64_t input_seed;
  auto new_rng_state = new_state.data_dtype_initialized<uint8_t>();
  memcpy(&input_seed, new_rng_state, seed_size);
  this->set_current_seed(input_seed);
  int64_t philox_offset = 0;
  if (!no_philox_seed) {
    memcpy(&philox_offset, new_rng_state + seed_size, offset_size);
  }
  this->set_philox_offset_per_thread(static_cast<uint64_t>(philox_offset));
}

/**
 * Sets the philox_offset_per_thread_ to be used by murandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  at::musa::assertNotCapturing(
      "Cannot call MUSAGeneratorImpl::set_philox_offset_per_thread");
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::philox_offset_per_thread() const {
  at::musa::assertNotCapturing(
      "Cannot call MUSAGeneratorImpl::philox_offset_per_thread");
  return philox_offset_per_thread_;
}

/**
 * Called by MUSAGraph to prepare this instance for a graph capture region.
 * offset_extragraph is the initial offset at the start of the graphed region.
 * offset_intragraph tracks the offset in the graphed region.
 */
void MUSAGeneratorImpl::capture_prologue(
    int64_t* seed_extragraph,
    int64_t* offset_extragraph) {
  seed_extragraph_ = seed_extragraph;
  offset_extragraph_ = offset_extragraph;
  offset_intragraph_ = 0;
  graph_expects_this_gen_ = true;
}

/**
 * Called by MUSAGraph to finalize a graph capture region for this instance.
 */
uint64_t MUSAGeneratorImpl::capture_epilogue() {
  graph_expects_this_gen_ = false;
  return offset_intragraph_;
}

/**
 * Gets the seed and philox offset value to be used in
 * murandStatePhilox4_32_10, in an opaque PhiloxMusaState that's safe
 * and can be used non-divergently in callers whether MUSA graph
 * capture is underway or not.  See
 * Note [MUSA Graph-safe RNG states]
 *
 * Each kernel using philox has to sensibly increment offset
 * for future users of philox. So it gets the "old" value for
 * itself (before add), and tells subsequent users which offset
 * they should use, since only the kernel knows how many randoms
 * it intends to generate.
 *
 * Increment should be at least the number of murand() random numbers used in
 * each thread. It is the user's responsibility to make sure the increment
 * for philox is never smaller than the number of murand() calls. Increment
 * value > the number of murand() calls won't harm but anything less would mean
 * that you would be reusing random values from previous calls.
 *
 * See Note [Acquire lock when using random generators]
 */
PhiloxMusaState MUSAGeneratorImpl::philox_musa_state(uint64_t increment) {
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  if (at::musa::currentStreamCaptureStatus() != at::musa::CaptureStatus::None) {
    TORCH_CHECK(
        graph_expects_this_gen_,
        "philox_musa_state for an unexpected MUSA generator used during capture. " CAPTURE_DEFAULT_GENS_MSG);
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->offset_intragraph_ % 4 == 0);
    uint32_t offset = this->offset_intragraph_;
    TORCH_INTERNAL_ASSERT(
        this->offset_intragraph_ <=
        std::numeric_limits<uint32_t>::max() - increment);
    this->offset_intragraph_ += increment;
    return PhiloxMusaState(
        this->seed_extragraph_, this->offset_extragraph_, offset);
  } else {
    TORCH_CHECK(
        !graph_expects_this_gen_,
        "MUSA generator expects graph capture to be underway, "
        "but the current stream is not capturing.");
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
    uint64_t offset = this->philox_offset_per_thread_;
    this->philox_offset_per_thread_ += increment;
    return PhiloxMusaState(this->seed_, offset);
  }
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_musa_state.
 */
std::pair<uint64_t, uint64_t> MUSAGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::musa::assertNotCapturing(
      "Refactor this op to use MUSAGeneratorImpl::philox_musa_state. "
      "Cannot call MUSAGeneratorImpl::philox_engine_inputs");
  // rounds increment up to the nearest multiple of 4
  increment = ((increment + 3) / 4) * 4;
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_INTERNAL_ASSERT(this->philox_offset_per_thread_ % 4 == 0);
  uint64_t offset = this->philox_offset_per_thread_;
  this->philox_offset_per_thread_ += increment;
  return std::make_pair(this->seed_, offset);
}

/*
 * Gets the DeviceType of MUSAGeneratorImpl.
 * Used for type checking during run time.
 */
DeviceType MUSAGeneratorImpl::device_type() {
  return at::musa::kMUSA;
}

/**
 * Public clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
std::shared_ptr<MUSAGeneratorImpl> MUSAGeneratorImpl::clone() const {
  return std::shared_ptr<MUSAGeneratorImpl>(this->clone_impl());
}

/**
 * Private clone method implementation
 *
 * See Note [Acquire lock when using random generators]
 */
MUSAGeneratorImpl* MUSAGeneratorImpl::clone_impl() const {
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::clone_impl");
  auto gen = new MUSAGeneratorImpl(this->device().index());
  gen->set_current_seed(this->seed_);
  gen->set_philox_offset_per_thread(this->philox_offset_per_thread_);
  return gen;
}

at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
  return at::make_generator<MUSAGeneratorImpl>(id);
}

REGISTER_GENERATOR_PRIVATEUSE1(MakeGeneratorForPrivateuse1);

} // namespace at
