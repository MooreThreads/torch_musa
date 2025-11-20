#include <ATen/Utils.h>
#include <ATen/core/GeneratorForPrivateuseone.h>
#include <ATen/ops/empty.h>
#include <c10/core/StreamGuard.h>
#include <c10/util/CallOnce.h>

#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/aten/musa/MUSAGraphsUtils.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/Device.h"
#include "torch_musa/csrc/core/MUSAFunctions.h"

#include <ATen/Functions.h>
#include <ATen/Tensor.h>
#include <deque>

namespace at {
namespace musa::detail {

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

} // namespace musa::detail

/**
 * Creates a clone of this MUSA Generator State.
 */
c10::intrusive_ptr<MUSAGeneratorState> MUSAGeneratorState::clone() {
  return make_intrusive<MUSAGeneratorState>(
      seed_, philox_offset_per_thread_, offset_intragraph_);
}

/**
 * Function to increase the internal offset based on the specified increment.
 */
void MUSAGeneratorState::increase(uint64_t increment) {
  // Rounds increment up to the nearest multiple of 4 to meet alignment
  // requirements.
  // see Note [Why enforce RNG offset % 4 == 0?]
  increment = ((increment + 3) / 4) * 4;
  // Handling different behaviors based on whether capturing is active.
  if (at::musa::currentStreamCaptureStatus() != at::musa::CaptureStatus::None) {
    // Ensures that the state is actually capturing.
    TORCH_CHECK(
        capturing_,
        "Attempt to increase offset for a MUSA generator not in capture mode.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ % 4 == 0, "RNG offset must be a multiple of 4.");
    // Ensures the increment does not cause overflow.
    TORCH_INTERNAL_ASSERT(
        offset_intragraph_ <= std::numeric_limits<uint32_t>::max() - increment,
        "Increment causes overflow in the offset value.");
    offset_intragraph_ += increment;
  } else {
    // Checks that the increment is expected outside graph capturing.
    TORCH_CHECK(
        !capturing_,
        "Offset increment outside graph capture encountered unexpectedly.");
    // Ensures the offset is a multiple of 4
    // see Note [Why enforce RNG offset % 4 == 0?]
    TORCH_INTERNAL_ASSERT(
        philox_offset_per_thread_ % 4 == 0,
        "RNG offset must be a multiple of 4.");
    philox_offset_per_thread_ += increment;
  }
}

/**
 * Registers this state to a MUSA graph to manage within the graph.
 */
void MUSAGeneratorState::register_graph(musa::MUSAGraph* graph) {
  // Ensures that the RNG state is not currently being captured.
  at::musa::assertNotCapturing(
      "Cannot register the state during capturing stage.");

  // If this is the first graph to be registered, allocate memory for the seed
  // and offset on the GPU.
  if (registered_graphs_.empty()) {
    auto options = at::TensorOptions().device(at::kMUSA).dtype(at::kLong);
    seed_extragraph_ = at::empty({1}, options);
    offset_extragraph_ = at::empty({1}, options);
  }

  // Insert the graph into the set of registered graphs if it's not already
  // registered.
  if (registered_graphs_.find(graph) == registered_graphs_.end()) {
    registered_graphs_.insert(graph);
  }
}

/**
 * Unregisters a MUSA graph from the RNG state.
 */
void MUSAGeneratorState::unregister_graph(musa::MUSAGraph* graph) {
  // Verify the graph was previously registered.
  TORCH_CHECK(
      registered_graphs_.find(graph) != registered_graphs_.end(),
      "The graph should be registered to the state");

  // Remove the graph from the set of registered graphs.
  registered_graphs_.erase(graph);

  // If no more graphs are registered, deallocate the GPU memory for the seed
  // and offset.
  if (registered_graphs_.empty()) {
    seed_extragraph_.reset();
    offset_extragraph_.reset();
  }
}

/**
 * Note [Explicit Registration of Generators to the MUSA Graph]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * Ideally, it would be more user-friendly if the state could be exchanged and
 * generators could be registered with the MUSA graph implicitly. However,
 * resetting GPU tensors during the capture stage causes these reset operations
 * to be recorded within the MUSA graph. This behavior is undesirable because we
 * do not want these tensors to be reset during the replay stage of the graph.
 *
 * As of now, there is no available method to perform a MUSA operation during
 * the graph's recording phase without having that operation be included in the
 * MUSA graph. This limitation necessitates explicit user action to register
 * generators with the graph. By requiring users to manually register their
 * generators, we can ensure that state resets (capture_prologue) only occur
 * before the graph capture begins, thus avoiding unintended resets during the
 * replay of the graph. See https://github.com/pytorch/pytorch/pull/114068.
 */

/**
 * Performs the prologue steps for capturing a MUSA graph state.
 * This method is intended to reset graph-related state variables before
 * capturing begins.
 */
void MUSAGeneratorState::capture_prologue() {
  capturing_ = true;
  offset_intragraph_ = 0;
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(0));
}

/**
 * Ends the capturing phase and resets related variables, returning the whole
 * graph increment.
 */
uint64_t MUSAGeneratorState::capture_epilogue() {
  capturing_ = false;
  return offset_intragraph_;
}

/**
 * Prepares the state for replay by setting initial state tensors and applying
 * total increment.
 */
void MUSAGeneratorState::replay_prologue(uint64_t wholegraph_increment) {
  // Ensures the generator is not in capturing mode.
  at::musa::assertNotCapturing(
      "Cannot prepare for replay during capturing stage.");
  seed_extragraph_.fill_(int64_t(seed_));
  offset_extragraph_.fill_(int64_t(philox_offset_per_thread_));
  // Applies the total increment achieved during previous captures to update the
  // offset.
  increase(wholegraph_increment);
}

/**
 * Note [Why enforce RNG offset % 4 == 0?]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Curand philox does allow offsets that aren't a multiple of 4.
 * But jit kernels don't use curand, they use a custom "Philox" class (see
 * torch/csrc/jit/tensorexpr/musa_random.h or
 * torch/csrc/jit/codegen/musa/runtime/random_numbers.cu).
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
  state_ = make_intrusive<MUSAGeneratorState>();
  no_reset_rnn_state_.clear();
}

MUSAGeneratorImpl::MUSAGeneratorImpl(
    DeviceIndex device_index,
    c10::intrusive_ptr<MUSAGeneratorState> state)
    : c10::
          GeneratorImpl{Device(at::musa::kMUSA, device_index), DispatchKeySet(at::musa::kMUSAKey)},
      state_(std::move(state)) {
  no_reset_rnn_state_.clear();
}

/**
 * Sets the seed to be used by curandStatePhilox4_32_10
 * Resets the philox_offset_per_thread_ to 0
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_current_seed(uint64_t seed) {
  at::musa::assertNotCapturing(
      "Cannot call MUSAGeneratorImpl::set_current_seed");
  state_->seed_ = seed;
  state_->philox_offset_per_thread_ = 0;
  no_reset_rnn_state_.clear();
}

/**
 * Sets the offset to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_offset(uint64_t offset) {
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::set_offset");
  // the set function checks if the offset is a multiple of 4.
  set_philox_offset_per_thread(offset);
  no_reset_rnn_state_.clear();
}

/**
 * Gets the current offset of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::get_offset() const {
  // Debatable if get_offset() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::get_offset");
  return state_->philox_offset_per_thread_;
}

/**
 * Gets the current seed of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::current_seed() const {
  // Debatable if current_seed() should be allowed in captured regions.
  // Conservatively disallow it for now.
  at::musa::assertNotCapturing("Cannot call MUSAGeneratorImpl::current_seed");
  return state_->seed_;
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
  at::musa::assertNotCapturing(
      "Please ensure to utilize the MUSAGeneratorImpl::set_state_index method during capturing.");
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

  uint64_t input_seed = 0;
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
 * Sets the generator's current state to
 * This function allows switching between different registered states of
 * the generator.
 */
void MUSAGeneratorImpl::graphsafe_set_state(
    const c10::intrusive_ptr<GeneratorImpl>& gen) {
  c10::intrusive_ptr<MUSAGeneratorImpl> musa_gen =
      dynamic_intrusive_pointer_cast<MUSAGeneratorImpl>(gen);
  TORCH_CHECK(musa_gen, "Expected a MUSA Generator");
  state_ = musa_gen->state_;
}

/**
 * Get the GeneratorImpl that point to current state_
 */
c10::intrusive_ptr<c10::GeneratorImpl> MUSAGeneratorImpl::graphsafe_get_state()
    const {
  auto gen = make_intrusive<MUSAGeneratorImpl>(device().index(), state_);
  return gen;
}

/**
 * Sets the philox_offset_per_thread_ to be used by curandStatePhilox4_32_10
 *
 * See Note [Acquire lock when using random generators]
 */
void MUSAGeneratorImpl::set_philox_offset_per_thread(uint64_t offset) {
  // see Note [Why enforce RNG offset % 4 == 0?]
  TORCH_CHECK(offset % 4 == 0, "offset must be a multiple of 4");
  state_->philox_offset_per_thread_ = offset;
}

/**
 * Gets the current philox_offset_per_thread_ of MUSAGeneratorImpl.
 */
uint64_t MUSAGeneratorImpl::philox_offset_per_thread() const {
  return state_->philox_offset_per_thread_;
}

/**
 * Registers this state to a MUSA graph to manage within the graph.
 */
void MUSAGeneratorImpl::register_graph(musa::MUSAGraph* graph) {
  graph->register_generator_state(state_);
  state_->register_graph(graph);
}

/**
 * Unregisters a MUSA graph from the RNG state.
 */
void MUSAGeneratorImpl::unregister_graph(musa::MUSAGraph* graph) {
  state_->unregister_graph(graph);
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
  if (at::musa::currentStreamCaptureStatus() != at::musa::CaptureStatus::None) {
    uint32_t offset = state_->offset_intragraph_;
    state_->increase(increment);
    return PhiloxMusaState(
        state_->seed_extragraph_.data_ptr<int64_t>(),
        state_->offset_extragraph_.data_ptr<int64_t>(),
        offset);
  } else {
    uint64_t offset = state_->philox_offset_per_thread_;
    state_->increase(increment);
    return PhiloxMusaState(state_->seed_, offset);
  }
}

/**
 * Temporarily accommodates call sites that use philox_engine_inputs.
 * Allows incremental refactor of call sites to use philox_musa_state.
 */
std::pair<uint64_t, uint64_t> MUSAGeneratorImpl::philox_engine_inputs(
    uint64_t increment) {
  at::musa::assertNotCapturing(
      "Refactor this op to use MUSAGeneratorImpl::philox_musa_state. Cannot call MUSAGeneratorImpl::philox_engine_inputs");
  uint64_t offset = state_->philox_offset_per_thread_;
  state_->increase(increment);
  return std::make_pair(state_->seed_, offset);
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
  auto gen = new MUSAGeneratorImpl(this->device().index(), state_->clone());
  return gen;
}

at::Generator MakeGeneratorForPrivateuse1(c10::DeviceIndex id) {
  return at::make_generator<MUSAGeneratorImpl>(id);
}

} // namespace at