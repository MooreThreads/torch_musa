#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <chrono>
#include <thread>

#include <iostream>
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
namespace at {
namespace musa {

constexpr int kSynchronizeBusyWaitMillis = 10;
static bool _musa_graphs_debug = false;

MempoolId_t graph_pool_handle() {
  // uuid count starts at 1. 0 is reserved to mean "wasn't set by
  // graph_pool_handle".
  static std::atomic<CaptureId_t> uid{1};
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // from cudaStreamGetCaptureInfo id_s in capture_begin.
  return {0, uid++};
}

// Get the expected id of a capture sequence so that we can call
// beginAllocateStreamToPool before starting a graph capture
CaptureId_t capture_sequence_id() {
  static std::atomic<CaptureId_t> uuid{1};
  return uuid++;
}

/**
 * MUSAGraph, a.k.a the wrapper class of musaGraph_t in MUSA
 */

std::atomic<int> MUSAGraph::pending_event_queries = 0;

// Track any outstanding event queries that could happen e.g., in a NCCL
// watchdog so that they can be resolved before the capture begins. Note that
// event queries are not allowed during a graph capture in the default capture
// mode.
void MUSAGraph::inc_pending_event_queries() {
  pending_event_queries++;
}

void MUSAGraph::dec_pending_event_queries() {
  TORCH_INTERNAL_ASSERT(
      pending_event_queries > 0,
      "Attempted to decrement the number of outstanding events to be queried, but it was <= 0.");
  pending_event_queries--;
}

int MUSAGraph::num_pending_event_queries() {
  return pending_event_queries;
}

MUSAGraph::MUSAGraph()
    // may not be default-constructed.
    : capture_stream_(at::musa::getCurrentMUSAStream()) {}

void MUSAGraph::capture_begin(
    MempoolId_t pool /*=0*/,
    musaStreamCaptureMode capture_mode) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This MUSAGraph instance already owns a captured graph. "
      "To capture a new graph, create a new instance.");

  // For now, a MUSAGraph instance only accommodates the default generator on
  // the device that's current when capture begins. If any op in the captured
  // region uses a non-default generator, or a generator on another device, the
  // offending generator will throw an error. These restrictions simplify
  // MUSAGraph, but could be relaxed in the future: in principle, the underlying
  // Cuda calls do permit cross-device ops to be captured.
  auto* gen = get_generator_or_default<MUSAGeneratorImpl>(
      c10::nullopt, musa::detail::getDefaultMUSAGenerator());

  auto options = TensorOptions().device(at::kMUSA).dtype(at::kLong);
  seed_extragraph_ = at::empty({1}, options);
  offset_extragraph_ = at::empty({1}, options);

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  gen->capture_prologue(
      seed_extragraph_.data_ptr<int64_t>(),
      offset_extragraph_.mutable_data_ptr<int64_t>());

  auto stream = at::musa::getCurrentMUSAStream();

  TORCH_CHECK(
      stream != at::musa::getDefaultMUSAStream(),
      "MUSA graphs must be captured on a non-default stream. "
      "(However, after capture, it's ok to replay them on the "
      "default stream.)");

  capture_stream_ = stream;
  capture_gen_ = gen;
  capture_dev_ = c10::musa::current_device();

  id_ = capture_sequence_id();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Use our own id_ as our
    // mempool_id_. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = {id_, 0};
  }

  // Addendum: beginAllocateStreamToPool is now called before
  // musaStreamBeginCapture to prevent an autograd thread's free() call
  // triggering an invalid cudaEventRecord in the caching allocator due to the
  // capture status being updated _after_ a capture had already started.
  c10::musa::MUSACachingAllocator::beginAllocateStreamToPool(
      capture_dev_, capture_stream_, mempool_id_);

  // At this point, any NCCL watchdogs should be aware that we are in capture
  // mode and therefore should not enqueue any additional work that could be
  // event-queried. We still must wait on any existing work that has not been
  // cleaned up.
  while (num_pending_event_queries()) {
    TORCH_WARN_ONCE(
        "Waiting for pending NCCL work to finish before starting graph capture.");
    std::this_thread::sleep_for(
        std::chrono::milliseconds(kSynchronizeBusyWaitMillis));
  }

  // cudaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe MUSA API calls during capture.  See
  // https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html#group__CUDART__STREAM_1g9d0535d93a214cbf126835257b16ba85
  AT_MUSA_CHECK(musaStreamBeginCapture(capture_stream_, capture_mode));

  // musaStreamCaptureStatus status;
  // AT_MUSA_CHECK(musaStreamGetCaptureInfo(stream, &status, nullptr));
  // TORCH_INTERNAL_ASSERT(
  //     status == musaStreamCaptureStatus::musaStreamCaptureStatusActive);

  TORCH_INTERNAL_ASSERT(id_ > 0);
}

void MUSAGraph::capture_end() {
  auto stream = at::musa::getCurrentMUSAStream();
  TORCH_CHECK(
      stream == capture_stream_,
      "Capture must end on the same stream it began on.");

  AT_MUSA_CHECK(musaStreamEndCapture(capture_stream_, &graph_));

  c10::musa::MUSACachingAllocator::endAllocateStreamToPool(
      capture_dev_, capture_stream_);

  TORCH_CHECK(graph_ != NULL, "Invalid capture.");
  has_graph_ = true;

  AT_MUSA_CHECK(musaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;

  auto* gen = get_generator_or_default<MUSAGeneratorImpl>(
      c10::nullopt, musa::detail::getDefaultMUSAGenerator());
  TORCH_CHECK(
      gen == capture_gen_,
      "Default MUSA RNG generator on current device at capture end "
      "is different from default generator on current device "
      "when capture began");
  wholegraph_increment_ = gen->capture_epilogue();

  size_t numMUSAGraphNodes = 0;
  AT_MUSA_CHECK(musaGraphGetNodes(graph_, NULL, &numMUSAGraphNodes));
  if (numMUSAGraphNodes == 0) {
    TORCH_WARN(
        "The MUSA Graph is empty. This usually means that the graph was ",
        "attempted to be captured on wrong device or stream.");
  }

  // Unlike the cuda implementation, we can't destroy the graph_
  // check if debug path is set
  // if (!_musa_graphs_debug) {
  //   // Now that we've instantiated graph_ into graph_exec_,
  //   // we don't need graph_ anymore.
  //   // AT_MUSA_CHECK(musaGraphDestroy(graph_));
  //   // has_graph_ = false;
  // } else {
  //   TORCH_WARN(
  //       "DEBUG: TORCH_CUDAGRAPHS_DEBUG_PATH detected. graph_ will not be
  //       freed until debug_dump is called.");
  // }
}

void MUSAGraph::replay() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called CUDAGraph::replay without a preceding successful capture.");
  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  // Just like any RNG consumer kernel!
  auto* gen = get_generator_or_default<MUSAGeneratorImpl>(
      c10::nullopt, musa::detail::getDefaultMUSAGenerator());
  PhiloxMusaState rng_engine_inputs;
  {
    std::lock_guard<std::mutex> lock(gen->mutex_);
    rng_engine_inputs = gen->philox_musa_state(wholegraph_increment_);
  }

  seed_extragraph_.fill_(int64_t(gen->current_seed()));
  offset_extragraph_.fill_(int64_t(rng_engine_inputs.offset_.val));

  // graph_exec_ may be replayed in any stream.
  AT_MUSA_CHECK(musaGraphLaunch(graph_exec_, at::musa::getCurrentMUSAStream()));
  // AT_MUSA_CHECK(musaDeviceSynchronize());
}

void MUSAGraph::enable_debug_mode() {
  _musa_graphs_debug = true;
}

void MUSAGraph::debug_dump(const std::string& debug_path) {
  if (_musa_graphs_debug) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling musaGraphDebugDotPrint() with ", debug_path);
      C10_MUSA_CHECK_WARN(musaGraphDebugDotPrint(
          graph_, debug_path.c_str(), 1 << 10)); // most verbose output
      AT_MUSA_CHECK(musaGraphDestroy(graph_));
    }
  } else {
    TORCH_WARN(
        "MUSA Graphs debug not enabled, set with torch._C._cuda_enable_graphs_debug_mode");
  }
}

void MUSAGraph::reset() {
  // I'd prefer these checks throw exceptions, not print warnings,
  // but the destructor calls reset(), and at least one CI build
  // refuses to compile with a throwing destructor.
  //
  // Instead of calling reset() in the destructor to clean up, I could
  // call reset() in the __del__ method of a thin Python wrapper,
  // in which case reset would be allowed to throw exceptions.
  // But Stackoverflow does not like user-defined __del__.
  // __del__ prevents Graph instances from EVER being garbage collected
  // if they participate in a reference cycle.
  // And exceptions thrown in __del__ only print a warning anyway.
  //
  // Calling reset() in the C++ destructor, with warnings instead of exceptions
  // if calls fail, is the compromise we chose.
  //
  // If capture_begin, the capture, or capture_end failed at some point, this
  // MUSAGraph, the generator, and the allocator could end up in all kinds of
  // weird states depending where failure occurred. If the user catches the
  // failure exception in a script, or is running in REPL or (god forbid) a
  // Jupyter notebook, I don't see an easy way for reset() to gracefully fix all
  // such possible error states.
  if (has_graph_ || has_graph_exec_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    c10::musa::MUSACachingAllocator::releasePool(capture_dev_, mempool_id_);
  }
  if (has_graph_) {
    C10_MUSA_CHECK_WARN(musaGraphDestroy(graph_));
    has_graph_ = false;
  }
  if (has_graph_exec_) {
    C10_MUSA_CHECK_WARN(musaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory
// pool as this graph.
MempoolId_t MUSAGraph::pool() {
  TORCH_CHECK(
      has_graph_exec_,
      "Called MUSAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

MUSAGraph::~MUSAGraph() {
  reset();
}

} // namespace musa
} // namespace at
