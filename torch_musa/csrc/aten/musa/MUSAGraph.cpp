#include <ATen/ATen.h>
#include <ATen/Functions.h>

#include <chrono>
#include <thread>

#include <iostream>
#include "torch_musa/csrc/aten/musa/MUSAGraph.h"
#include "torch_musa/csrc/core/MUSACachingAllocator.h"
namespace at {

struct Generator;
struct MUSAGeneratorImpl;
struct MUSAGeneratorState;

namespace musa {

constexpr int kSynchronizeBusyWaitMillis = 10;
static bool _musa_graphs_debug = false;

MempoolId_t graph_pool_handle() {
  // Sets just the second value, to distinguish it from MempoolId_ts created
  // from musaStreamGetCaptureInfo id_s in capture_begin.
  return c10::musa::MemPool::graph_pool_handle();
}

/**
 * Note [MUSA Graph Wrapper Class]
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * Q: Why do we need graph capture and launch bindings in Pytorch?
 *    Why can't they live in a user extension, for example?
 *
 * A1: Convenience.
 * A2: To ensure valid numerics on replay, some native MUSA ops (like RNG ops
 * with CPU statefulness) need cooperation from the capture and replay bindings
 *     (see Note [MUSA Graph-safe RNG states] in MUSAGeneratorImpl.h).
 *
 *     We can't expect users to know about this cooperation.  If users write
 * capture bindings naively in an extension, they likely won't interact with the
 * native ops properly.  Their graphs would yield invalid numerics on replay.
 */

/**
 * Note [Interaction with MUSA graph capture] in MUSACachingAllocator.cpp
 * describes memory management for captures.
 */

MUSAGraph::MUSAGraph(bool keep_graph)
    // may not be default-constructed.
    : capture_stream_(at::musa::getCurrentMUSAStream()),
      keep_graph_(keep_graph) {}

void MUSAGraph::register_generator_state(
    c10::intrusive_ptr<at::MUSAGeneratorState> state) {
  captured_generator_states_[std::move(state)] = 0;
}

void MUSAGraph::register_generator_state(const at::Generator& generator) {
  c10::intrusive_ptr<MUSAGeneratorImpl> musa_gen =
      dynamic_intrusive_pointer_cast<MUSAGeneratorImpl>(
          generator.getIntrusivePtr());
  musa_gen->register_graph(this);
}

void MUSAGraph::capture_begin(
    MempoolId_t pool /*=0*/,
    musaStreamCaptureMode capture_mode) {
  TORCH_CHECK(
      !has_graph_exec_,
      "This MUSAGraph instance already owns a captured graph. "
      "To capture a new graph, create a new instance.");

  // default generator is always registered
  auto* gen = get_generator_or_default<MUSAGeneratorImpl>(
      c10::nullopt, musa::detail::getDefaultMUSAGenerator());
  gen->register_graph(this);

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->capture_prologue();
  }

  auto stream = at::musa::getCurrentMUSAStream();

  TORCH_CHECK(
      stream != at::musa::getDefaultMUSAStream(),
      "MUSA graphs must be captured on a non-default stream. "
      "(However, after capture, it's ok to replay them on the "
      "default stream.)");

  capture_stream_ = stream;
  capture_dev_ = c10::musa::current_device();

  if (pool.first != 0 || pool.second != 0) {
    // Either value being nonzero means the user supplied a pool to share.
    // But only one should be nonzero.
    // If pool was created by another graph's capture_begin, first should be
    // nonzero. If pool was created by graph_pool_handle, second should be
    // nonzero.
    TORCH_INTERNAL_ASSERT(!(pool.first && pool.second));
    mempool_id_ = pool;
  } else {
    // User did not ask us to share a mempool. Create graph pool handle using
    // is_user_created=false. Sets just the first value, to distinguish it from
    // MempoolId_ts created by graph_pool_handle().
    mempool_id_ = c10::musa::MemPool::graph_pool_handle(false);
    TORCH_INTERNAL_ASSERT(mempool_id_.first > 0);
  }

  // Addendum: beginAllocateToPool is now called before
  // musaStreamBeginCapture to prevent an autograd thread's free() call
  // triggering an invalid musaEventRecord in the caching allocator due to the
  // capture status being updated _after_ a capture had already started.
  c10::musa::MUSACachingAllocator::beginAllocateToPool(
      // TODO(MUSA runtime): Confirm musaStreamGetCaptureInfo_v2's capture id
      // is stable and comparable with the id recorded after begin capture.
      capture_dev_,
      mempool_id_,
      [this](musaStream_t stream) {
        musaStreamCaptureStatus status{};
        CaptureId_t stream_capture_id = 0;
        AT_MUSA_CHECK(musaStreamGetCaptureInfo_v2(
            stream, &status, &stream_capture_id, nullptr, nullptr, nullptr));
        return status ==
            musaStreamCaptureStatus::musaStreamCaptureStatusActive &&
            stream_capture_id == capture_id_;
      });

  // musaStreamCaptureModeGlobal is the most conservative option to
  // prevent potentially unsafe MUSA API calls during capture.
  AT_MUSA_CHECK(musaStreamBeginCapture(capture_stream_, capture_mode));

  musaStreamCaptureStatus status{};
  AT_MUSA_CHECK(musaStreamGetCaptureInfo_v2(
      stream, &status, &capture_id_, nullptr, nullptr, nullptr));
  TORCH_INTERNAL_ASSERT(
      status == musaStreamCaptureStatus::musaStreamCaptureStatusActive);
}

void MUSAGraph::capture_end() {
  auto stream = at::musa::getCurrentMUSAStream();
  TORCH_CHECK(
      stream == capture_stream_,
      "Capture must end on the same stream it began on.");

  AT_MUSA_CHECK(musaStreamEndCapture(capture_stream_, &graph_));

  c10::musa::MUSACachingAllocator::endAllocateToPool(capture_dev_, mempool_id_);

  TORCH_CHECK(graph_ != nullptr, "Invalid capture.");

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    wholegraph_increments = generator_state->capture_epilogue();
  }

  size_t numMUSAGraphNodes = 0;
  AT_MUSA_CHECK(musaGraphGetNodes(graph_, nullptr, &numMUSAGraphNodes));
  if (numMUSAGraphNodes == 0) {
    TORCH_WARN(
        "The MUSA Graph is empty. This usually means that the graph was ",
        "attempted to be captured on wrong device or stream.");
  }

  capture_ended_ = true;
  has_graph_ = true;
  if (!keep_graph_) {
    instantiate();
    // TODO(MUSA runtime): Confirm whether musaGraphExec_t is independent from
    // musaGraph_t after instantiation. CUDA destroys graph_ here when debug is
    // off. Unlike the cuda implementation, we can't destroy the graph_ if
    // (!_musa_graphs_debug) {
    //   AT_CUDA_CHECK(musaGraphDestroy(graph_));
    // }
    // has_graph_ = false;
  }
}

void MUSAGraph::instantiate() {
  TORCH_CHECK(
      capture_ended_,
      "capture_end() must have been called before calling instantiate");

  if (has_graph_exec_) {
    TORCH_CHECK(
        keep_graph_,
        "instantiate() is intended to be called by the user only when keep_graph=true");
    AT_MUSA_CHECK(musaGraphExecDestroy(graph_exec_));
    graph_exec_ = nullptr;
  }

  // TODO(MUSA runtime): Check whether MUSA has an instantiate-with-flags API
  // equivalent to cudaGraphInstantiateFlagAutoFreeOnLaunch.
  AT_MUSA_CHECK(musaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;
}

void MUSAGraph::reinstantiate_graph() {
  // TODO(MUSA profiler/runtime): Confirm why profiling requires
  // re-instantiating graph_exec_ and whether this depends on keeping graph_
  // alive.
  TORCH_CHECK(
      capture_ended_ && graph_ != nullptr && has_graph_,
      "The graph_ musa has been created in the graph capture stage");
  if (has_graph_exec_) {
    // TODO(MUSA runtime): Confirm musaGraphExecDestroy releases resources
    // synchronously and does not require a device sync before graph memory is
    // released.
    C10_MUSA_CHECK_WARN(musaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
    graph_exec_ = nullptr;
  }
  // TODO(MUSA runtime): Check whether MUSA has an instantiate-with-flags API
  // equivalent to cudaGraphInstantiateFlagAutoFreeOnLaunch.
  AT_MUSA_CHECK(musaGraphInstantiate(&graph_exec_, graph_, NULL, NULL, 0));
  has_graph_exec_ = true;
}

void MUSAGraph::replay() {
  TORCH_CHECK(
      capture_ended_,
      "Called MUSAGraph::replay without a preceding successful capture.");

  if (!has_graph_exec_) {
    TORCH_INTERNAL_ASSERT(keep_graph_);
    instantiate();
  }

  c10::OptionalDeviceGuard device_guard{capture_stream_.device()};

  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->replay_prologue(wholegraph_increments);
  }

  // graph_exec_ may be replayed in any stream.
  AT_MUSA_CHECK(musaGraphLaunch(graph_exec_, at::musa::getCurrentMUSAStream()));
}

void MUSAGraph::enable_debug_mode() {
  _musa_graphs_debug = true;
}

void MUSAGraph::debug_dump(const std::string& debug_path) {
  if (_musa_graphs_debug || keep_graph_) {
    TORCH_WARN("DEBUG: calling debug_dump()");
    if (has_graph_) {
      TORCH_WARN("DEBUG: calling musaGraphDebugDotPrint() with ", debug_path);
      // TODO(MUSA runtime): Replace 1 << 10 with the official verbose flag if
      // MUSA exposes a cudaGraphDebugDotFlagsVerbose equivalent.
      C10_MUSA_CHECK_WARN(musaGraphDebugDotPrint(
          graph_, debug_path.c_str(), 1 << 10)); // most verbose output
      // TODO(MUSA runtime): If musaGraphExec_t depends on musaGraph_t, this
      // destroy is unsafe and debug_dump should keep graph_ alive.
      if (!keep_graph_) {
        AT_MUSA_CHECK(musaGraphDestroy(graph_));
        has_graph_ = false;
      }
    }
  } else {
    TORCH_WARN(
        "MUSA Graphs debug not enabled, set with [graph].enable_debug_mode()");
  }
}

musaGraph_t MUSAGraph::raw_musa_graph() {
  TORCH_CHECK(
      keep_graph_,
      "You cannot access the raw musaGraph_t instance unless MUSAGraph was initialized with keep_graph=true");
  TORCH_CHECK(
      has_graph_,
      "You cannot access the raw musaGraph_t instance until capture_end() has been called");
  return graph_;
}

musaGraphExec_t MUSAGraph::raw_musa_graph_exec() {
  TORCH_CHECK(
      has_graph_exec_,
      "You cannot access the raw musaGraphExec_t instance until instantiate() has been called");
  return graph_exec_;
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
  if (capture_ended_) {
    // notifyCaptureDestroy may throw. How should we handle this?
    // TODO(MUSA allocator/runtime): Confirm graph private pools should be
    // released at reset/destruction with the same ordering as CUDA.
    c10::musa::MUSACachingAllocator::releasePool(capture_dev_, mempool_id_);
    capture_ended_ = false;
  }
  if (has_graph_) {
    C10_MUSA_CHECK_WARN(musaGraphDestroy(graph_));
    has_graph_ = false;
    graph_ = nullptr;
  }
  if (has_graph_exec_) {
    C10_MUSA_CHECK_WARN(musaGraphExecDestroy(graph_exec_));
    has_graph_exec_ = false;
    graph_exec_ = nullptr;
  }
}

// Returns an id another graph's capture_begin can use to share the same memory
// pool as this graph.
MempoolId_t MUSAGraph::pool() {
  TORCH_CHECK(
      capture_ended_,
      "Called MUSAGraph::pool() without a preceding successful capture.");
  return mempool_id_;
}

MUSAGraph::~MUSAGraph() {
  for (auto& [generator_state, wholegraph_increments] :
       captured_generator_states_) {
    generator_state->unregister_graph(this);
  }
  reset();
}

} // namespace musa
} // namespace at
