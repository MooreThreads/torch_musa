#ifndef TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGRAPH_H_
#define TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGRAPH_H_

#include <ATen/Tensor.h>
#include "torch_musa/csrc/aten/musa/MUSAGeneratorImpl.h"
#include "torch_musa/csrc/core/MUSAGraphsC10Utils.h"

#include "torch_musa/csrc/aten/musa/Exceptions.h"

namespace at {
namespace musa {

/**
 * TODO: (lijing) Maybe we could just use the ported version of MISAGraph.h, But
 * a lot of modification would be made to the musa_porting tool. And It's not
 * easy to change the MUSA version macro.
 */

// Standalone way to get a unique mempool id usable as a pool=... argument
// to MUSAGraph::capture_begin
C10_EXPORT MempoolId_t graph_pool_handle();

struct C10_EXPORT MUSAGraph {
  MUSAGraph();
  ~MUSAGraph();

  static void inc_pending_event_queries();
  static void dec_pending_event_queries();
  static int num_pending_event_queries();
  void capture_begin(
      MempoolId_t pool = {0, 0},
      musaStreamCaptureMode capture_mode = musaStreamCaptureModeGlobal);
  void capture_end();
  void replay();
  void reset();
  MempoolId_t pool();
  void enable_debug_mode();
  void debug_dump(const std::string& debug_path);

 protected:
  musaGraph_t graph_ = NULL;
  musaGraphExec_t graph_exec_ = NULL;

  static std::atomic<int> pending_event_queries;

  // internal states so reset() can do its best cleaning up
  // Set to true in capture_end if musaStreamEndCapture succeeded
  // Set back to false soon after, when graph_ is consumed by
  // MUSAGraphInstantiate to create graph_exec_, then graph_ is deleted
  bool has_graph_ = false;
  // Set to true in capture_end if MUSAGraphInstantiate succeeded
  bool has_graph_exec_ = false;

  // uuid of this instance's current capture, retrieved from musa
  CaptureId_t id_;

  // uuid used to request a particular private mempool from
  // musaCachingAllocator. By default, this will be set to {id_, 0}.
  //
  // If capture_begin is called with "pool=other_graph.pool()", this graph's
  // mempool_id_ will be set to the other graph's mempool_id_, and therefore
  // share a mempool with the other graph.
  //
  // If capture_begin is called with "pool=handle" where "handle" came from
  // graph_pool_handle(), it will share a mempool with any other captures that
  // used "pool=handle".
  //
  // Sharing a mempool across graphs saves memory, and it's safe if you
  // know you'll replay those graphs in the same order you captured them.
  MempoolId_t mempool_id_;

  // Stream on which capture began
  at::musa::MUSAStream capture_stream_;

  // Default generator on device where capture began
  at::MUSAGeneratorImpl* capture_gen_;

  // Device where capture occurred. Right now, for simplicity, we require all
  // ops in a capture to run on the same device, but this is a limitation of
  // MUSAGraph, not MUSA itself.  We can straightforwardly modify MUSAGraph to
  // support multi-device captures if needed.
  int capture_dev_;

  // RNG state trackers
  at::Tensor seed_extragraph_;
  at::Tensor offset_extragraph_;
  uint64_t wholegraph_increment_;
};

} // namespace musa

} // namespace at

#endif // TORCH_MUSA_CSRC_ATEN_MUSA_MUSAGRAPH_H_
