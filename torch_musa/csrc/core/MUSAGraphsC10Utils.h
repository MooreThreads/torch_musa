#ifndef TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
#define TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_

#include <utility>

// MUSA Graphs utils used by c10 and aten.
// aten/musa/MUSAGraphsUtils.cuh adds utils used by aten only.

namespace c10 {
namespace musa {

using CaptureId_t = unsigned long long;

// first is set if the instance is created by MUSAGraph::capture_begin.
// second is set if the instance is created by at::musa::graph_pool_handle.
using MempoolId_t = std::pair<CaptureId_t, CaptureId_t>;

} // namespace musa
} // namespace c10
#endif // TORCH_MUSA_CSRC_CORE_MUSAGRAPHSC10UTILS_H_
