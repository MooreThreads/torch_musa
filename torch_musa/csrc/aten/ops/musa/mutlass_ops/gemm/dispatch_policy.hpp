#pragma once

#include <mute/int_tuple.hpp>

namespace mutlass_kernels {

using namespace mute;

// Compile-time dispatch policy for quantized GEMM with TME warp-specialized pipeline.
//
// Template parameters:
//   TileShape_MNK_       : CTA tile shape, e.g. Shape<_128, _256, _128>
//   BlockNPerConsumer_    : N-dim per consumer warp squad (TileShape_N / NumMmaWarpSquads)
//   Stages_              : Number of pipeline stages (double-buffer = 2)
//   NumLoadWarpSquads_   : Number of producer warp squads (default 1)
//   NumMmaWarpSquads_    : Number of consumer warp squads (default 4)
template <
  class TileShape_MNK_,
  int   BlockNPerConsumer_,
  int   Stages_,
  int   NumLoadWarpSquads_ = 1,
  int   NumMmaWarpSquads_  = 4
>
struct QuantGemmTmeWarpSpecializedPolicy {
  using TileShape_MNK = TileShape_MNK_;

  static constexpr int BlockM = get<0>(TileShape_MNK{});
  static constexpr int BlockN = get<1>(TileShape_MNK{});
  static constexpr int BlockK = get<2>(TileShape_MNK{});

  static constexpr int BlockNPerConsumer    = BlockNPerConsumer_;
  static constexpr int Stages               = Stages_;
  static constexpr int NumLoadWarpSquads    = NumLoadWarpSquads_;
  static constexpr int NumMmaWarpSquads     = NumMmaWarpSquads_;
  static constexpr int ProducerPrefetchTiles = Stages + 2;
};

}  // namespace mutlass_kernels
