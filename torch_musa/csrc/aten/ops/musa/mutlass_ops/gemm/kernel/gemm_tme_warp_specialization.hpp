#pragma once

#include <cstdint>

#include <mutlass/mutlass.h>
#include <mute/tensor.hpp>

namespace mutlass_kernels::kernel {

using namespace mute;

// Quantized GEMM kernel with TME warp-specialized pipeline.
//
// Orchestrates producer (TME load) and consumer (MMA + epilogue) warp squads.
// Agnostic to quantization scheme -- all quant logic lives in CollectiveMainloop.
//
// Template parameters:
//   CollectiveMainloop_  : e.g. MainloopW4A16TmeWs<...> MainloopW4A16TmePingPong(TODO)...
//   CollectiveEpilogue_  : e.g. EpilogueQuantSimple<...> EpilogueQuantBiasAct<...>(TODO)...
template <class CollectiveMainloop_, class CollectiveEpilogue_>
struct QuantGemmTmeWarpSpecialized {

  // ===== Type extraction from sub-layers =====
  using CollectiveMainloop = CollectiveMainloop_;
  using CollectiveEpilogue = CollectiveEpilogue_;

  using DispatchPolicy  = typename CollectiveMainloop::DispatchPolicy;
  using SharedStorage   = typename CollectiveMainloop::SharedStorage;
  using BarrierStorage  = typename CollectiveMainloop::BarrierStorage;

  using PipelineA          = typename CollectiveMainloop::PipelineA;
  using PipelineBTme       = typename CollectiveMainloop::PipelineBTme;
  using PipelineAParams    = typename CollectiveMainloop::PipelineAParams;
  using PipelineAState     = typename CollectiveMainloop::PipelineAState;
  using PipelineBTmeParams = typename CollectiveMainloop::PipelineBTmeParams;
  using PipelineBTmeState  = typename CollectiveMainloop::PipelineBTmeState;

  using TiledMma = typename CollectiveMainloop::TiledMma;

  static constexpr int BlockM            = DispatchPolicy::BlockM;
  static constexpr int BlockN            = DispatchPolicy::BlockN;
  static constexpr int BlockK            = DispatchPolicy::BlockK;
  static constexpr int BlockNPerConsumer = DispatchPolicy::BlockNPerConsumer;
  static constexpr int Stages            = DispatchPolicy::Stages;
  static constexpr int NumLoadWarpSquads = DispatchPolicy::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = DispatchPolicy::NumMmaWarpSquads;
  static constexpr int ProducerPrefetchTiles = DispatchPolicy::ProducerPrefetchTiles;

  static constexpr int      SharedStorageSize  = sizeof(SharedStorage);
  static constexpr int      SmemAlignmentBytes = CollectiveMainloop::SmemAlignmentBytes;
  static constexpr uint32_t MaxThreadsPerBlock =
      (NumLoadWarpSquads + NumMmaWarpSquads) * mutlass::NumThreadsPerWarpSquad;
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;

  // ===== Linear K-tile scheduler (trivial, kept inline), TODO: PersistentScheduler =====
  struct LinearKTileScheduler {
    int current_k_tile;
    int end_k_tile;
    MUTLASS_DEVICE explicit LinearKTileScheduler(int k_tiles)
        : current_k_tile(0), end_k_tile(k_tiles) {}
    MUTLASS_DEVICE bool fetch_next_task(int& k_tile_idx) {
      if (current_k_tile >= end_k_tile) return false;
      k_tile_idx = current_k_tile++;
      return true;
    }
  };

  // ===== Aggregated arguments (host side) =====
  struct Arguments {
    typename CollectiveMainloop::Arguments mainloop{};
    typename CollectiveEpilogue::Arguments epilogue{};
  };

  // ===== Aggregated params (device side) =====
  struct Params {
    typename CollectiveMainloop::Params mainloop;
    typename CollectiveEpilogue::Params epilogue;
  };

  static Params to_underlying_arguments(Arguments const& args) {
    return {
      CollectiveMainloop::to_underlying_arguments(args.mainloop),
      CollectiveEpilogue::to_underlying_arguments(args.epilogue)
    };
  }

  static dim3 get_grid_shape(Arguments const& args) {
    int m_tiles = (args.mainloop.m + BlockM - 1) / BlockM;
    int n_tiles = (args.mainloop.n + BlockN - 1) / BlockN;
    return dim3(n_tiles, m_tiles);
  }

  static dim3 get_block_shape() {
    return dim3(MaxThreadsPerBlock);
  }

  // ===== Kernel entry point =====
  MUTLASS_DEVICE
  void operator()(Params const& params, char* smem) {
    // ---------- 1. Role assignment ----------
    enum class WarpSquadRole {
      Producer  = 0,
      Consumer0 = 1,
      Consumer1 = 2,
      Consumer2 = 3,
      Consumer3 = 4,
    };

    int thread_idx               = threadIdx.x;
    int warp_idx                 = mutlass::canonical_warp_idx_sync();
    int warp_squad_idx           = mutlass::canonical_warp_squad_idx();
    int thread_idx_in_warp_squad = thread_idx % mutlass::NumThreadsPerWarpSquad;
    int warp_idx_in_warp_squad   = warp_idx % mutlass::NumWarpsPerWarpSquad;
    int consumer_warp_squad_idx  = warp_squad_idx - static_cast<int>(WarpSquadRole::Consumer0);

    if (warp_squad_idx >= NumLoadWarpSquads + NumMmaWarpSquads) return;
    auto warp_squad_role = static_cast<WarpSquadRole>(warp_squad_idx);

    // ---------- 2. Tile coordinates and bounds ----------
    int32_t m = params.mainloop.m;
    int32_t n = params.mainloop.n;
    int32_t k = params.mainloop.k;
    const int tile_m_idx = blockIdx.y;
    const int tile_n_idx = blockIdx.x;
    const int m_begin    = tile_m_idx * BlockM;
    const int n_begin    = tile_n_idx * BlockN;
    const int k_tiles    = (k + BlockK - 1) / BlockK;
    if (m_begin >= m || n_begin >= n || k_tiles <= 0) return;

    // ---------- 3. Barrier allocation and pipeline init ----------
    mutlass::arch::allocate_async_barriers(sizeof(BarrierStorage));
    BarrierStorage* barrier_storage = reinterpret_cast<BarrierStorage*>(0);

    PipelineAParams pipeline_params_a;
    pipeline_params_a.transaction_bytes = CollectiveMainloop::TmeTransactionBytesA;
    pipeline_params_a.num_consumers     = NumMmaWarpSquads * mutlass::NumWarpsPerWarpSquad;
    pipeline_params_a.num_producers     = 1;
    PipelineA      pipeline_a(pipeline_params_a,
                              reinterpret_cast<uint64_t>(&barrier_storage->PipelineA));
    PipelineAState pipeline_a_producer_state = mutlass::make_producer_start_state<PipelineA>();
    PipelineAState pipeline_a_consumer_state;

    PipelineBTmeParams pipeline_params_btme;
    pipeline_params_btme.transaction_bytes = CollectiveMainloop::TmeTransactionBytesB4
                                           + CollectiveMainloop::TmeTransactionBytesScale;
    pipeline_params_btme.num_consumers     = NumMmaWarpSquads * mutlass::NumWarpsPerWarpSquad;
    pipeline_params_btme.num_producers     = 1;
    PipelineBTme      pipeline_btme(pipeline_params_btme,
                                    reinterpret_cast<uint64_t>(&barrier_storage->PipelineBTme));
    PipelineBTmeState pipeline_btme_producer_state = mutlass::make_producer_start_state<PipelineBTme>();
    PipelineBTmeState pipeline_btme_consumer_state;

    __syncthreads();

    // ---------- 4. Mainloop and epilogue instances ----------
    CollectiveMainloop mainloop;
    CollectiveEpilogue epilogue;

    // ---------- 5. Producer branch ----------
    if (warp_squad_role == WarpSquadRole::Producer) {
      LinearKTileScheduler scheduler(k_tiles);
      int next_k_tile;
      int prefetched = 0;

      // Prefetch phase: fill pipeline
      while (prefetched < ProducerPrefetchTiles &&
             scheduler.fetch_next_task(next_k_tile)) {
        mainloop.load(params.mainloop, smem,
                      pipeline_a, pipeline_btme,
                      pipeline_a_producer_state, pipeline_btme_producer_state,
                      next_k_tile, tile_m_idx, tile_n_idx,
                      warp_idx_in_warp_squad);
        ++prefetched;
      }
      // Steady-state phase
      while (scheduler.fetch_next_task(next_k_tile)) {
        mainloop.load(params.mainloop, smem,
                      pipeline_a, pipeline_btme,
                      pipeline_a_producer_state, pipeline_btme_producer_state,
                      next_k_tile, tile_m_idx, tile_n_idx,
                      warp_idx_in_warp_squad);
      }
    }
    // ---------- 6. Consumer branch ----------
    else if (warp_squad_role == WarpSquadRole::Consumer0 ||
             warp_squad_role == WarpSquadRole::Consumer1 ||
             warp_squad_role == WarpSquadRole::Consumer2 ||
             warp_squad_role == WarpSquadRole::Consumer3) {

      TiledMma tiled_mma;
      auto accum = partition_fragment_C(tiled_mma,
                       make_shape(Int<BlockM>{}, Int<BlockNPerConsumer>{}));
      clear(accum);

      // MMA + dequant over all K-tiles
      mainloop.mma(params.mainloop, smem,
                   pipeline_a, pipeline_btme,
                   pipeline_a_consumer_state, pipeline_btme_consumer_state,
                   accum, k_tiles,
                   consumer_warp_squad_idx,
                   thread_idx_in_warp_squad);

      // Epilogue: cast + store
      epilogue.store(params.epilogue, accum, tiled_mma,
                     thread_idx_in_warp_squad,
                     consumer_warp_squad_idx,
                     tile_m_idx, tile_n_idx,
                     NumMmaWarpSquads);
    }
  }
};

}  // namespace mutlass_kernels::kernel
