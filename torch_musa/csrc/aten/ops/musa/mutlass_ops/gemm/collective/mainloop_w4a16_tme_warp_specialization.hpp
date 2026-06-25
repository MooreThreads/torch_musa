#pragma once

#include <cstdint>

#include <mutlass/mutlass.h>
#include <mutlass/numeric_conversion.h>
#include <mutlass/gemm/collective/collective_builder.hpp>
#include <mute/int_tuple.hpp>
#include <mute/tensor.hpp>

namespace mutlass_kernels::collective {

using namespace mute;

// W4A16 Mainloop: A(fp16) x B(int4 packed as uint8) with per-group fp32 scale.
//
// Implements the producer (TME load) and consumer (MMA + dequant) sides of
// the warp-specialized pipeline.
//
// Template parameters:
//   DispatchPolicy_      : QuantGemmTmeWarpSpecializedPolicy<...>
//   ElementA_            : A element type (mutlass::half_t)
//   StrideA_             : A stride type
//   ElementB_            : B logical element type (mutlass::int4b_t)
//   ElementScale_        : Scale element type (float)
//   ElementAccumulator_  : Accumulator type (float)
template <
  class DispatchPolicy_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class ElementScale_,
  class ElementAccumulator_
>
struct MainloopW4A16TmeWs {

  // ===== Policy and element types =====
  using DispatchPolicy     = DispatchPolicy_;
  using ElementA           = ElementA_;
  using ElementB           = ElementB_;
  using ElementScale       = ElementScale_;
  using ElementAccumulator = ElementAccumulator_;
  using StrideA            = StrideA_;

  // ===== Tile constants from policy =====
  static constexpr int BlockM            = DispatchPolicy::BlockM;
  static constexpr int BlockN            = DispatchPolicy::BlockN;
  static constexpr int BlockK            = DispatchPolicy::BlockK;
  static constexpr int BlockNPerConsumer = DispatchPolicy::BlockNPerConsumer;
  static constexpr int BlockKBytes       = BlockK / 2;
  static constexpr int Stages            = DispatchPolicy::Stages;
  static constexpr int NumLoadWarpSquads = DispatchPolicy::NumLoadWarpSquads;
  static constexpr int NumMmaWarpSquads  = DispatchPolicy::NumMmaWarpSquads;

  // ===== MMA instruction selection =====
  using ConsumerTileShape = Shape<Int<BlockM>, Int<BlockNPerConsumer>, Int<BlockK>>;

  using TiledMma = decltype(make_tiled_mma(
      MP31::SQMMA::ss_op_selector<ElementA,
                                  ElementB,
                                  ElementAccumulator,
                                  ConsumerTileShape,
                                  TCE::Major::K,
                                  TCE::Major::K,
                                  Int<BlockM>>(),
      Layout<Shape<_1, _1, _1>>{}));

  // ===== Shared memory layout derivation =====
  using SmemLayoutAtomA = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_A<TCE::Major::K,
                                                          ElementA,
                                                          typename TiledMma::Atom::MMA_Op,
                                                          ConsumerTileShape>());
  using SmemLayoutAtomB = decltype(mutlass::gemm::collective::detail::
                                       ss_smem_selector_B<TCE::Major::K,
                                                          ElementB,
                                                          typename TiledMma::Atom::MMA_Op,
                                                          ConsumerTileShape>());

  using SmemLayoutA     = decltype(tile_to_shape(SmemLayoutAtomA{},
                            Shape<Int<BlockM>, Int<BlockK>, Int<Stages>>{}));
  using SmemLayoutB     = decltype(tile_to_shape(SmemLayoutAtomB{},
                            Shape<Int<BlockNPerConsumer>, Int<BlockK>, Int<Stages>>{}));
  using SmemLayoutBFull = decltype(tile_to_shape(SmemLayoutAtomB{},
                            Shape<Int<BlockN>, Int<BlockK>, Int<Stages>>{}));

  using SmemLayoutBU8     = decltype(recast_layout<ElementB, uint8_t>(SmemLayoutB{}));
  using SmemLayoutBFullU8 = decltype(recast_layout<ElementB, uint8_t>(SmemLayoutBFull{}));

  using SmemLayoutScaleTme =
      Layout<Shape<Int<BlockN>, _1, Int<Stages>>, Stride<_1, _0, Int<BlockN>>>;

  // ===== TME copy descriptor types =====
  using TME_A = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementA const*>(nullptr)),
                  repeat_like(StrideA{}, int32_t(0)), StrideA{}),
      take<0, 2>(SmemLayoutA{})));

  using TME_B4 = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<std::uint8_t const*>(nullptr)),
                  make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                  make_stride(int32_t(0), int32_t(0), int32_t(0), int32_t(0))),
      take<0, 2>(SmemLayoutBFullU8{})));

  using TME_Scale = decltype(make_tme_copy(
      MP31_TME_LOAD{},
      make_tensor(make_gmem_ptr(static_cast<ElementScale const*>(nullptr)),
                  make_shape(int32_t(0), int32_t(0), int32_t(0), int32_t(0)),
                  make_stride(int32_t(0), int32_t(0), int32_t(0), int32_t(0))),
      take<0, 2>(SmemLayoutScaleTme{})));

  // ===== Pipeline types =====
  using PipelineA       = mutlass::Mp31PipelineTmeAsync<Stages>;
  using PipelineBTme    = mutlass::Mp31PipelineTmeAsync<Stages>;
  using PipelineAParams = typename PipelineA::Params;
  using PipelineAState  = typename PipelineA::PipelineState;
  using PipelineBTmeParams = typename PipelineBTme::Params;
  using PipelineBTmeState  = typename PipelineBTme::PipelineState;

  // ===== TME transaction bytes =====
  static constexpr int TmeTransactionBytesA =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutA{})) * sizeof_bits_v<ElementA>);
  static constexpr int TmeTransactionBytesB4 =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutBFullU8{})) * sizeof_bits_v<std::uint8_t>);
  static constexpr int TmeTransactionBytesScale =
      mutlass::bits_to_bytes(size(take<0, 2>(SmemLayoutScaleTme{})) * sizeof_bits_v<ElementScale>);

  // ===== Shared memory storage =====
  struct SharedStorage {
    mute::array_aligned<ElementA,     cosize_v<SmemLayoutA>,       256> smem_a;
    mute::array_aligned<ElementScale, BlockN * Stages,             256> smem_scale_tme0;
    mute::array_aligned<uint8_t,      cosize_v<SmemLayoutBFullU8>, 256> smem_b;
  };

  struct MUTE_ALIGNAS(1) BarrierStorage {
    uint8_t PipelineA[PipelineA::NumBarriers];
    uint8_t PipelineBTme[PipelineBTme::NumBarriers];
  };

  static constexpr int SmemAlignmentBytes = 256;

  // ===== Host-side arguments =====
  struct Arguments {
    ElementA const*        ptr_a;
    StrideA                stride_a;
    std::uint8_t const*    ptr_b4;
    ElementScale const*    ptr_scale_b_prepacked;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t k_packed;
    int32_t scale_k;
    int32_t group_size;
  };

  // ===== Device-side parameters =====
  struct Params {
    TME_A          tme_a;
    TME_B4         tme_b4;
    TME_Scale      tme_scale;
    std::uint8_t const*    ptr_b4;
    ElementScale const*    ptr_scale_b_prepacked;
    int32_t m;
    int32_t n;
    int32_t k;
    int32_t k_packed;
    int32_t scale_k;
    int32_t group_size;
  };

  // ===== Arguments → Params conversion (host side) =====
  static Params to_underlying_arguments(Arguments const& args) {
    TME_A tme_a = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_a),
                    make_shape(args.m, args.k), args.stride_a),
        take<0, 2>(SmemLayoutA{}));

    int32_t n_tiles      = (args.n + BlockN - 1) / BlockN;
    int32_t k_byte_tiles = (args.k_packed + BlockKBytes - 1) / BlockKBytes;

    auto stride_scale = make_stride(int32_t(1),
                                    int32_t(0),
                                    int32_t(BlockN),
                                    int32_t(args.n));
    auto stride_b4    = make_stride(int32_t(BlockKBytes),
                                    int32_t(1),
                                    k_byte_tiles * int32_t(BlockN * BlockKBytes),
                                    int32_t(BlockN * BlockKBytes));

    TME_B4 tme_b4 = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_b4),
                    make_shape(int32_t(BlockN), int32_t(BlockKBytes), n_tiles, k_byte_tiles),
                    stride_b4),
        take<0, 2>(SmemLayoutBFullU8{}));

    TME_Scale tme_scale = make_tme_copy(
        MP31_TME_LOAD{},
        make_tensor(make_gmem_ptr(args.ptr_scale_b_prepacked),
                    make_shape(int32_t(BlockN), int32_t(1), n_tiles, args.scale_k),
                    stride_scale),
        take<0, 2>(SmemLayoutScaleTme{}));

    return {tme_a, tme_b4, tme_scale,
            args.ptr_b4, args.ptr_scale_b_prepacked,
            args.m, args.n, args.k, args.k_packed,
            args.scale_k, args.group_size};
  }

  // ===========================================================================
  // Device methods
  // ===========================================================================

  // ----- Producer: issue TME loads for one K-tile -----
  //
  // Called by producer warp squad for each K-tile.
  // warp 0 in the squad loads A, warp 1 loads B + scale (sharing pipeline_btme).
  MUTLASS_DEVICE void load(
      Params const& params,
      char*         smem,
      PipelineA&       pipeline_a,
      PipelineBTme&    pipeline_btme,
      PipelineAState&    pipeline_a_state,
      PipelineBTmeState& pipeline_btme_state,
      int k_tile_idx,
      int tile_m_idx,
      int tile_n_idx,
      int warp_idx_in_warp_squad) const {

    // -- Construct smem tensors --
    // smem_b is stored packed (uint8_t): 2 int4 per byte.
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
    auto sA          = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    auto sScaleTme0  = make_tensor(make_smem_ptr(storage.smem_scale_tme0.data()),
                                     make_shape(Int<BlockN>{}, Int<Stages>{}));
    auto sB_u8_full  = make_tensor(make_smem_ptr(storage.smem_b.data()),
                                     SmemLayoutBFullU8{});

    // -- TME partition --
    auto cta_tme_a     = params.tme_a.get_slice(0);
    auto cta_tme_b4    = params.tme_b4.get_slice(0);
    auto cta_tme_scale = params.tme_scale.get_slice(0);

    int32_t m = params.m, n = params.n, k = params.k, k_packed = params.k_packed;
    int k_byte_tiles = (k_packed + BlockKBytes - 1) / BlockKBytes;

    auto mA    = params.tme_a.get_tme_tensor(make_shape(m, k));
    auto mB4   = params.tme_b4.get_tme_tensor(
        make_shape(int32_t(BlockN), int32_t(BlockKBytes), (n + BlockN - 1) / BlockN, k_byte_tiles));
    auto mScale = params.tme_scale.get_tme_tensor(
        make_shape(int32_t(BlockN), int32_t(1), (n + BlockN - 1) / BlockN, params.scale_k));

    auto gA_full   = local_tile(mA, Shape<Int<BlockM>, Int<BlockK>>{}, make_coord(_, _));
    auto gB4_tile  = mB4(_, _, tile_n_idx, _);
    auto gScale    = mScale(_, _, tile_n_idx, _);
    auto gA        = gA_full(_, _, tile_m_idx, _);

    auto tAgA_load  = cta_tme_a.partition_S(gA);
    auto tAsA_store = cta_tme_a.partition_D(sA);
    auto tSgScale   = cta_tme_scale.partition_S(gScale);
    auto tSsScale   = cta_tme_scale.partition_D(sScaleTme0);
    auto tBsB_store = cta_tme_b4.partition_D(sB_u8_full);
    auto tBgB_load  = cta_tme_b4.partition_S(gB4_tile);

    // -- Issue loads --
    if (warp_idx_in_warp_squad == 0) {
      // Warp 0: load A
      pipeline_a.producer_acquire(pipeline_a_state);
      uint32_t bar_id = pipeline_a.producer_get_barrier_id(pipeline_a_state);
      copy(params.tme_a.with(bar_id),
           tAgA_load(_, _, _, k_tile_idx),
           tAsA_store(_, _, _, pipeline_a_state.index()));
      ++pipeline_a_state;
    }
    if (warp_idx_in_warp_squad == 1) {
      // Warp 1: load B4 + scale (shared barrier)
      pipeline_btme.producer_acquire(pipeline_btme_state);
      uint32_t bar_id = pipeline_btme.producer_get_barrier_id(pipeline_btme_state);
      copy(params.tme_b4.with(bar_id),
           tBgB_load(_, _, _, k_tile_idx),
           tBsB_store(_, _, _, pipeline_btme_state.index()));
      copy(params.tme_scale.with(bar_id),
           tSgScale(_, _, 0, k_tile_idx),
           tSsScale(_, _, pipeline_btme_state.index()));
      ++pipeline_btme_state;
    }
  }

  // ----- Consumer: MMA + per-group dequant for all K-tiles -----
  //
  // Called by each consumer warp squad. Loops over all k_tiles, performing:
  //   1. Wait for pipeline data
  //   2. MMA into partial accumulator
  //   3. Prefetch next tile wait (overlap with dequant)
  //   4. Dequant: accum += accum_partial * scale
  //   5. Release pipeline stage
  template <class AccumTensor>
  MUTLASS_DEVICE void mma(
      Params const& params,
      char*         smem,
      PipelineA&       pipeline_a,
      PipelineBTme&    pipeline_btme,
      PipelineAState&    pipeline_a_consumer_state,
      PipelineBTmeState& pipeline_btme_consumer_state,
      AccumTensor&     accum,
      int              k_tiles,
      int              consumer_warp_squad_idx,
      int              thread_idx_in_warp_squad) const {

    // -- Construct smem tensors --
    // smem_b is stored packed (uint8_t): 2 int4 per byte.
    SharedStorage& storage = *reinterpret_cast<SharedStorage*>(smem);
    auto sA          = make_tensor(make_smem_ptr(storage.smem_a.data()), SmemLayoutA{});
    auto sScaleTme0  = make_tensor(make_smem_ptr(storage.smem_scale_tme0.data()),
                                     make_shape(Int<BlockN>{}, Int<Stages>{}));
    auto sB_u8_full  = make_tensor(make_smem_ptr(storage.smem_b.data()),
                                     SmemLayoutBFullU8{});

    // -- MMA thread partition --
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warp_squad);

    auto tAsA = thr_mma.partition_A(sA);
    auto tArA = thr_mma.make_fragment_A(tAsA);

    // -- B slices: each consumer takes 1/4 of the N dimension --
    // Must use compile-time coords for local_tile so TiledMma types resolve.
    auto sB0_u8 = local_tile(sB_u8_full, Shape<Int<BlockNPerConsumer>, Int<BlockKBytes>>{}, make_coord(_0{}, _));
    auto sB1_u8 = local_tile(sB_u8_full, Shape<Int<BlockNPerConsumer>, Int<BlockKBytes>>{}, make_coord(_1{}, _));
    auto sB2_u8 = local_tile(sB_u8_full, Shape<Int<BlockNPerConsumer>, Int<BlockKBytes>>{}, make_coord(_2{}, _));
    auto sB3_u8 = local_tile(sB_u8_full, Shape<Int<BlockNPerConsumer>, Int<BlockKBytes>>{}, make_coord(_3{}, _));

    auto sB0 = recast<ElementB>(sB0_u8);
    auto sB1 = recast<ElementB>(sB1_u8);
    auto sB2 = recast<ElementB>(sB2_u8);
    auto sB3 = recast<ElementB>(sB3_u8);

    auto tBsB0 = thr_mma.partition_B(sB0);
    auto tBsB1 = thr_mma.partition_B(sB1);
    auto tBsB2 = thr_mma.partition_B(sB2);
    auto tBsB3 = thr_mma.partition_B(sB3);

    auto tBrB0 = thr_mma.make_fragment_B(tBsB0);
    auto tBrB1 = thr_mma.make_fragment_B(tBsB1);
    auto tBrB2 = thr_mma.make_fragment_B(tBsB2);
    auto tBrB3 = thr_mma.make_fragment_B(tBsB3);

    // Select this consumer's B fragment
    auto tBrB_consumer = (consumer_warp_squad_idx == 0 ? tBrB0
                         : consumer_warp_squad_idx == 1 ? tBrB1
                         : consumer_warp_squad_idx == 2 ? tBrB2
                                                        : tBrB3);

    // -- Identity tensor for scale indexing --
    auto cAccumTile = make_identity_tensor(make_shape(Int<BlockM>{}, Int<BlockNPerConsumer>{}));
    auto tCcAccum   = thr_mma.partition_C(cAccumTile);

    // -- Wait helper --
    auto wait_consumer_tile = [&](PipelineAState const& state_a,
                                  PipelineBTmeState const& state_b) {
      pipeline_a.consumer_wait(state_a);
      pipeline_btme.consumer_wait(state_b);
    };

    // -- Wait for first tile --
    wait_consumer_tile(pipeline_a_consumer_state, pipeline_btme_consumer_state);

    // -- K-tile loop --
    for (int tile_idx = 0; tile_idx < k_tiles; ++tile_idx) {
      int a_stage = pipeline_a_consumer_state.index();
      int b_stage = pipeline_btme_consumer_state.index();

      // 1) MMA into partial accumulator
      constexpr int MmaKIterations = decltype(size<2>(tBrB0))::value;
      static_assert(MmaKIterations > 0, "MmaKIterations must be positive");
      static_assert((BlockK % MmaKIterations) == 0, "BlockK must be divisible by MMA K-iterations");

      auto accum_partial = partition_fragment_C(tiled_mma, make_shape(Int<BlockM>{}, Int<BlockNPerConsumer>{}));
      clear(accum_partial);

      MUTE_UNROLL
      for (int k_iter = 0; k_iter < MmaKIterations; ++k_iter) {
        mute::gemm(tiled_mma,
                   tArA(_, _, k_iter, a_stage),
                   tBrB_consumer(_, _, k_iter, _0{}, b_stage),
                   accum_partial);
      }

      // 2) Prefetch next tile (overlap wait with dequant)
      bool has_next_tile = (tile_idx + 1) < k_tiles;
      PipelineAState    next_a_state = pipeline_a_consumer_state;
      PipelineBTmeState next_b_state = pipeline_btme_consumer_state;
      if (has_next_tile) {
        ++next_a_state;
        ++next_b_state;
        wait_consumer_tile(next_a_state, next_b_state);
      }
      warpsquad_wait();

      // 3) Dequant: multiply partial accumulator by per-group scale
      MUTE_UNROLL
      for (int i = 0; i < size(accum); ++i) {
        int rn_local  = static_cast<int>(get<1>(tCcAccum(i)));
        int rn_global = consumer_warp_squad_idx * BlockNPerConsumer + rn_local;
        ElementScale scale = sScaleTme0(rn_global, b_stage);
        accum(i) += accum_partial(i) * scale;
      }

      // 4) Release pipeline stages
      pipeline_a.consumer_release(pipeline_a_consumer_state);
      pipeline_btme.consumer_release(pipeline_btme_consumer_state);

      if (has_next_tile) {
        pipeline_a_consumer_state    = next_a_state;
        pipeline_btme_consumer_state = next_b_state;
      } else {
        ++pipeline_a_consumer_state;
        ++pipeline_btme_consumer_state;
      }
    }
  }
};

}  // namespace mutlass_kernels::collective
