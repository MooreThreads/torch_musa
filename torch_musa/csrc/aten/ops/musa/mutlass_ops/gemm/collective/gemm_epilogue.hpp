#pragma once

#include <mutlass/mutlass.h>
#include <mute/tensor.hpp>

namespace mutlass_kernels::collective {

using namespace mute;

// ============================================================================
// Base Epilogue
// ============================================================================
// Simple quantized GEMM epilogue: cast accumulator (fp32) to output type (fp16/bf16)
// with boundary mask and direct gmem store. No bias, no activation fusion.
//
// Reusable across W4A16, W4A8, W8A16 etc. -- only ElementOutput may differ.
//
// Template parameters:
//   TiledMma_           : The TiledMma type used in mainloop (for partition_C)
//   ElementOutput_      : Output element type (e.g. mutlass::half_t)
//   ElementAccumulator_ : Accumulator element type (e.g. float)
//   StrideC_            : Output matrix stride type
//   BlockM_             : Tile M dimension
//   BlockNPerConsumer_  : Tile N dimension per consumer warp squad
template <
  class TiledMma_,
  class ElementOutput_,
  class ElementAccumulator_,
  class StrideC_,
  int   BlockM_,
  int   BlockNPerConsumer_
>
struct EpilogueQuantSimple {

  using TiledMma           = TiledMma_;
  using ElementOutput      = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using StrideC            = StrideC_;

  static constexpr int BlockM            = BlockM_;
  static constexpr int BlockNPerConsumer = BlockNPerConsumer_;

  struct Arguments {
    ElementOutput* ptr_c;
    StrideC        stride_c;
    int32_t        m;
    int32_t        n;
  };

  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  // Store accumulator to global memory with boundary masking and type cast.
  //
  // Parameters:
  //   params                  : epilogue params (output pointer, stride, m, n)
  //   accum                   : accumulator fragment (fp32)
  //   tiled_mma               : TiledMma instance (for partition_C)
  //   thread_idx_in_warp_squad: thread index within warp squad
  //   consumer_warp_squad_idx : which consumer warp squad (0..NumMmaWarpSquads-1)
  //   tile_m_idx, tile_n_idx  : CTA tile indices
  //   num_mma_warp_squads     : total number of consumer warp squads
  template <class AccumTensor>
  MUTLASS_DEVICE void store(
      Params const&    params,
      AccumTensor const& accum,
      TiledMma const&  tiled_mma,
      int              thread_idx_in_warp_squad,
      int              consumer_warp_squad_idx,
      int              tile_m_idx,
      int              tile_n_idx,
      int              num_mma_warp_squads) const {

    int32_t m = params.m;
    int32_t n = params.n;

    int n_tile_consumer = tile_n_idx * num_mma_warp_squads + consumer_warp_squad_idx;

    auto mC      = make_tensor(make_gmem_ptr(params.ptr_c),
                                 make_shape(m, n), params.stride_c);
    auto gC_tile = local_tile(mC,
                                Shape<Int<BlockM>, Int<BlockNPerConsumer>>{},
                                make_coord(tile_m_idx, n_tile_consumer));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warp_squad);
    auto tCgC  = thr_mma.partition_C(gC_tile);

    auto cC_tile = make_identity_tensor(shape(gC_tile));
    auto tCcC    = thr_mma.partition_C(cC_tile);

    int m_begin      = tile_m_idx * BlockM;
    int n_begin      = tile_n_idx * num_mma_warp_squads * BlockNPerConsumer +
                       consumer_warp_squad_idx * BlockNPerConsumer;
    int residue_m    = (m - m_begin < BlockM ? m - m_begin : BlockM);
    int residue_n_raw = n - n_begin;
    int residue_n    = (residue_n_raw < 0 ? 0
                        : (residue_n_raw < BlockNPerConsumer ? residue_n_raw
                                                             : BlockNPerConsumer));
    auto residue_mn  = make_coord(residue_m, residue_n);

    MUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      if (elem_less(tCcC(i), residue_mn)) {
        tCgC(i) = static_cast<ElementOutput>(accum(i));
      }
    }
  }
};

// ============================================================================
// Activation Functors
// ============================================================================

template <class T> 
struct EpilogueIdentity {
  MUTLASS_DEVICE T operator()(T x) const {return x;}
};

template <class T>
struct EpilogueRelu {
  MUTLASS_DEVICE T operator()(T x) const {return x > T(0) ? x : T(0);}
};

template <class T> 
struct EpilogueSilu {
  MUTLASS_DEVICE T operator()(T x) const {
    return x / (T(1) + exp(-x));  // x * sigmoid(x)
  }
};

// gelu not use in mixed_dtypes_linear_kernel
template <class T>
struct EpilogueGelu {
  MUTLASS_DEVICE T operator()(T x) const {
    constexpr T c0 = T(0.7978845608f); // 2 / sqrt(pi)
    constexpr T c1 = T(0.044715f);
    return x * T(0.5) * (T(1) + tanh(c0 * (x + c1 * x * x * x)));
  }
};



// ============================================================================
// EpilogueQuantBiasAct -- epilogue with optional bias + activation
// ============================================================================
// Computation per element:  output = cast( ActivationOp( accum + bias[n] ) )
//
// Drop-in replacement for EpilogueQuantSimple -- same store() signature,
// so the kernel template accepts either with zero code change.
//
// Extra template parameter:
//   ActivationOp_  : activation functor (default: EpilogueIdentity = no-op)
template <
  class TiledMma_,
  class ElementOutput_,
  class ElementAccumulator_,
  class StrideC_,
  int   BlockM_,
  int   BlockNPerConsumer_,
  class ActivationOp_ = EpilogueIdentity<ElementAccumulator_>
>
struct EpilogueQuantBiasAct {

  using TiledMma           = TiledMma_;
  using ElementOutput      = ElementOutput_;
  using ElementAccumulator = ElementAccumulator_;
  using StrideC            = StrideC_;
  using ActivationOp       = ActivationOp_;

  static constexpr int BlockM            = BlockM_;
  static constexpr int BlockNPerConsumer = BlockNPerConsumer_;

  struct Arguments {
    ElementOutput*            ptr_c;
    StrideC                   stride_c;
    int32_t                   m;
    int32_t                   n;
    ElementOutput const*      ptr_bias;   // (n,), per-column bias, same type as output (fp16/bf16). nullptr = no bias.
  };

  using Params = Arguments;

  static Params to_underlying_arguments(Arguments const& args) {
    return args;
  }

  template <class AccumTensor>
  MUTLASS_DEVICE void store(
      Params const&      params,
      AccumTensor const& accum,
      TiledMma const&    tiled_mma,
      int                thread_idx_in_warp_squad,
      int                consumer_warp_squad_idx,
      int                tile_m_idx,
      int                tile_n_idx,
      int                num_mma_warp_squads) const {

    int32_t m = params.m;
    int32_t n = params.n;

    int n_tile_consumer = tile_n_idx * num_mma_warp_squads + consumer_warp_squad_idx;

    auto mC      = make_tensor(make_gmem_ptr(params.ptr_c),
                                 make_shape(m, n), params.stride_c);
    auto gC_tile = local_tile(mC,
                                Shape<Int<BlockM>, Int<BlockNPerConsumer>>{},
                                make_coord(tile_m_idx, n_tile_consumer));

    auto thr_mma = tiled_mma.get_thread_slice(thread_idx_in_warp_squad);
    auto tCgC  = thr_mma.partition_C(gC_tile);

    auto cC_tile = make_identity_tensor(shape(gC_tile));
    auto tCcC    = thr_mma.partition_C(cC_tile);

    int m_begin       = tile_m_idx * BlockM;
    int n_begin       = tile_n_idx * num_mma_warp_squads * BlockNPerConsumer
                      + consumer_warp_squad_idx * BlockNPerConsumer;
    int residue_m     = (m - m_begin < BlockM ? m - m_begin : BlockM);
    int residue_n_raw = n - n_begin;
    int residue_n     = (residue_n_raw < 0 ? 0
                         : (residue_n_raw < BlockNPerConsumer ? residue_n_raw
                                                              : BlockNPerConsumer));
    auto residue_mn   = make_coord(residue_m, residue_n);

    ActivationOp act_op{};

    MUTE_UNROLL
    for (int i = 0; i < size(accum); ++i) {
      if (elem_less(tCcC(i), residue_mn)) {
        ElementAccumulator val = accum(i);

        // +bias (per-N column, cast from output type to accumulator type)
        if (params.ptr_bias != nullptr) {
          int rn_local  = static_cast<int>(get<1>(tCcC(i)));
          int rn_global = n_begin + rn_local;
          val += static_cast<ElementAccumulator>(params.ptr_bias[rn_global]); // load bias from gmem (TODO: use tme load)
        }

        // activation
        val = act_op(val);

        tCgC(i) = static_cast<ElementOutput>(val);
      }
    }
  }
};

}  // namespace mutlass_kernels::collective
