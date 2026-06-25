#pragma once

#include <cstdint>

#include <mutlass/mutlass.h>
#include <mutlass/device_kernel.h>
#include <mute/int_tuple.hpp>
#include <mute/tensor.hpp>

#include "../dispatch_policy.hpp"
#include "../collective/mainloop_w4a16_tme_warp_specialization.hpp"
#include "../collective/gemm_epilogue.hpp"
#include "../kernel/gemm_tme_warp_specialization.hpp"

namespace mutlass_kernels {

using namespace mute;

// ============================================================================
// W4A16 GEMM type assembly
// ============================================================================
// C = dequant( A @ B_int4^T, scale )
//
// Template parameters:
//   ElementA_  : A/C element type (mutlass::half_t or mutlass::bfloat16_t)
//   StrideA_   : A stride type
//   StrideC_   : C stride type
template <class ElementA_, class StrideA_, class StrideC_>
struct W4A16Gemm {
  using ElementA = ElementA_;
  using StrideA  = StrideA_;
  using StrideC  = StrideC_;

  // ----- Policy-----
  using Policy = QuantGemmTmeWarpSpecializedPolicy<
      Shape<_128, _256, _128>,  // TileShape_MNK
      64,                       // BlockNPerConsumer
      3                         // Stages (NumLoadWarpSquads=1, NumMmaWarpSquads=4 default)
  >;

  // ----- Collective layers -----
  using Mainloop = collective::MainloopW4A16TmeWs<
      Policy,
      ElementA,               // ElementA (fp16 or bf16)
      StrideA,
      mutlass::int4b_t,       // ElementB
      float,                  // ElementScale
      float                   // ElementAccumulator
  >;

  using Epilogue = collective::EpilogueQuantSimple<
      typename Mainloop::TiledMma,
      ElementA,               // ElementOutput (same as ElementA)
      float,                  // ElementAccumulator
      StrideC,
      Policy::BlockM,
      Policy::BlockNPerConsumer
  >;

  // ----- Kernel -----
  using Kernel = kernel::QuantGemmTmeWarpSpecialized<Mainloop, Epilogue>;

  // ----- Launch helper -----
  static void launch(
      typename Kernel::Arguments const& args,
      typename Kernel::Params const& params,
      musaStream_t stream) {
    dim3 grid  = Kernel::get_grid_shape(args);
    dim3 block = Kernel::get_block_shape();
    mutlass::device_kernel<Kernel>
        <<<grid, block, Kernel::SharedStorageSize, stream>>>(params);
  }
};

// ============================================================================
// W4A16 GEMM + bias + activation
// ============================================================================
// C = cast( ActivationOp( dequant(A @ B_int4^T, scale) + bias ) )
//
// Template parameters:
//   ElementA_      : A/C element type (fp16 or bf16)
//   StrideA_       : A stride type
//   StrideC_       : C stride type
//   ActivationOp_  : activation functor (default: EpilogueIdentity = no-op)
template <
  class ElementA_,
  class StrideA_,
  class StrideC_,
  class ActivationOp_ = collective::EpilogueIdentity<float>
>
struct W4A16GemmBiasAct {
  using ElementA     = ElementA_;
  using StrideA      = StrideA_;
  using StrideC      = StrideC_;
  using ActivationOp = ActivationOp_;

  using Policy = QuantGemmTmeWarpSpecializedPolicy<
      Shape<_128, _256, _128>, 64, 3>;

  using Mainloop = collective::MainloopW4A16TmeWs<
      Policy, ElementA, StrideA,
      mutlass::int4b_t, float, float>;

  using Epilogue = collective::EpilogueQuantBiasAct<
      typename Mainloop::TiledMma,
      ElementA, float, StrideC,
      Policy::BlockM, Policy::BlockNPerConsumer,
      ActivationOp>;

  using Kernel = kernel::QuantGemmTmeWarpSpecialized<Mainloop, Epilogue>;

  static void launch(
      typename Kernel::Arguments const& args,
      typename Kernel::Params const& params,
      musaStream_t stream) {
    dim3 grid  = Kernel::get_grid_shape(args);
    dim3 block = Kernel::get_block_shape();
    mutlass::device_kernel<Kernel>
        <<<grid, block, Kernel::SharedStorageSize, stream>>>(params);
  }
};

// ============================================================================
// Convenience launch functions
// ============================================================================

// Plain: no bias, no activation
template <class ElementA, class StrideA, class StrideC>
void run_w4a16_gemm(
    ElementA const*     ptr_a,
    StrideA             stride_a,
    std::uint8_t const* ptr_b4,
    float const*        ptr_scale_b_prepacked,
    ElementA*           ptr_c,
    StrideC             stride_c,
    int32_t m, int32_t n, int32_t k,
    int32_t k_packed, int32_t scale_k, int32_t group_size,
    musaStream_t stream) {

  using Gemm   = W4A16Gemm<ElementA, StrideA, StrideC>;
  using Kernel = typename Gemm::Kernel;

  typename Kernel::Arguments args{
    {ptr_a, stride_a, ptr_b4, ptr_scale_b_prepacked,
     m, n, k, k_packed, scale_k, group_size},
    {ptr_c, stride_c, m, n}
  };

  auto params = Kernel::to_underlying_arguments(args);
  Gemm::launch(args, params, stream);
}

// With bias + activation
template <class ActivationOp, class ElementA, class StrideA, class StrideC>
void run_w4a16_gemm_bias_act(
    ElementA const*     ptr_a,
    StrideA             stride_a,
    std::uint8_t const* ptr_b4,
    float const*        ptr_scale_b_prepacked,
    ElementA*           ptr_c,
    StrideC             stride_c,
    ElementA const*     ptr_bias,       // (n,), fp16/bf16 same as ElementA. nullptr = no bias
    int32_t m, int32_t n, int32_t k,
    int32_t k_packed, int32_t scale_k, int32_t group_size,
    musaStream_t stream) {

  using Gemm   = W4A16GemmBiasAct<ElementA, StrideA, StrideC, ActivationOp>;
  using Kernel = typename Gemm::Kernel;

  typename Kernel::Arguments args{
    {ptr_a, stride_a, ptr_b4, ptr_scale_b_prepacked,
     m, n, k, k_packed, scale_k, group_size},
    {ptr_c, stride_c, m, n, ptr_bias}
  };

  auto params = Kernel::to_underlying_arguments(args);
  Gemm::launch(args, params, stream);
}

}  // namespace mutlass_kernels
