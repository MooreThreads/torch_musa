#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/musa/MUSAUtils.h>
#include "torch_musa/csrc/aten/musa/MUSAMacros.muh"

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
#include <musa_runtime.h>
#include <mutlass/arch/mma.h>
#include <mutlass/layout/layout.h>
#include <mutlass/mutlass.h>
#include <mutlass/tensor_ref.h>
#endif

#include <tuple>
#include <type_traits>

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
#define MUTLASS_STATUS_CHECK(status)         \
  {                                          \
    TORCH_CHECK(                             \
        status == mutlass::Status::kSuccess, \
        __func__,                            \
        " : MUTLASS error: ",                \
        cutlassGetStatusString(status));     \
  }
#endif

namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
static inline void check_meta_2to4(const Tensor& meta, const int64_t k) {
  TORCH_CHECK(meta.defined(), "meta(tensor_e) must be defined");
  TORCH_CHECK(meta.dim() == 3, "meta must be 3D, got dim=", meta.dim());
  TORCH_CHECK(
      meta.size(1) == 2 && meta.size(2) == k / 32,
      "meta expected shape [m, 2, k/32], got ",
      meta.sizes());
  TORCH_CHECK(
      meta.scalar_type() == at::kInt || meta.scalar_type() == at::kShort ||
          meta.scalar_type() == at::kChar || meta.scalar_type() == at::kByte,
      "meta should be integer type, got ",
      meta.scalar_type());
}

template <typename out_dtype>
at::Tensor decode_2to4(
    const at::Tensor& packed,
    const at::Tensor& meta,
    int64_t M,
    int64_t K) {
  auto A = at::zeros({M, K}, packed.options());
  const out_dtype* packed_ptr = (const out_dtype*)packed.data_ptr();
  const uint8_t* meta_ptr = (const uint8_t*)meta.data_ptr();
  out_dtype* A_ptr = (out_dtype*)A.data_ptr();
  const int64_t packed_row_elems = K / 2; // each 4 cols -> 2 nonzeros
  const int64_t meta_row_bytes = K / 8; // each 4 cols -> 4 bits (nibble)
  for (int64_t i = 0; i < M; ++i) {
    const int64_t packed_row_base = i * packed_row_elems;
    const int64_t meta_row_base = i * meta_row_bytes;
    for (int64_t q = 0; q < K / 4; ++q) {
      const int64_t col_base = q * 4;
      const int64_t packed_base = packed_row_base + q * 2;
      const uint8_t byte_val = meta_ptr[meta_row_base + (q >> 1)];
      const uint8_t nibble =
          (q & 1) ? ((byte_val >> 4) & 0xF) : (byte_val & 0xF);
      const int idx0 = nibble & 0x3;
      const int idx1 = (nibble >> 2) & 0x3;
      A_ptr[i * K + col_base + idx0] = packed_ptr[packed_base + 0];
      A_ptr[i * K + col_base + idx1] = packed_ptr[packed_base + 1];
    }
  }
  return A;
}

static inline Tensor run_cutlass_semi_structured_spmm(
    const Tensor& packed,
    const Tensor& meta,
    const Tensor& B,
    int64_t m,
    int64_t k,
    int64_t n) {
  // Step 1: decode
  at::Tensor A;
  at::Tensor packed_cpu = packed.to(at::kCPU).contiguous();
  at::Tensor meta_cpu = meta.to(at::kCPU).contiguous();
  if (packed.scalar_type() == at::kBFloat16) {
    A = decode_2to4<mutlass::bfloat16_t>(packed_cpu, meta_cpu, m, k);
  } else if (packed.scalar_type() == at::kHalf) {
    A = decode_2to4<mutlass::half_t>(packed_cpu, meta_cpu, m, k);
  } else if (packed.scalar_type() == at::kFloat) {
    A = decode_2to4<float>(packed_cpu, meta_cpu, m, k);
  } else {
    A = decode_2to4<int8_t>(packed_cpu, meta_cpu, m, k);
  }

  // Step 2: GEMM
  at::Tensor out = at::mm(A.to(packed.device()), B);
  return out;
}
// Wrapper function for MUTLASS sparse GEMM implementation, used
// solely to simplify dispatching from
// sparse_semi_structured_mad_op() function below.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    typename LayoutInputA,
    typename LayoutInputB,
    bool use_tensor_c>
void spgemm_cutlass(
    const Tensor& tensor_a,
    const at::IntArrayRef::value_type& tensor_a_stride,
    const Tensor& tensor_b,
    const at::IntArrayRef::value_type& tensor_b_stride,
    const Tensor& tensor_c,
    const Tensor& tensor_e,
    const Scalar& alpha,
    const Scalar& beta,
    Tensor& tensor_d) {
  // Fix MUTLASS sparse GEMM template arguments that are not
  // provided as template argument of this function, and create an
  // alias for particular instantiation of this template.
  /*
  using LayoutOutput = mutlass::layout::RowMajor; // Result of the operation
  will be provided in row-major format. using MMAOp =
  mutlass::arch::OpClassTensorOp; // Tensor cores are to be used for maximum
  performance. using SmArch = mutlass::arch::Mp31; // Only CC 8.x devices are
  supported at the moment. using SwizzleThreadBlock =
  mutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // This choice
  provides good performance across wide range of operand sizes. constexpr int
  NumStages = 3; // This choice provides good performance across wide range of
  operand sizes. constexpr int NumEVTEpilogueStages = 1;

  constexpr int AlignmentInputA = 128 /
  mutlass::sizeof_bits<ElementInputA>::value; constexpr int AlignmentInputB =
  128 / mutlass::sizeof_bits<ElementInputB>::value; constexpr int
  AlignmentOutput = 128 / mutlass::sizeof_bits<ElementOutput>::value;

  using ElementComputeEpilogue = ElementAccumulator; // Typically slightly
  slower, but more precise than if ElementOutput used. constexpr int
  AlignmentComputeEpilogue = 128 /
  mutlass::sizeof_bits<ElementComputeEpilogue>::value; using ElementC =
  ElementOutput; using LayoutC = LayoutOutput; constexpr int AlignmentC = 128 /
  mutlass::sizeof_bits<ElementC>::value;

  using OutputTileThreadMap = mutlass::epilogue::fusion::OutputTileThreadLayout<
      ThreadblockShape,
      WarpShape,
      ElementOutput,
      AlignmentOutput,
      NumEVTEpilogueStages>;

  using Accum = mutlass::epilogue::fusion::Mp22AccFetch;

  using Alpha =
      mutlass::epilogue::fusion::Mp22ScalarBroadcast<ElementComputeEpilogue>;
  using AlphaArguments = typename Alpha::Arguments;

  using ApplyAlpha = mutlass::epilogue::fusion::VisitorCompute<
      mutlass::multiplies, ElementComputeEpilogue, ElementComputeEpilogue,
      mutlass::FloatRoundStyle::round_to_nearest>;
  using EVTApplyAlpha = mutlass::epilogue::fusion::Mp22EVT<
      ApplyAlpha,
      Alpha,
      Accum>;

  using Beta =
      mutlass::epilogue::fusion::Mp22ScalarBroadcast<ElementComputeEpilogue>;
  using BetaArguments = typename Beta::Arguments;

  using TensorCScalar =
      mutlass::epilogue::fusion::Mp22ScalarBroadcast<ElementC>;
  using TensorCTensor =
      mutlass::epilogue::fusion::VisitorColBroadcast<
          OutputTileThreadMap,
          ElementC,
          mute::Stride<mute::_1, mute::_0, int64_t>>;
  using TensorC = std::conditional_t<use_tensor_c, TensorCTensor,
  TensorCScalar>; using TensorCArguments = typename TensorC::Arguments;

  using ApplyBeta = mutlass::epilogue::fusion::VisitorCompute<
      mutlass::multiplies, ElementComputeEpilogue, ElementComputeEpilogue,
      mutlass::FloatRoundStyle::round_to_nearest>;
  using EVTApplyBeta = mutlass::epilogue::fusion::Mp22EVT<
      ApplyBeta,
      Beta,
      TensorC>;

  using ApplySum = mutlass::epilogue::fusion::VisitorCompute<
      mutlass::plus, ElementComputeEpilogue, ElementComputeEpilogue,
      mutlass::FloatRoundStyle::round_to_nearest>;
  using EVTApplySum = mutlass::epilogue::fusion::Mp22EVT<
      ApplySum,
      EVTApplyAlpha,
      EVTApplyBeta>;

  using Output = mutlass::epilogue::fusion::Mp22AuxStore<
      OutputTileThreadMap, mutlass::FloatRoundStyle::round_to_nearest,
      mute::Stride<int64_t, mute::_1, int64_t>>;

  using EVTOutput = mutlass::epilogue::fusion::Mp22EVT<
      Output,
      EVTApplySum>;

  using Gemm = mutlass::gemm::device::SparseGemmWithVisitor<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EVTOutput,
      SwizzleThreadBlock,
      NumStages,
      AlignmentInputA,
      AlignmentInputB,
      Operator,
      NumEVTEpilogueStages>;

  // Datatype and layout of metadata matrix are inferred from sparse
  // GEMM template.
  using ElementInputE = typename Gemm::ElementE;
  using LayoutInputE = mutlass::layout::RowMajor;
  using ReorderedLayoutInputE = typename Gemm::LayoutE;
  static_assert(
      std::is_same<ReorderedLayoutInputE,
                   mutlass::layout::ColumnMajorInterleaved<2>>::value,
      "Matrix layout used by MUTLASS for reordered metadata for sparse GEMM "
      "change, thus code doing conversions from/to dense matrix has to be "
      "updated.");

  constexpr auto kSparse = Gemm::kSparse;
  constexpr int kElementsPerElementE = Gemm::kElementsPerElementE;

  // Operand sizes.
  const int length_m = tensor_a.size(0);
  const int length_k = tensor_b.size(0);
  const int length_n = tensor_b.size(1);
  const auto tensor_e_ncols = length_k / kSparse / kElementsPerElementE;

  // Determine PyTorch datatype for the metadata matrix.
  auto tensor_e_dtype = at::kChar;
  switch (sizeof(ElementInputE)) {
  case 2:
      tensor_e_dtype = at::kShort;
      break;
  case 4:
      tensor_e_dtype = at::kInt;
      break;
  default:
      TORCH_CHECK(false, __func__, ": invalid size of meta tensor datatype "
               "encountered");
  }
  TORCH_CHECK(tensor_e.dtype() == tensor_e_dtype,
              __func__, " : Expected meta datatype ", tensor_e_dtype,
              ", but got ", tensor_e.dtype());

  // Prepare arguments for MUTLASS sparse GEMM kernel.
  mutlass::gemm::GemmCoord problem_size(length_m, length_n, length_k);
  LayoutInputA layout_a(tensor_a_stride);
  LayoutInputB layout_b(tensor_b_stride);
  auto tensor_a_device_ref =
      mutlass::TensorRef<ElementInputA, LayoutInputA>(
          (ElementInputA*)tensor_a.data_ptr(), layout_a);
  auto tensor_b_device_ref =
      mutlass::TensorRef<ElementInputB, LayoutInputB>(
          (ElementInputB*)tensor_b.data_ptr(), layout_b);
  auto tensor_e_reordered_device_ref =
      mutlass::TensorRef<ElementInputE, ReorderedLayoutInputE>(
          (ElementInputE*)tensor_e.data_ptr(),
          ReorderedLayoutInputE::packed({length_m, tensor_e_ncols}));

  AlphaArguments alpha_arguments{
      [&]() -> AlphaArguments {
          if constexpr (std::is_same_v<ElementComputeEpilogue, mutlass::half_t>
  || std::is_same_v<ElementComputeEpilogue, mutlass::bfloat16_t>) { return
  {ElementComputeEpilogue{alpha.to<float>()}}; } else { return
  {{alpha.to<ElementComputeEpilogue>()}};
          }
      }()
  };
  BetaArguments beta_arguments{
      [&]() -> BetaArguments {
          if constexpr (std::is_same_v<ElementComputeEpilogue, mutlass::half_t>
  || std::is_same_v<ElementComputeEpilogue, mutlass::bfloat16_t>) { return
  {ElementComputeEpilogue{beta.to<float>()}}; } else { return
  {{beta.to<ElementComputeEpilogue>()}};
          }
      }()
  };
  TensorCArguments tensor_c_arguments{
      [&]() -> TensorCArguments {
          if constexpr (use_tensor_c) {
              return {(ElementC*)tensor_c.data_ptr(),
                      ElementC(0),
                      {mute::_1{}, mute::_0{}, problem_size.m()}};
          } else {
              return {{ElementC(0)}};
          }
      }()
  };
  typename Output::Arguments output_arguments{
      (ElementOutput*)tensor_d.data_ptr(),
      {problem_size.n(), mute::_1{}, problem_size.mn().product()}
  };
  typename EVTOutput::Arguments callback_arguments{
      {
          {
              alpha_arguments,     // Alpha
              {},                  // Accum
              {}                   // ApplyAlpha
          },                       // EVTApplyAlpha
          {
              beta_arguments,      // Beta
              tensor_c_arguments,  // TensorC
              {}                   // ApplyBeta
          },                       // EVTApplyBeta
          {}                       // ApplySum
      },                           // EVTApplySum
      output_arguments             // Output
  };                               // EVTOutput

  // Create a tuple of MUTLASS sparse GEMM kernel arguments.
  typename Gemm::Arguments arguments{
      problem_size,
      tensor_a_device_ref,
      tensor_b_device_ref,
      tensor_e_reordered_device_ref,
      callback_arguments};

  mutlass::Status status;

  // Create MUTLASS sparse GEMM kernel object.
  Gemm gemm_op;

  // Verify that sparse GEMM operation with given arguments can be
  // performed by MUTLASS.
  status = gemm_op.can_implement(arguments);
  MUTLASS_STATUS_CHECK(status);

  // Allocate workspace for MUTLASS sparse GEMM kernel.
  const auto workspace_size = Gemm::get_workspace_size(arguments);
  auto workspace = tensor_a.new_empty({(int64_t)workspace_size},
                                      at::TensorOptions().dtype(at::kByte));

  // Initialize MUTLASS sparse GEMM object.
  status = gemm_op.initialize(arguments, workspace.data_ptr(),
                              at::musa::getCurrentMUSAStream());
  MUTLASS_STATUS_CHECK(status);

  // Perform sparse GEMM operation.
  status = gemm_op.run(at::musa::getCurrentMUSAStream());
  MUTLASS_STATUS_CHECK(status);
  */

  TORCH_CHECK(tensor_b.defined(), "B must be defined");
  TORCH_CHECK(tensor_b.dim() == 2, "B must be 2D, got dim=", tensor_b.dim());

  // packed + meta
  const Tensor& packed = tensor_a;
  const Tensor& meta = tensor_e;
  TORCH_CHECK(packed.defined(), "packed(tensor_a) must be defined");
  TORCH_CHECK(meta.defined(), "meta(tensor_e) must be defined");

  // 设备检查（按你们的 kMUSA）
  TORCH_CHECK(
      packed.device() == meta.device() && packed.device() == tensor_b.device(),
      "packed/meta/B must be on same device. packed=",
      packed.device(),
      " meta=",
      meta.device(),
      " B=",
      tensor_b.device());
  const at::OptionalDeviceGuard device_guard(device_of(packed));

  // meta shape: [m,2,16]
  const int64_t m = meta.size(0);
  const int64_t k = tensor_b.size(0);
  const int64_t n = tensor_b.size(1);
  TORCH_CHECK(
      k > 0 && n > 0 && m > 0,
      "invalid matmul sizes m,k,n = ",
      m,
      ",",
      k,
      ",",
      n);
  check_meta_2to4(meta, k);

  // 如果 tensor_d 是 out 参数，推荐允许它 resize 到正确 shape
  if (!tensor_d.defined() || tensor_d.numel() == 0 ||
      tensor_d.sizes() != at::IntArrayRef({m, n})) {
    tensor_d.resize_({m, n});
  }
  TORCH_CHECK(
      tensor_d.dim() == 2 && tensor_d.size(0) == m && tensor_d.size(1) == n,
      "D must be [m,n]=",
      m,
      ",",
      n,
      " but got ",
      tensor_d.sizes());

  // Tensor Bc = tensor_b.contiguous();

  // 1) CUTLASS semi-structured SpMM: out = A @ B
  Tensor out =
      run_cutlass_semi_structured_spmm(packed, meta, tensor_b, m, k, n);

  TORCH_CHECK(
      out.dim() == 2 && out.size(0) == m && out.size(1) == n,
      "kernel output shape mismatch, got ",
      out.sizes(),
      " expected [",
      m,
      ",",
      n,
      "]");

  // 2) alpha scaling
  out.mul_(alpha);

  // 3) beta * C
  if (tensor_c.defined()) {
    Tensor c = tensor_c;

    // dtype/device align
    if (c.scalar_type() != out.scalar_type()) {
      c = c.to(out.scalar_type());
    }
    if (c.device() != out.device()) {
      c = c.to(out.device());
    }

    // Support common addmm inputs:
    // - [m, n] full matrix
    // - [m] row-bias style (your test case) -> [m,1] then broadcast
    // - [n] col-bias style -> [1,n] then broadcast
    // - scalar
    if (c.dim() == 1) {
      if (c.size(0) == m) {
        c = c.unsqueeze(1); // [m,1] -> broadcast to [m,n]
      } else if (c.size(0) == n) {
        c = c.unsqueeze(0); // [1,n] -> broadcast to [m,n]
      } else {
        TORCH_CHECK(
            false,
            "C 1D shape must be [m] or [n], got ",
            c.sizes(),
            ", expected [",
            m,
            "] or [",
            n,
            "]");
      }
    } else if (c.dim() > 2) {
      TORCH_CHECK(false, "C must be scalar/1D/2D, got dim=", c.dim());
    }
    // Let ATen handle broadcasting
    out.add_(c, beta);
  }

  tensor_d.copy_(out);
  C10_MUSA_KERNEL_LAUNCH_CHECK();
}

// Dispatch according to the input tensors layouts combination.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts,
    bool use_tensor_c>
void spgemm_cutlass_dispatch_layouts(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    const Tensor& tensor_c,
    const Tensor& tensor_e,
    const Scalar& alpha,
    const Scalar& beta,
    Tensor& tensor_d) {
  // Determine layouts (row-major or column-major) of input tensors.
  const auto strides_a = tensor_a.strides();
  auto tensor_a_row_major = strides_a[1] == 1;
  auto tensor_a_stride = tensor_a_row_major ? strides_a[0] : strides_a[1];
  const auto strides_b = tensor_b.strides();
  auto tensor_b_row_major = strides_b[1] == 1;
  auto tensor_b_stride = tensor_b_row_major ? strides_b[0] : strides_b[1];

  // Perform dispatching.
  if constexpr (EnableRowMajorRowMajorLayouts) {
    if (tensor_a_row_major && tensor_b_row_major) {
      spgemm_cutlass<
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Operator,
          mutlass::layout::RowMajor,
          mutlass::layout::RowMajor,
          use_tensor_c>(
          tensor_a,
          tensor_a_stride,
          tensor_b,
          tensor_b_stride,
          tensor_c,
          tensor_e,
          alpha,
          beta,
          tensor_d);
      return;
    }
  }
  if constexpr (EnableRowMajorColumnMajorLayouts) {
    if (tensor_a_row_major && !tensor_b_row_major) {
      spgemm_cutlass<
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Operator,
          mutlass::layout::RowMajor,
          mutlass::layout::ColumnMajor,
          use_tensor_c>(
          tensor_a,
          tensor_a_stride,
          tensor_b,
          tensor_b_stride,
          tensor_c,
          tensor_e,
          alpha,
          beta,
          tensor_d);
      return;
    }
  }
  if constexpr (EnableColumnMajorRowMajorLayouts) {
    if (!tensor_a_row_major && tensor_b_row_major) {
      spgemm_cutlass<
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Operator,
          mutlass::layout::ColumnMajor,
          mutlass::layout::RowMajor,
          use_tensor_c>(
          tensor_a,
          tensor_a_stride,
          tensor_b,
          tensor_b_stride,
          tensor_c,
          tensor_e,
          alpha,
          beta,
          tensor_d);
      return;
    }
  }
  if constexpr (EnableColumnMajorColumnMajorLayouts) {
    if (!tensor_a_row_major && !tensor_b_row_major) {
      spgemm_cutlass<
          ElementInputA,
          ElementInputB,
          ElementOutput,
          ElementAccumulator,
          ThreadblockShape,
          WarpShape,
          InstructionShape,
          Operator,
          mutlass::layout::ColumnMajor,
          mutlass::layout::ColumnMajor,
          use_tensor_c>(
          tensor_a,
          tensor_a_stride,
          tensor_b,
          tensor_b_stride,
          tensor_c,
          tensor_e,
          alpha,
          beta,
          tensor_d);
      return;
    }
  }

  TORCH_CHECK(
      false,
      __func__,
      "_dispatch_layouts: Combination of ",
      tensor_a_row_major ? "row-major" : "column_major",
      " and ",
      tensor_b_row_major ? "row-major" : "column_major",
      " layouts for input tensors is not supported");
}

// Dispatch according to the tensor_c tensor being provided or not.
template <
    typename ElementInputA,
    typename ElementInputB,
    typename ElementOutput,
    typename ElementAccumulator,
    typename ThreadblockShape,
    typename WarpShape,
    typename InstructionShape,
    typename Operator,
    bool EnableRowMajorRowMajorLayouts,
    bool EnableRowMajorColumnMajorLayouts,
    bool EnableColumnMajorRowMajorLayouts,
    bool EnableColumnMajorColumnMajorLayouts>
void spgemm_cutlass_dispatch_layouts_tensor_c(
    const Tensor& tensor_a,
    const Tensor& tensor_b,
    const Tensor& tensor_c,
    const Tensor& tensor_e,
    const Scalar& alpha,
    const Scalar& beta,
    Tensor& tensor_d) {
  if (tensor_c.numel() > 0) {
    spgemm_cutlass_dispatch_layouts<
        ElementInputA,
        ElementInputB,
        ElementOutput,
        ElementAccumulator,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Operator,
        EnableRowMajorRowMajorLayouts,
        EnableRowMajorColumnMajorLayouts,
        EnableColumnMajorRowMajorLayouts,
        EnableColumnMajorColumnMajorLayouts,
        true>(tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
  } else {
    spgemm_cutlass_dispatch_layouts<
        ElementInputA,
        ElementInputB,
        ElementOutput,
        ElementAccumulator,
        ThreadblockShape,
        WarpShape,
        InstructionShape,
        Operator,
        EnableRowMajorRowMajorLayouts,
        EnableRowMajorColumnMajorLayouts,
        EnableColumnMajorRowMajorLayouts,
        EnableColumnMajorColumnMajorLayouts,
        false>(tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
  }
}
#endif

// Perform multiply-add operation, using corresponding MUTLASS
// sparse GEMM kernel, to given arguments:
//     result = alpha * mat1 @ mat2 + beta * input
// The "mat2" tensor is a dense tensor, while the "mat1" tensor is a
// sparse semi-structured matrix.  The "input" tensor is optional; if
// provided, it should be a vector, with the number of elements equal
// to the number of rows of "mat1" matrix.  It is assumed that "mat1"
// and "mat2" are 2D tensors, supplied either in row-major or
// column-major layouts (different layouts between these two tensors
// are OK, but not all combinations of formats are supported for some
// datatypes of these matrices).  The "mat1_meta" argument contains
// sparse semi-strucutred metadata.
//
// There exists numerous limitations of MUTLASS sparse GEMM kernel,
// with regards to sizes and alignments of input tensors, their
// layouts and datatypes, and so on; this is the reason for large
// number of checks throughout the code.
//
// TODO: The "input" tensor has to be a vector, such that it could be
// broadcasted to columns of mat1 * mat2.  The case of broadcasting to
// rows of mat1 * mat2 could be also supported, if "input" tensor is a
// vector of corresponding length; and same for the case when "input"
// tensor is a matrix of same size as mat1 * mat2 product.  If these
// updates made here, then remember to update corresponding bits in
// the Inductor code that are handling meta registrations and
// lowerings of aten._sparse_semi_structured_mm and
// aten._sparse_semi_structured_addmm operators.
Tensor sparse_semi_structured_mad_op(
    const Tensor& mat1,
    const Tensor& mat1_meta,
    const Tensor& mat2,
    const std::optional<Tensor>& input_opt,
    const Scalar& alpha,
    const Scalar& beta,
    const std::optional<c10::ScalarType> out_dtype_opt) {
#if defined(USE_ROCM) || defined(_MSC_VER)
  TORCH_CHECK(false, __func__, " : MUTLASS not supported");
  return Tensor{};
#else
  // No need to check that all tensors are on CUDA device, as this
  // is provided by dispatch.

  const auto& input = input_opt.value_or(Tensor{});
  const auto out_dtype = out_dtype_opt.value_or(mat2.scalar_type());

  // For now, only CC 8.x devices are supported.
  const auto dprops = at::musa::getCurrentDeviceProperties();
  // const auto is_sm8x = dprops->major == 8;
  // TORCH_CHECK(is_sm8x,
  //             __func__, " : Supported only on GPUs with compute capability "
  //             "8.x");

  // Validate datatypes of input tensors.
  TORCH_CHECK(
      mat2.dtype() == at::kChar || mat2.dtype() == at::kHalf ||
          mat2.dtype() == at::kBFloat16 || mat2.dtype() == at::kFloat,
      __func__,
      " : The mat2 datatype ",
      mat2.dtype(),
      " is not supported");
  TORCH_CHECK(
      mat1.dtype() == mat2.dtype(),
      __func__,
      " : Expected mat1 datatype ",
      mat2.dtype(),
      ", but got ",
      mat1.dtype());
  if (input.numel() != 0) {
    TORCH_CHECK(
        input.dtype() == out_dtype,
        __func__,
        " : Expected input datatype ",
        out_dtype,
        ", but got ",
        input.dtype());
  }

  // Validate layouts of input tensors.
  TORCH_CHECK(
      mat1.layout() == Layout::Strided,
      __func__,
      " : Expected mat1 argument to be strided, but got "
      "layout ",
      mat1.layout());
  TORCH_CHECK(
      mat1.dim() == 2,
      __func__,
      " : Expected mat1 argument to be 2D tensor, got ",
      mat1.dim(),
      " dims");
  const auto strides_a = mat1.strides();
  TORCH_CHECK(
      strides_a[0] == 1 || strides_a[1] == 1,
      __func__,
      " : Invalid strides for mat1 argument: row stride = ",
      strides_a[0],
      ", column stride = ",
      strides_a[1]);
  TORCH_CHECK(
      mat2.layout() == Layout::Strided,
      __func__,
      " : Expected mat2 argument to be "
      "strided, but got layout ",
      mat2.layout());
  TORCH_CHECK(
      mat2.dim() == 2,
      __func__,
      " : Expected mat2 argument to be 2D tensor, got ",
      mat2.dim(),
      " dims");
  const auto strides_b = mat2.strides();
  TORCH_CHECK(
      strides_b[0] == 1 || strides_b[1] == 1,
      __func__,
      " : Invalid strides for mat2 argument: row stride = ",
      strides_b[0],
      ", column stride = ",
      strides_b[1]);
  if (input.numel() != 0) {
    TORCH_CHECK(
        input.layout() == Layout::Strided,
        __func__,
        " : Expected input argument to be strided, but "
        "got layout ",
        input.layout());
    TORCH_CHECK(
        input.dim() == 1,
        __func__,
        " : Expected input argument to be 1D tensor, "
        "got ",
        input.dim(),
        " dims");
  }

  // Validate sizes of input tensors.
  TORCH_CHECK(
      mat1.size(1) == mat2.size(0) / 2,
      __func__,
      " : Expected mat1 argument to have ",
      mat2.size(0) / 2,
      " columns, but got ",
      mat1.size(1));
  if (input.numel() != 0) {
    TORCH_CHECK(
        input.size(0) == mat1.size(0),
        __func__,
        " : Expected input argument to have ",
        mat1.size(0),
        " elements, but got ",
        input.size(0));
  }

  // Introduce alias names for arguments, according to the MUTLASS
  // naming conventions.
  const auto& tensor_a = mat1;
  const auto& tensor_b = mat2;
  const auto& tensor_c = input;
  const auto& tensor_e = mat1_meta;

  // Create output tensor.
  Tensor tensor_d = tensor_b.new_empty(
      {tensor_a.size(0), tensor_b.size(1)},
      at::TensorOptions().dtype(out_dtype));

  // Call wrapper function for MUTLASS sparse GEMM, dispatching on
  // the input datatype, and then on input tensors layouts.
  // According to the input tensors datatypes and layouts,
  // corresponding template arguments are supplied for instantiating
  // the wrapper function.  The tile sizes template arguments are
  // selected according to the MUTLASS profiler results, for number
  // of runs.
  AT_DISPATCH_SWITCH(
      tensor_a.scalar_type(),
      "sparse_semi_structured_mad_op",
      AT_DISPATCH_CASE(at::ScalarType::Char, [&]() {
        using ElementInputA = int8_t;
        using ElementInputB = int8_t;
        using ElementAccumulator = int32_t;
        using ThreadblockShape = mutlass::gemm::GemmShape<128, 128, 128>;
        using WarpShape = mutlass::gemm::GemmShape<64, 64, 128>;
        using InstructionShape = mutlass::gemm::GemmShape<16, 8, 64>;
        using Operator = mutlass::arch::OpMultiplyAddSaturate;
        const auto EnableRowMajorRowMajorLayouts = false;
        const auto EnableRowMajorColumnMajorLayouts = true;
        const auto EnableColumnMajorRowMajorLayouts = false;
        const auto EnableColumnMajorColumnMajorLayouts = false;
        if (out_dtype == at::kInt) {
          using ElementOutput = int32_t;
          spgemm_cutlass_dispatch_layouts_tensor_c<
              ElementInputA,
              ElementInputB,
              ElementOutput,
              ElementAccumulator,
              ThreadblockShape,
              WarpShape,
              InstructionShape,
              Operator,
              EnableRowMajorRowMajorLayouts,
              EnableRowMajorColumnMajorLayouts,
              EnableColumnMajorRowMajorLayouts,
              EnableColumnMajorColumnMajorLayouts>(
              tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
        } else if (out_dtype == at::kChar) {
          using ElementOutput = int8_t;
          spgemm_cutlass_dispatch_layouts_tensor_c<
              ElementInputA,
              ElementInputB,
              ElementOutput,
              ElementAccumulator,
              ThreadblockShape,
              WarpShape,
              InstructionShape,
              Operator,
              EnableRowMajorRowMajorLayouts,
              EnableRowMajorColumnMajorLayouts,
              EnableColumnMajorRowMajorLayouts,
              EnableColumnMajorColumnMajorLayouts>(
              tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
        }
      }) AT_DISPATCH_CASE(at::ScalarType::Half, [&]() {
        using ElementInputA = mutlass::half_t;
        using ElementInputB = mutlass::half_t;
        using ElementOutput = mutlass::half_t;
        using ElementAccumulator = float;
        using ThreadblockShape = mutlass::gemm::GemmShape<128, 128, 64>;
        using WarpShape = mutlass::gemm::GemmShape<64, 64, 64>;
        using InstructionShape = mutlass::gemm::GemmShape<16, 8, 32>;
        using Operator = mutlass::arch::OpMultiplyAdd;
        const auto EnableRowMajorRowMajorLayouts = true;
        const auto EnableRowMajorColumnMajorLayouts = true;
        const auto EnableColumnMajorRowMajorLayouts = true;
        const auto EnableColumnMajorColumnMajorLayouts = true;
        spgemm_cutlass_dispatch_layouts_tensor_c<
            ElementInputA,
            ElementInputB,
            ElementOutput,
            ElementAccumulator,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            Operator,
            EnableRowMajorRowMajorLayouts,
            EnableRowMajorColumnMajorLayouts,
            EnableColumnMajorRowMajorLayouts,
            EnableColumnMajorColumnMajorLayouts>(
            tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
      }) AT_DISPATCH_CASE(at::ScalarType::BFloat16, [&]() {
        using ElementInputA = mutlass::bfloat16_t;
        using ElementInputB = mutlass::bfloat16_t;
        using ElementOutput = mutlass::bfloat16_t;
        using ElementAccumulator = float;
        using ThreadblockShape = mutlass::gemm::GemmShape<128, 128, 64>;
        using WarpShape = mutlass::gemm::GemmShape<64, 64, 64>;
        using InstructionShape = mutlass::gemm::GemmShape<16, 8, 32>;
        using Operator = mutlass::arch::OpMultiplyAdd;
        const auto EnableRowMajorRowMajorLayouts = true;
        const auto EnableRowMajorColumnMajorLayouts = true;
        const auto EnableColumnMajorRowMajorLayouts = true;
        const auto EnableColumnMajorColumnMajorLayouts = true;
        spgemm_cutlass_dispatch_layouts_tensor_c<
            ElementInputA,
            ElementInputB,
            ElementOutput,
            ElementAccumulator,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            Operator,
            EnableRowMajorRowMajorLayouts,
            EnableRowMajorColumnMajorLayouts,
            EnableColumnMajorRowMajorLayouts,
            EnableColumnMajorColumnMajorLayouts>(
            tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
      }) AT_DISPATCH_CASE(at::ScalarType::Float, [&]() {
        using ElementInputA = float;
        using ElementInputB = float;
        using ElementOutput = float;
        using ElementAccumulator = float;
        using ThreadblockShape = mutlass::gemm::GemmShape<128, 64, 32>;
        using WarpShape = mutlass::gemm::GemmShape<64, 32, 32>;
        using InstructionShape = mutlass::gemm::GemmShape<16, 8, 16>;
        using Operator = mutlass::arch::OpMultiplyAdd;
        const auto EnableRowMajorRowMajorLayouts = true;
        const auto EnableRowMajorColumnMajorLayouts = true;
        const auto EnableColumnMajorRowMajorLayouts = true;
        const auto EnableColumnMajorColumnMajorLayouts = true;
        spgemm_cutlass_dispatch_layouts_tensor_c<
            ElementInputA,
            ElementInputB,
            ElementOutput,
            ElementAccumulator,
            ThreadblockShape,
            WarpShape,
            InstructionShape,
            Operator,
            EnableRowMajorRowMajorLayouts,
            EnableRowMajorColumnMajorLayouts,
            EnableColumnMajorRowMajorLayouts,
            EnableColumnMajorColumnMajorLayouts>(
            tensor_a, tensor_b, tensor_c, tensor_e, alpha, beta, tensor_d);
      }));

  return tensor_d;
#endif
}

// Implementation of aten._sparse_semi_structured_mm operator.
Tensor _sparse_semi_structured_mm(
    const Tensor& mat1,
    const Tensor& mat1_meta,
    const Tensor& mat2,
    const std::optional<c10::ScalarType> out_dtype_opt) {
  return sparse_semi_structured_mad_op(
      mat1, mat1_meta, mat2, std::optional<Tensor>(), 1, 0, out_dtype_opt);
}

// Implementation of aten._sparse_semi_structured_addmm operator.
Tensor _sparse_semi_structured_addmm(
    const Tensor& input,
    const Tensor& mat1,
    const Tensor& mat1_meta,
    const Tensor& mat2,
    const Scalar& alpha,
    const Scalar& beta,
    const std::optional<c10::ScalarType> out_dtype_opt) {
  return sparse_semi_structured_mad_op(
      mat1, mat1_meta, mat2, input, alpha, beta, out_dtype_opt);
}

} // namespace at::native

// Following is just for testing purposes.
namespace at::native {

#if defined(USE_ROCM) || defined(_MSC_VER)
#else
// Copied from tools/util/include/host_reorder.h, from MUTLASS source
// tree.  This is for simplicity - namely, this file is not under
// include/cutlass in this tree, as other MUTLASS include files
// needed, so it would require changing PyTorch CMake configuration;
// furthermore, including this file produces build errors in PyTorch
// at the moment.
template <typename Element, typename LayoutDest, typename LayoutSrc>
static void reorder_meta(
    mutlass::TensorRef<Element, LayoutDest> dest,
    mutlass::TensorRef<Element, LayoutSrc> src,
    const int problem_size_m,
    const int problem_size_k) {
  for (int m = 0; m < problem_size_m; m++) {
    for (int k = 0; k < problem_size_k; k++) {
      // First reorder the rows.
      int group = (sizeof(Element) == 2) ? 32 : 16;
      int interweave = (sizeof(Element) == 2) ? 4 : 2;

      int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
      int dest_col = k;

      // Next swizzle the 2x2 blocks from Z to N.
      if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
        ++dest_row;
        --dest_col;
      } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
        --dest_row;
        ++dest_col;
      }

      dest.at({dest_row, dest_col}) = src.at({m, k});
    }
  }
}
#endif

std::tuple<Tensor, Tensor> _to_sparse_semi_structured(const Tensor& dense) {
#if defined(USE_ROCM) || defined(_MSC_VER)
  TORCH_CHECK(false, __func__, " : MUTLASS not supported");
  return std::make_tuple(Tensor{}, Tensor{});
#else
  // Check dimensions of the dense matrix.
  TORCH_CHECK(
      dense.dim() == 2,
      __func__,
      " : Expected dense argument to be 2D tensor, got ",
      dense.dim(),
      " dims");

  // Determine PyTorch datatype for the metadata matrix.
  auto meta_dtype = at::kChar;
  auto ksparse = 0;
  auto dense_elems_per_meta_elem = 0;
  if (dense.dtype() == at::kChar) {
    meta_dtype = at::kInt;
    ksparse = 4;
    dense_elems_per_meta_elem = 32;
  } else if (dense.dtype() == at::kHalf || dense.dtype() == at::kBFloat16) {
    meta_dtype = at::kShort;
    ksparse = 4;
    dense_elems_per_meta_elem = 16;
  } else if (dense.dtype() == at::kFloat) {
    meta_dtype = at::kShort;
    ksparse = 2;
    dense_elems_per_meta_elem = 8;
  } else {
    TORCH_CHECK(
        false,
        "_to_sparse_semi_structured: Invalid dense argument datatype ",
        dense.dtype(),
        " encountered");
  }

  const auto dense_nrows = dense.size(0);
  const auto dense_ncols = dense.size(1);

  if (dense_nrows % (meta_dtype == at::kShort ? 32 : 16) != 0) {
    TORCH_CHECK(
        false,
        "_to_sparse_semi_structured: Number of rows of dense matrix must "
        "be divisible by ",
        (meta_dtype == at::kShort ? 32 : 16),
        ", but it is ",
        dense_nrows);
  }
  if (dense_ncols % dense_elems_per_meta_elem != 0) {
    TORCH_CHECK(
        false,
        "_to_sparse_semi_structured: Number of columns of dense matrix "
        "must be divisible by ",
        dense_elems_per_meta_elem,
        ", but it is ",
        dense_ncols);
  }

  const auto dense_cpu = dense.to("cpu");

  const auto mask_cpu = dense_cpu != at::zeros({1}, dense_cpu.options());

  const auto sparse_cpu =
      dense_cpu.masked_select(mask_cpu).view({dense_nrows, dense_ncols / 2});

  const auto meta_nrows = dense_nrows;
  const auto meta_ncols = dense_ncols / dense_elems_per_meta_elem;
  auto meta_cpu = dense_cpu.new_empty(
      {meta_nrows, meta_ncols}, at::TensorOptions().dtype(meta_dtype));

  auto* mask_cpu_ptr = mask_cpu.data_ptr<bool>();
  for (auto i = 0; i < meta_nrows; ++i) {
    for (auto j = 0; j < meta_ncols; ++j) {
      uint64_t meta_val = 0;
      for (auto k = 0; k < dense_elems_per_meta_elem / ksparse;
           ++k, mask_cpu_ptr += ksparse) {
        const auto mask_elems = (ksparse == 4) ? std::make_tuple(
                                                     mask_cpu_ptr[0],
                                                     mask_cpu_ptr[1],
                                                     mask_cpu_ptr[2],
                                                     mask_cpu_ptr[3])
                                               : std::make_tuple(
                                                     mask_cpu_ptr[0],
                                                     mask_cpu_ptr[0],
                                                     mask_cpu_ptr[1],
                                                     mask_cpu_ptr[1]);
        auto meta_quadruple = 0;
        if (mask_elems == std::make_tuple(1, 1, 0, 0)) {
          meta_quadruple = 4; // 0100
        } else if (mask_elems == std::make_tuple(1, 0, 1, 0)) {
          meta_quadruple = 8; // 1000
        } else if (mask_elems == std::make_tuple(0, 1, 1, 0)) {
          meta_quadruple = 9; // 1001
        } else if (mask_elems == std::make_tuple(1, 0, 0, 1)) {
          meta_quadruple = 12; // 1100
        } else if (mask_elems == std::make_tuple(0, 1, 0, 1)) {
          meta_quadruple = 13; // 1101
        } else if (mask_elems == std::make_tuple(0, 0, 1, 1)) {
          meta_quadruple = 14; // 1110
        } else {
          TORCH_CHECK(
              false,
              "_to_sparse_semi_structured: dense argument does not match ",
              (dense.dtype() != at::kFloat) ? "2:4" : "1:2",
              "sparsity pattern");
        }
        meta_val = meta_val | (meta_quadruple << (4 * k));
      }
      const auto idx = i * meta_ncols + j;
      if (meta_dtype == at::kShort) {
        using MetaElement = int16_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      } else if (meta_dtype == at::kInt) {
        using MetaElement = int32_t;
        const auto meta_cpu_ptr = meta_cpu.data_ptr<MetaElement>();
        meta_cpu_ptr[idx] = (MetaElement)meta_val;
      }
    }
  }

  auto meta_reordered_cpu = meta_cpu.new_empty({meta_nrows, meta_ncols});
  using MetaLayout = mutlass::layout::RowMajor;
  using MetaReorderedLayout = mutlass::layout::ColumnMajorInterleaved<2>;
  if (meta_dtype == at::kShort) {
    using MetaElement = int16_t;
    auto meta_cpu_ref = mutlass::TensorRef<MetaElement, MetaLayout>(
        meta_cpu.data_ptr<MetaElement>(),
        MetaLayout::packed({meta_nrows, meta_ncols}));
    auto meta_reordered_cpu_ref =
        mutlass::TensorRef<MetaElement, MetaReorderedLayout>(
            meta_reordered_cpu.data_ptr<MetaElement>(),
            MetaReorderedLayout::packed({meta_nrows, meta_ncols}));
    reorder_meta(meta_reordered_cpu_ref, meta_cpu_ref, meta_nrows, meta_ncols);
  } else if (meta_dtype == at::kInt) {
    using MetaElement = int32_t;
    auto meta_cpu_ref = mutlass::TensorRef<MetaElement, MetaLayout>(
        meta_cpu.data_ptr<MetaElement>(),
        MetaLayout::packed({meta_nrows, meta_ncols}));
    auto meta_reordered_cpu_ref =
        mutlass::TensorRef<MetaElement, MetaReorderedLayout>(
            meta_reordered_cpu.data_ptr<MetaElement>(),
            MetaReorderedLayout::packed({meta_nrows, meta_ncols}));
    reorder_meta(meta_reordered_cpu_ref, meta_cpu_ref, meta_nrows, meta_ncols);
  }

  return std::make_tuple(
      sparse_cpu.to(dense.device()), meta_reordered_cpu.to(dense.device()));
#endif
}

} // namespace at::native
