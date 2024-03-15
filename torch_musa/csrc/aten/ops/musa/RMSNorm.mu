#include <ATen/ATen.h>
#include "ATen/AccumulateType.h"

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>
#else
#include <ATen/ops/empty_native.h>
#endif

#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include "torch_musa/csrc/aten/musa/MUSADeviceUtils.muh"
#include "torch_musa/csrc/aten/ops/musa/RMSNorm.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {

namespace musa {

namespace {
template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
  __device__ float* getPointer() {
    extern __shared__ float s_float[];
    return s_float;
  }
};

template <>
struct SharedMemory<double> {
  __device__ double* getPointer() {
    extern __shared__ double s_double[];
    return s_double;
  }
};

template <typename U>
__device__ inline void RMSOnlineSum(const U curr, U& sigma2) {
  sigma2 = sigma2 + curr * curr;
}

template <typename U>
__device__ inline void ChanRMSOnlineSum(const U sigma2B, U& sigma2) {
  sigma2 = sigma2 + sigma2B;
}

template <typename T, typename U>
__device__ void WelfordMuSigma2(
    const T* __restrict__ vals,
    const int n1,
    const int n2,
    const int i1,
    U& sigma2,
    U* buf) {
  // Assumptions:
  // 1) blockDim.x == warpSize
  // 2) Tensor is contiguous
  // 3) 2*blockDim.y*sizeof(U)+blockDim.y*sizeof(int) shared memory available.
  //
  // compute variance and mean over n2
  U count = U(0);
  sigma2 = U(0);
  if (i1 < n1) {
    // one warp normalizes one n1 index,
    // synchronization is implicit
    // initialize with standard Welford algorithm
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    const T* lvals = vals + i1 * n2;
    int l = 4 * thrx;
    for (; l + 3 < n2; l += 4 * numx) { // each thread count 4 elements to
                                        // execute one mem transaction event.
      for (int k = 0; k < 4; ++k) {
        U curr = static_cast<U>(lvals[l + k]);
        RMSOnlineSum<U>(curr, sigma2);
      }
    }
    for (; l < n2; ++l) {
      U curr = static_cast<U>(lvals[l]);
      RMSOnlineSum<U>(curr, sigma2);
    }
    // FIXME: (lms) remove hard code 4. intra-warp reductions, 4 warps
    for (int l = 0; l <= 4; ++l) {
      int srcLaneB = (threadIdx.x + (1 << l)) & (warpSize - 1);
      U sigma2B = WARP_SHFL(sigma2, srcLaneB);
      ChanRMSOnlineSum<U>(sigma2B, sigma2);
    }
    // threadIdx.x == 0 has correct values for each warp
    // inter-warp reductions
    if (blockDim.y > 1) {
      U* ubuf = (U*)buf;
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.x == 0 && threadIdx.y >= offset &&
            threadIdx.y < 2 * offset) {
          const int wrt_y = threadIdx.y - offset;
          ubuf[2 * wrt_y + 1] = sigma2;
        }
        __syncthreads();
        // lower half merges
        if (threadIdx.x == 0 && threadIdx.y < offset) {
          U sigma2B = ubuf[2 * threadIdx.y + 1];
          ChanRMSOnlineSum<U>(sigma2B, sigma2);
        }
        __syncthreads();
      }
      // threadIdx.x = 0 && threadIdx.y == 0 only thread that has correct values
      if (threadIdx.x == 0 && threadIdx.y == 0) {
        ubuf[1] = sigma2;
      }
      __syncthreads();
      sigma2 = ubuf[1] / U(n2);
      // don't care about final value of count, we know count == n2
    } else {
      sigma2 = WARP_SHFL(sigma2 / U(n2), 0);
    }
  }
}

} // namespace

template <typename T, typename U>
__global__ void musa_rms_norm_forward_kernel_impl(
    T* __restrict__ output,
    U* __restrict__ invvar,
    const T* __restrict__ input,
    const int n1,
    const int n2,
    double epsilon,
    const T* __restrict__ gamma) {
  // We strict the blockDim.x == warpSize
  // gridDim: [1,outter,1]
  // blockDim: [32,4,1]
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    U sigma2; // sigma2 -> for n1 index i1, the i1 row sigma2
    WelfordMuSigma2(input, n1, n2, i1, sigma2, buf);

    const T* lvals = input + i1 * n2;
    T* ovals = output + i1 * n2;
    U c_invvar = ::rsqrt(sigma2 + epsilon);
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = gamma[i] * static_cast<T>(c_invvar * curr);
      }
    } else {
      for (int i = thrx; i < n2; i += numx) {
        U curr = static_cast<U>(lvals[i]);
        ovals[i] = static_cast<T>(c_invvar * curr);
      }
    }
    if (threadIdx.x == 0 && threadIdx.y == 0) {
      invvar[i1] = c_invvar;
    }
    __syncthreads();
  }
}

template <typename T, typename U>
void musa_rms_norm_kernel(
    T* output,
    const T* input,
    U* invvar,
    const T* gamma,
    int inner,
    int outter,
    double epsilon) {
  auto stream = at::musa::getCurrentMUSAStream().stream();
  // 4 warps
  dim3 threads{warpSize, 4, 1};
  // FIXME: const can't be removed?
  const uint64_t max_grid_Y =
      at::musa::getCurrentDeviceProperties()->maxGridSize[1];

  // n1, n2 = N,D
  dim3 blocks(1, std::min((uint64_t)outter, max_grid_Y), 1);
  int bytes_shm_needed =
      threads.y > 1 ? 2 * threads.y * sizeof(T) : 0; // greater the int and fp32
  musa_rms_norm_forward_kernel_impl<<<
      blocks,
      threads,
      bytes_shm_needed,
      stream>>>(output, invvar, input, outter, inner, epsilon, gamma);
}

void musa_rms_norm(
    const at::Tensor& input,
    at::Tensor& invvar,
    at::Tensor& output,
    at::Tensor& gamma,
    int inner,
    int outter,
    at::IntArrayRef normalized_shape,
    double epsilon) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      output.scalar_type(),
      "musa_rms_norm_forward",
      [&]() {
        using accumlation_scalar_t = at::acc_type<scalar_t, true /*is_cuda*/>;
        TORCH_INTERNAL_ASSERT(
            c10::CppTypeToScalarType<accumlation_scalar_t>::value ==
                invvar.scalar_type(),
            "Accumulation ScalarType must be the same as invvar scalar type.");

        musa_rms_norm_kernel<scalar_t, accumlation_scalar_t>(
            output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            invvar.data_ptr<accumlation_scalar_t>(),
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            inner,
            outter,
            epsilon);
      });
}

template <typename V>
__device__ V ClampByMagnitude(V curr_gamma, double eps) {
  const V kMinGamma = V(eps);
  if (curr_gamma >= 0) {
    if (curr_gamma < kMinGamma) {
      return kMinGamma;
    } else {
      return curr_gamma;
    }
  } else {
    if (curr_gamma > -kMinGamma) {
      return -kMinGamma;
    } else {
      return curr_gamma;
    }
  }
}

template <typename T, typename U, typename V>
__device__ void LoadWriteStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf2[write_idx] = curr_dout * (c_h)*invvar[i1];

      } else {
        warp_buf2[write_idx] = U(0);
      }
    }
  } else {
    for (int k = 0; k < blockDim.y; ++k) {
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      warp_buf2[write_idx] = U(0);
    }
  }
}

template <typename T, typename U, typename V>
__device__ void LoadAddStridedInputs(
    const int i1_block,
    const int thr_load_row_off,
    const int thr_load_col_off,
    const int i2_off,
    const int row_stride,
    U* warp_buf2,
    const T* input_or_output,
    const V* dout,
    const int i1_end,
    const int n2,
    const U* __restrict__ invvar,
    const V* __restrict__ gamma,
    const double eps) {
  int i1 = i1_block + thr_load_row_off;
  if (i1 < i1_end) {
    for (int k = 0; k < blockDim.y; ++k) {
      int i2 = i2_off + k;
      int load_idx = i1 * n2 + i2;
      int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
      if (i2 < n2) {
        U c_h = static_cast<U>(input_or_output[load_idx]);
        U curr_dout = static_cast<U>(dout[load_idx]);
        warp_buf2[write_idx] += curr_dout * (c_h)*invvar[i1];
      }
    }
  }
}

template <typename T, typename U, typename V>
__global__ void ComputePartGradGamma(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ invvar,
    U epsilon,
    const V* __restrict__ gamma,
    U* part_grad_gamma,
    const double eps) {
  const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) /
      (blockDim.y * blockDim.y); // divide in rows
  const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
  const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
  const int i1_beg_plus_one =
      (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
  const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
  const int row_stride = blockDim.x + 1;
  const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
  const int thr_load_row_off =
      (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
  const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
  SharedMemory<U> shared;
  U* buf = shared.getPointer(); // buf has at least blockDim.x * blockDim.y *
                                // blockDim.y + (blockDim.y -
                                // 1)*(blockDim.x/blockDim.y) elements
  U* warp_buf2 = (U*)buf + blockDim.y * blockDim.y * row_stride;
  // compute partial sums from strided inputs
  // do this to increase number of loads in flight
  LoadWriteStridedInputs<T, U, V>(
      i1_beg,
      thr_load_row_off,
      thr_load_col_off,
      i2_off,
      row_stride,
      warp_buf2,
      input_or_output,
      dout,
      i1_end,
      n2,
      invvar,
      gamma,
      eps);
  for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
       i1_block += blockDim.y * blockDim.y) {
    LoadAddStridedInputs<T, U, V>(
        i1_block,
        thr_load_row_off,
        thr_load_col_off,
        i2_off,
        row_stride,
        warp_buf2,
        input_or_output,
        dout,
        i1_end,
        n2,
        invvar,
        gamma,
        eps);
  }
  __syncthreads();
  // inter-warp reductions
  // sum within each warp
  U acc2 = U(0);
  for (int k = 0; k < blockDim.y; ++k) {
    int row1 = threadIdx.y + k * blockDim.y;
    int idx1 = row1 * row_stride + threadIdx.x;
    acc2 += warp_buf2[idx1];
  }
  warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
  __syncthreads();
  // sum all warps
  for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
    if (threadIdx.y < offset) {
      int row1 = threadIdx.y;
      int row2 = threadIdx.y + offset;
      int idx1 = row1 * row_stride + threadIdx.x;
      int idx2 = row2 * row_stride + threadIdx.x;
      warp_buf2[idx1] += warp_buf2[idx2];
    }
    __syncthreads();
  }
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (threadIdx.y == 0 && i2 < n2) {
    int row1 = threadIdx.y;
    int row2 = threadIdx.y + 1;
    int idx1 = row1 * row_stride + threadIdx.x;
    int idx2 = row2 * row_stride + threadIdx.x;
    part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
  }
}

/*
U = accumulation
V = output and input
*/
template <typename U, typename V>
__global__ void cuComputeGradGammaBeta(
    const U* part_grad_gamma,
    const int part_size,
    const int n1,
    const int n2,
    V* grad_gamma) {
  // sum partial gradients for gamma and beta
  SharedMemory<U> shared;
  U* buf = shared.getPointer();
  int i2 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i2 < n2) {
    // each warp does sequential reductions until reduced part_size is num_warps
    int num_warp_reductions = part_size / blockDim.y;
    U sum_gamma = U(0);
    U sum_beta = U(0);
    const U* part_grad_gamma_ptr =
        part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
    for (int warp_offset = 0; warp_offset < num_warp_reductions;
         ++warp_offset) {
      sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
    }
    // inter-warp reductions
    const int nbsize3 = blockDim.x * blockDim.y / 2;
    for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
      // top half write to shared memory
      if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
        const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
        buf[write_idx] = sum_gamma;
      }
      __syncthreads();
      // bottom half sums
      if (threadIdx.y < offset) {
        const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
        sum_gamma += buf[read_idx];
      }
      __syncthreads();
    }
    // write out fully summed gradients
    if (threadIdx.y == 0) {
      grad_gamma[i2] = sum_gamma;
    }
  }
}

template <typename T, typename U, typename V>
__global__ void cuComputeGradInput(
    const V* __restrict__ dout,
    const T* __restrict__ input_or_output,
    const int n1,
    const int n2,
    const U* __restrict__ invvar,
    U epsilon,
    const V* gamma,
    T* grad_input,
    const double eps) {
  for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
    U sum_loss1 = U(0);
    U sum_loss2 = U(0);
    const T* k_h = input_or_output + i1 * n2;
    const V* k_dout = dout + i1 * n2;
    const U c_invvar = invvar[i1];
    const int numx = blockDim.x * blockDim.y;
    const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
    if (gamma != NULL) {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_h[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss2 += c_loss * gamma[l + k] * (c_h)*c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss2 += c_loss * gamma[l] * (c_h)*c_invvar;
      }
    } else {
      int l = 4 * thrx;
      for (; l + 3 < n2; l += 4 * numx) {
        for (int k = 0; k < 4; ++k) {
          const U c_h = static_cast<U>(k_h[l + k]);
          const U c_loss = static_cast<U>(k_dout[l + k]);
          sum_loss2 += c_loss * (c_h)*c_invvar;
        }
      }
      for (; l < n2; ++l) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        sum_loss2 += c_loss * (c_h)*c_invvar;
      }
    }
    // intra-warp reductions
    for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
      sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
    }
    // inter-warp reductions
    if (blockDim.y > 1) {
      SharedMemory<U> shared;
      U* buf = shared.getPointer();
      for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
        // upper half of warps write to shared
        if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
          const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
          buf[2 * wrt_i + 1] = sum_loss2;
        }
        __syncthreads();
        // lower half merges the loss
        if (threadIdx.y < offset) {
          const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
          sum_loss2 += buf[2 * read_i + 1];
        }
        __syncthreads();
      }
      if (threadIdx.y == 0) {
        buf[2 * threadIdx.x + 1] = sum_loss2;
      }
      __syncthreads();
      if (threadIdx.y != 0) {
        sum_loss2 = buf[2 * threadIdx.x + 1];
      }
    }
    // all threads now have the two sums over l
    U fH = (U)n2;
    U term1 = (U(1) / fH) * c_invvar;
    T* k_grad_input = grad_input + i1 * n2;
    if (gamma != NULL) {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        const U k_gamma = static_cast<U>(ClampByMagnitude(gamma[l], eps));
        U f_grad_input = fH * c_loss * k_gamma;

        f_grad_input -= c_h * c_invvar * sum_loss2;

        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    } else {
      for (int l = thrx; l < n2; l += numx) {
        const U c_h = static_cast<U>(k_h[l]);
        const U c_loss = static_cast<U>(k_dout[l]);
        U f_grad_input = fH * c_loss;
        f_grad_input -= c_h * c_invvar * sum_loss2;

        f_grad_input *= term1;
        k_grad_input[l] = static_cast<T>(f_grad_input);
      }
    }
    // prevent race where buf is written again before reads are done
    __syncthreads();
  }
}

/**
 * @brief MUSA RMSNorm backward func.
 *
 * @tparam T T is for input and output tensor's data type.
 * @tparam U U is for accumulation computation data type for RMS
 * @param dout
 * @param invvar
 * @param input_or_output
 * @param n1 outter batch size
 * @param n2 inner vector length
 * @param gamma
 * @param epsilon
 * @param grad_input
 * @param grad_gamma
 */
template <typename T, typename U = float>
void musa_rms_norm_backward_kernel(
    const T* dout,
    const U* invvar,
    const at::Tensor& input_or_output,
    int n1, // outter
    int n2, // inner
    const T* gamma,
    double epsilon,
    T* grad_input,
    T* grad_gamma) {
  auto stream = at::musa::getCurrentMUSAStream().stream();
  if (gamma != nullptr) {
    // FIXME: (lms) tuning
    const int part_size = 16;
    const dim3 threads2(32, 4, 1);
    const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1); //
    const auto nshared2_a =
        2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
    const auto nshared2_b = threads2.x * threads2.y * sizeof(U);
    const auto nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
    const auto part_grad_dtype =
        (input_or_output.scalar_type() == at::ScalarType::Half ||
         input_or_output.scalar_type() == at::ScalarType::BFloat16)
        ? at::ScalarType::Float
        : input_or_output.scalar_type();
    at::Tensor part_grad_gamma = at::empty(
        {part_size, n2}, input_or_output.options().dtype(part_grad_dtype));
    auto kernel = &ComputePartGradGamma<T, U, T>;
    kernel<<<blocks2, threads2, nshared2, stream>>>(
        dout,
        input_or_output.data_ptr<T>(),
        n1,
        n2,
        invvar,
        U(epsilon),
        gamma,
        part_grad_gamma.data_ptr<U>(),
        epsilon);
    // sum over part gradient
    const dim3 threads3(32, 8, 1);
    const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
    const int nshared3 = threads3.x * threads3.y * sizeof(U);
    cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
        part_grad_gamma.data_ptr<U>(), part_size, n1, n2, grad_gamma);
  }

  // compute grad_input
  const uint64_t maxGridY =
      at::musa::getCurrentDeviceProperties()->maxGridSize[1];
  const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
  const dim3 threads1(32, 4, 1);
  int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
  auto kernel = cuComputeGradInput<T, U, T>;
  kernel<<<blocks1, threads1, nshared, stream>>>(
      dout,
      input_or_output.data_ptr<T>(),
      n1,
      n2,
      invvar,
      U(epsilon),
      gamma,
      grad_input,
      epsilon);
}

void musa_rms_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& invvar,
    const at::Tensor& input,
    const at::Tensor& gamma,
    at::Tensor& grad_input,
    at::Tensor& grad_gamma,
    at::IntArrayRef normalized_shape,
    int n1,
    int n2,
    double eps) {
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::BFloat16,
      at::ScalarType::Half,
      grad_out.scalar_type(),
      "musa_rms_norm_backward",
      [&] {
        using accumlation_scalar_t = at::acc_type<scalar_t, true /*is_cuda*/>;
        TORCH_INTERNAL_ASSERT(
            c10::CppTypeToScalarType<accumlation_scalar_t>::value ==
                invvar.scalar_type(),
            "Accumulation ScalarType must be the same as invvar scalar type.");
        musa_rms_norm_backward_kernel(
            grad_out.data_ptr<scalar_t>(),
            invvar.data_ptr<accumlation_scalar_t>(),
            input,
            n1,
            n2,
            gamma.defined() ? gamma.data_ptr<scalar_t>() : nullptr,
            eps,
            grad_input.data_ptr<scalar_t>(),
            gamma.defined() ? grad_gamma.data_ptr<scalar_t>() : nullptr);
      });
}

} // namespace musa

} // namespace at
