#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/LinearAlgebra.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <musa_runtime.h>

namespace at {
namespace native {
namespace {

template <typename Dtype>
__global__ void SmallMatInverseKernel(
    Dtype* out,
    const Dtype* A,
    const int batches,
    const int N) {
  // square matrix smaller than 9x9
  // Gaussian Elimination to calculate matrix inverse
  int gid0 = threadIdx.x + blockIdx.x * blockDim.x;
  int global_stride = blockDim.x * gridDim.x;

  while (gid0 < batches * N) {
    int bat_id = gid0 / N;
    int col_id = gid0 % N;
    Dtype augmat[8][9];
    Dtype res[8];
    for (int i = 0; i < N; i++) {
      for (int j = 0; j < N; j++) {
        augmat[i][j] = A[bat_id * N * N + i * N + j];
      }
      augmat[i][N] = i == col_id ? (Dtype)1 : (Dtype)0;
    }

    for (int i = 0; i < N; i++) {
      bool can_be_inversed = 0;
      Dtype col_max_val = (Dtype)0;
      int col_max_id = i;
      for (int j = i; j < N; j++) {
        if (fabs(augmat[j][i]) > 1e-7) {
          // caculate max of the colum
          bool swap = fabs(augmat[j][i]) > fabs(col_max_val);
          col_max_val = swap ? augmat[j][i] : col_max_val;
          col_max_id = swap ? j : col_max_id;

          can_be_inversed = 1;
        }
      }
      if (can_be_inversed) {
        for (int j = i; j <= N; j++) {
          Dtype temp = augmat[i][j];
          augmat[i][j] = augmat[col_max_id][j];
          augmat[col_max_id][j] = temp;
        }

        for (int j = i + 1; j < N; j++) {
          Dtype coef = -augmat[j][i] / col_max_val;
          for (int k = i; k <= N; k++) {
            augmat[j][k] += coef * augmat[i][k];
          }
        }
      } else {
        return;
      }
    }

    for (int i = N - 1; i >= 0; i--) {
      res[i] = augmat[i][N];
      for (int j = N - 1; j > i; j--) {
        res[i] -= res[j] * augmat[i][j];
      }
      res[i] /= augmat[i][i];
      out[bat_id * N * N + i * N + col_id] = res[i];
    }

    gid0 += global_stride;
  }
}

#define REGISTER_KERNEL(_DTYPE)                                        \
  SmallMatInverseKernel<_DTYPE><<<block_num, block_size, 0, stream>>>( \
      static_cast<_DTYPE*>(out.data_ptr()),                            \
      static_cast<_DTYPE*>(in.data_ptr()),                             \
      batches,                                                         \
      m);                                                              \
  break;

void _SmallMatInverseRun(Tensor& out, const Tensor& in) {
  at::musa::muHandle& h = GetMudnnHandle();
  auto stream = c10::musa::getCurrentMUSAStream();
  int ndim = in.dim();
  int m = in.sizes()[-1 + ndim];
  int batches = 1;
  for (int i = 0; i < ndim - 2; i++) {
    batches *= in.sizes()[i];
  }

  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;

#if (defined(MUSA_ARCH) && MUSA_ARCH >= 21)
  int max_block_num = mp_num;
#else
  int max_block_num = INT32_MAX;
#endif
  const int block_size = 1024;
  const int block_num =
      std::min(at::musa::ceil_div(batches * m, block_size), max_block_num);

  switch (in.scalar_type()) {
    case at::ScalarType::Float:
      REGISTER_KERNEL(float)
    case at::ScalarType::Double:
      REGISTER_KERNEL(double)
    default:
      TORCH_CHECK(false, "Unsupported dtype: ", in.scalar_type());
  }
}

} // namespace

void InverseRun(Tensor& out, const Tensor& in) {
  int ndim = in.dim();
  TORCH_CHECK(
      in.scalar_type() == at::ScalarType::Float ||
          in.scalar_type() == at::ScalarType::Double,
      "Unsupported dtype ",
      out.scalar_type());
  TORCH_CHECK(ndim >= 2, "Input ndim must not be less than 2, but got ", ndim);
  int m = in.sizes()[-2 + ndim];
  int n = in.sizes()[-1 + ndim];
  TORCH_CHECK(
      m == n && m > 0,
      "Input is not a square matrix, whose height and width are: ",
      m,
      " and ",
      n);
  if (m <= 8) {
    _SmallMatInverseRun(out, in);
  } else {
    TORCH_CHECK(
        false,
        "Size of square matrix larger than 8 is not supported currently");
  }
}

REGISTER_MUSA_DISPATCH(inverse_stub, &InverseRun);

} // namespace native
} // namespace at
