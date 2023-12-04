#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/ops/Bucketize.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace native {

namespace {
template <typename DstDtype, typename SrcDtype, typename BoundariesDType>
__global__ void BucketizeKernel(
    DstDtype* out_ptr,
    SrcDtype* in_ptr,
    BoundariesDType* bd_ptr,
    bool right,
    const int elements,
    const int bd_num) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    SrcDtype in_val = in_ptr[index];
    DstDtype start_bd = 0;
    DstDtype end_bd = bd_num;
    DstDtype mid_bd;
    BoundariesDType mid_val;
    while (start_bd < end_bd) {
      mid_bd = start_bd + ((end_bd - start_bd) >> 1);
      mid_val = bd_ptr[mid_bd];
      if (right ? (mid_val <= in_val) : (mid_val < in_val)) {
        start_bd = mid_bd + 1;
      } else {
        end_bd = mid_bd;
      }
    }
    out_ptr[index] = start_bd;
  }
}

template <typename DstDtype, typename SrcDtype, typename BoundariesDType>
__global__ void BucketizeSmallBDKernel(
    DstDtype* out_ptr,
    SrcDtype* in_ptr,
    BoundariesDType* bd_ptr,
    bool right,
    const int elements,
    const int bd_num) {
  constexpr int BLOCK_SIZE = 512;
  const int block_size = blockDim.x;
  int lidx = threadIdx.x;
  __shared__ BoundariesDType shared_bd[BLOCK_SIZE];
  for (int i = lidx; i < bd_num; i += block_size) {
    shared_bd[i] = bd_ptr[i];
  }
  __syncthreads();

  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements;
       index += blockDim.x * gridDim.x) {
    SrcDtype in_val = in_ptr[index];
    DstDtype start_bd = 0;
    DstDtype end_bd = bd_num;
    DstDtype mid_bd;
    BoundariesDType mid_val;
    while (start_bd < end_bd) {
      mid_bd = start_bd + ((end_bd - start_bd) >> 1);
      mid_val = shared_bd[mid_bd];
      if (right ? (mid_val <= in_val) : (mid_val < in_val)) {
        start_bd = mid_bd + 1;
      } else {
        end_bd = mid_bd;
      }
    }
    out_ptr[index] = start_bd;
  }
}

#define GEN_FUNCTION(_OUTT, _INT, _BDT)                                      \
  [](Tensor& out,                                                            \
     const Tensor& in,                                                       \
     const Tensor& boundaries,                                               \
     const bool right,                                                       \
     const int elements,                                                     \
     const int bd_num,                                                       \
     const int nr_block,                                                     \
     const int nr_threads) {                                                 \
    auto stream = c10::musa::getCurrentMUSAStream();                         \
    BucketizeKernel<_OUTT, _INT, _BDT><<<nr_block, nr_threads, 0, stream>>>( \
        static_cast<_OUTT*>(out.data_ptr()),                                 \
        static_cast<_INT*>(in.data_ptr()),                                   \
        static_cast<_BDT*>(boundaries.data_ptr()),                           \
        right,                                                               \
        elements,                                                            \
        bd_num);                                                             \
  }

#define GEN_FUNCTION_SMALL(_OUTT, _INT, _BDT)          \
  [](Tensor& out,                                      \
     const Tensor& in,                                 \
     const Tensor& boundaries,                         \
     const bool right,                                 \
     const int elements,                               \
     const int bd_num,                                 \
     const int nr_block,                               \
     const int nr_threads) {                           \
    auto stream = c10::musa::getCurrentMUSAStream();   \
    BucketizeSmallBDKernel<_OUTT, _INT, _BDT>          \
        <<<nr_block, nr_threads, 0, stream>>>(         \
            static_cast<_OUTT*>(out.data_ptr()),       \
            static_cast<_INT*>(in.data_ptr()),         \
            static_cast<_BDT*>(boundaries.data_ptr()), \
            right,                                     \
            elements,                                  \
            bd_num);                                   \
  }

#define REGISTER_KERNEL(_INT_ENUM, _BDT_ENUM, _IN_CTYPE, _BDT_CTYPE) \
  bucketize_kernels[0][(int)_INT_ENUM][(int)_BDT_ENUM] =             \
      GEN_FUNCTION(int32_t, _IN_CTYPE, _BDT_CTYPE);                  \
  bucketize_kernels[1][(int)_INT_ENUM][(int)_BDT_ENUM] =             \
      GEN_FUNCTION(int64_t, _IN_CTYPE, _BDT_CTYPE);                  \
  bucketize_small_kernels[0][(int)_INT_ENUM][(int)_BDT_ENUM] =       \
      GEN_FUNCTION_SMALL(int32_t, _IN_CTYPE, _BDT_CTYPE);            \
  bucketize_small_kernels[1][(int)_INT_ENUM][(int)_BDT_ENUM] =       \
      GEN_FUNCTION_SMALL(int64_t, _IN_CTYPE, _BDT_CTYPE);

struct KernelTable {
  using KernelFunc = std::function<void(
      Tensor&,
      const Tensor&,
      const Tensor&,
      const bool,
      const int,
      const int,
      const int,
      const int)>;

  KernelTable() {
#define REGISTER_FUNC(_INT_ENUM, _BDT_ENUM, _IN_CTYPE, _BD_CTYPE) \
  REGISTER_KERNEL(_INT_ENUM, _BDT_ENUM, _IN_CTYPE, _BD_CTYPE)

#define REGISTER_F32_FUNC \
  REGISTER_FUNC(at::ScalarType::Float, at::ScalarType::Float, float, float)

#define REGISTER_I64_FUNC \
  REGISTER_FUNC(at::ScalarType::Long, at::ScalarType::Long, int64_t, int64_t)

#define REGISTER_I32_FUNC \
  REGISTER_FUNC(at::ScalarType::Int, at::ScalarType::Int, int32_t, int32_t)

#define REGISTER_I32_F32_FUNC \
  REGISTER_FUNC(at::ScalarType::Int, at::ScalarType::Float, int32_t, float)

#define REGISTER_I64_F32_FUNC \
  REGISTER_FUNC(at::ScalarType::Long, at::ScalarType::Float, int64_t, float)

    const int out_types_num = 2; // int32 & int64
    const int nr_dtype = (int)at::ScalarType::NumOptions;
    std::vector<std::vector<KernelFunc>> tem1;
    std::vector<std::vector<KernelFunc>> tem2;
    tem1.resize(nr_dtype, std::vector<KernelFunc>(nr_dtype));
    tem2.resize(nr_dtype, std::vector<KernelFunc>(nr_dtype));
    bucketize_kernels.resize(out_types_num, tem1);
    bucketize_small_kernels.resize(out_types_num, tem2);

    REGISTER_F32_FUNC;
    REGISTER_I64_FUNC;
    REGISTER_I32_FUNC;
    REGISTER_I32_F32_FUNC;
    REGISTER_I64_F32_FUNC;
  }

  void launch(
      Tensor& out,
      const Tensor& in,
      const Tensor& boundaries,
      const bool right,
      const int elements,
      const int bd_num,
      const int nr_block,
      const int nr_threads) const {
    // BLOCK_SIZE = 512
    // TODO(@mt-ai/@mt-sw-compute): update when LMS size change
    // consider the num of active warp, 28 * 1024 / 48 / 4(float) * 4 = 597.33
    int BIGGEST_SHARED_MEM_SIZE_PER_BLOCK = 512;
    bool small_bd = bd_num <= BIGGEST_SHARED_MEM_SIZE_PER_BLOCK;

    int out_idx = out.scalar_type() == at::ScalarType::Int ? 0 : 1;
    auto& func = small_bd
        ? bucketize_small_kernels[out_idx][(int)in.scalar_type()]
                                 [(int)boundaries.scalar_type()]
        : bucketize_kernels[out_idx][(int)in.scalar_type()]
                           [(int)boundaries.scalar_type()];
    if (func) {
      return func(
          out, in, boundaries, right, elements, bd_num, nr_block, nr_threads);
    } else {
      TORCH_CHECK(false, "Bucketize func unsupported!");
    }
  }

  // out_dtype[int32_t, int64_t], in_dtype, boundaries_dtype
  std::vector<std::vector<std::vector<KernelFunc>>> bucketize_kernels;
  std::vector<std::vector<std::vector<KernelFunc>>> bucketize_small_kernels;
};

} // namespace

void BucketizeRun(
    Tensor& out,
    const Tensor& in,
    const Tensor& boundaries,
    bool right) {
  TORCH_CHECK(boundaries.dim() == 1, "boundaries must be 1 dimension");
  TORCH_CHECK(
      out.scalar_type() == at::ScalarType::Int ||
          out.scalar_type() == at::ScalarType::Long,
      "Unsupported out dtype of Bucketize: ",
      out.scalar_type());

  at::musa::muHandle& h = GetMudnnHandle();

  int bd_num = boundaries.numel();
  int elements = in.numel();

  // device info
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  TORCH_CHECK(
      musaSuccess == musaGetDeviceProperties(&device_prop, device_id),
      "musaGetDeviceProperties error");
  const int mp_num = device_prop.multiProcessorCount;
  const uint32_t nr_threads = 512;
  const uint32_t nr_blocks =
      std::min(at::musa::ceil_div(elements, 512), mp_num);

  static KernelTable kernel_bucketize;
  kernel_bucketize.launch(
      out, in, boundaries, right, elements, bd_num, nr_blocks, nr_threads);
}

REGISTER_MUSA_DISPATCH(bucketize_stub, &BucketizeRun);

} // namespace native
} // namespace at
