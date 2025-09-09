#include "torch_musa/csrc/aten/ops/musa/OneHot.muh"

#include <musa_runtime_api.h>

#include <ATen/Dispatch_v2.h>

#include "torch_musa/csrc/aten/musa/Exceptions.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at::musa {

namespace {

constexpr int bits_of_byte = 8;

template <typename T, int vlen>
__global__ void OneHotOutAlign(
    T* out,
    const T* in,
    const int depth,
    const int n,
    FastDivmod fdm) {
  using Vec = VecType<T, vlen * sizeof(T) * bits_of_byte>;

  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x * vlen;

  for (int oid = tid * vlen; oid < n; oid += stride) {
    Vec vec(T(0));
    uint32_t iid, nid;
    fdm(iid, nid, oid);
    T tgt = in[iid];
#pragma unroll
    for (int i = 0; i < vlen; ++i, ++nid) {
      if (nid == tgt) {
        vec.val_.elem[i] = T(1);
      }
    }
    Vec::store(out, oid, vec);
  }
}

template <typename T>
__global__ void OneHotOutNotAlign(
    T* out,
    const T* in,
    const int depth,
    const int n,
    FastDivmod fdm) {
  const int tid = threadIdx.x + blockIdx.x * blockDim.x;
  const int stride = blockDim.x * gridDim.x;

  for (int oid = tid; oid < n; oid += stride) {
    uint32_t iid, nid;
    fdm(iid, nid, oid);
    out[oid] = (in[iid] == T(nid)) ? T(1) : T(0);
  }
}

template <typename T>
void LaunchOneHotKernel(Tensor& o, const Tensor& i, int depth) {
  auto& h = GetMudnnHandle();
  musaDeviceProp device_prop;
  int device_id = h.GetDeviceId();
  C10_MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_id));
  const int mp_num = device_prop.multiProcessorCount;

  T* o_dptr = o.data_ptr<T>();
  const T* i_dptr = i.const_data_ptr<T>();
  auto stream = c10::musa::getCurrentMUSAStream();

  const int nr_o = o.numel();
  const int threads = 1024;

  constexpr int vlen = 128 / (sizeof(T) * bits_of_byte);
  const bool align = (depth % vlen == 0);
  FastDivmod fdm(depth);

  if (align) {
    const int blocks = std::min(ceil_div(nr_o, threads * vlen), mp_num);
    OneHotOutAlign<T, vlen>
        <<<blocks, threads, 0, stream>>>(o_dptr, i_dptr, depth, nr_o, fdm);
  } else {
    const int blocks = std::min(ceil_div(nr_o, threads), mp_num);
    OneHotOutNotAlign<T>
        <<<blocks, threads, 0, stream>>>(o_dptr, i_dptr, depth, nr_o, fdm);
  }

  AT_MUSA_CHECK(musaGetLastError());
}

} // namespace

// Invariants: i.dtype() == kLong && i.dtype() == o.dtype()
void OneHotRun(Tensor& o, const Tensor& i, int num_classes) {
  AT_DISPATCH_V2(
      o.scalar_type(),
      "OneHotRun",
      AT_WRAP(
          [&]() { return LaunchOneHotKernel<scalar_t>(o, i, num_classes); }),
      at::ScalarType::Long);
}

} // namespace at::musa
