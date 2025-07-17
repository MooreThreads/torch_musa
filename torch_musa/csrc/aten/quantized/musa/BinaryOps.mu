#include <ATen/ATen.h>
#include <ATen/ops/_empty_affine_quantized.h>
#include <ATen/ops/empty.h>
#include <torch/library.h>

#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAStream.h"

namespace at {
namespace musa {
namespace {

inline void CheckInputs(const Tensor& qa, const Tensor& qb) {
  TORCH_CHECK(
      qa.qscheme() == kPerTensorAffine,
      "Only per tensor quantization is supported in Add.");
  TORCH_CHECK(
      qa.qscheme() == qb.qscheme(),
      "Both inputs to Add must have the same quantization scheme.");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "Add operands should have same data type.");
  TORCH_CHECK(
      qa.scalar_type() == c10::kQUInt8 || qa.scalar_type() == c10::kQInt8,
      "Add operands support QUint8 and QInt8, now got ",
      toString(qa.scalar_type()));
}

template <typename DataType>
__global__ void QuantizedAddKernel(
    DataType* out,
    const DataType* qa,
    const DataType* qb,
    double qb_dequant_scale,
    double requant_scale,
    double bias_optional,
    int64_t qmin,
    int64_t qmax,
    int64_t total_num) {
  const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int64_t idx = tid * 4;
  if (idx + 3 < total_num) {
    using VecType = typename std::
        conditional<std::is_same<DataType, int8_t>::value, char4, uchar4>::type;
    const VecType qa_val = reinterpret_cast<const VecType*>(qa)[tid];
    const VecType qb_val = reinterpret_cast<const VecType*>(qb)[tid];

    int4 qval;
    double dequant_b0 =
        (qa_val.x + qb_val.x * qb_dequant_scale) * requant_scale;
    double dequant_b1 =
        (qa_val.y + qb_val.y * qb_dequant_scale) * requant_scale;
    double dequant_b2 =
        (qa_val.z + qb_val.z * qb_dequant_scale) * requant_scale;
    double dequant_b3 =
        (qa_val.w + qb_val.w * qb_dequant_scale) * requant_scale;

    qval.x = __float2int_rn(dequant_b0 + bias_optional);
    qval.y = __float2int_rn(dequant_b1 + bias_optional);
    qval.z = __float2int_rn(dequant_b2 + bias_optional);
    qval.w = __float2int_rn(dequant_b3 + bias_optional);

    qval.x = std::min<int32_t>(std::max<int32_t>(qval.x, qmin), qmax);
    qval.y = std::min<int32_t>(std::max<int32_t>(qval.y, qmin), qmax);
    qval.z = std::min<int32_t>(std::max<int32_t>(qval.z, qmin), qmax);
    qval.w = std::min<int32_t>(std::max<int32_t>(qval.w, qmin), qmax);

    out[idx] = static_cast<DataType>(qval.x);
    out[idx + 1] = static_cast<DataType>(qval.y);
    out[idx + 2] = static_cast<DataType>(qval.z);
    out[idx + 3] = static_cast<DataType>(qval.w);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int64_t elem_idx = idx + i;
      if (elem_idx >= total_num)
        return;

      double dequant =
          (qa[elem_idx] + qb[elem_idx] * qb_dequant_scale) * requant_scale;
      int32_t qval = std::min<int32_t>(
          std::max<int32_t>(__float2int_rn(dequant + bias_optional), qmin),
          qmax);
      out[elem_idx] = static_cast<DataType>(qval);
    }
  }
}

template <bool kReluFused>
void QuantizedAddImpl(
    Tensor& qout,
    const Tensor& qa,
    const Tensor& qb,
    double qb_dequant_scale,
    double requant_scale,
    double bias_optional) {
  auto stream = c10::musa::getCurrentMUSAStream();

  int64_t numel = qout.numel();

  const uint32_t block_x = numel > 512 ? 1024 : 512;
  const uint32_t grid_x = (numel / 4 + block_x - 1) / block_x;

  dim3 block_size{block_x, 1, 1};
  dim3 grid_size{grid_x, 1, 1};

  if (ScalarType::QInt8 == qout.scalar_type()) {
    constexpr int64_t qmin =
        kReluFused ? 0 : std::numeric_limits<int8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<int8_t>::max();
    QuantizedAddKernel<int8_t><<<grid_size, block_size, 0, stream>>>(
        static_cast<int8_t*>(qout.data_ptr()),
        static_cast<int8_t*>(qa.data_ptr()),
        static_cast<int8_t*>(qb.data_ptr()),
        qb_dequant_scale,
        requant_scale,
        bias_optional,
        qmin,
        qmax,
        numel);
  } else if (ScalarType::QUInt8 == qout.scalar_type()) {
    constexpr int64_t qmin =
        kReluFused ? 0 : std::numeric_limits<uint8_t>::min();
    constexpr int64_t qmax = std::numeric_limits<uint8_t>::max();
    QuantizedAddKernel<uint8_t><<<grid_size, block_size, 0, stream>>>(
        static_cast<uint8_t*>(qout.data_ptr()),
        static_cast<uint8_t*>(qa.data_ptr()),
        static_cast<uint8_t*>(qb.data_ptr()),
        qb_dequant_scale,
        requant_scale,
        bias_optional,
        qmin,
        qmax,
        numel);
  } else {
    TORCH_CHECK(false, "quantized::add only support qint8 and quint8 dtype");
  }
  musaDeviceSynchronize();
}

// this could be implemented as:
// out_int = (a_fp + b_fp) / out_s + out_zp
//        = ((a_int8 - a_zp) * a_s + (b_int8 - b_zp) * b_s) / out_s + out_zp
//        = ((a_int8 - a_zp + b_int8 * b_s / a_s - b_zp * b_s / a_s)) * a_s /
//        out_s + out_zp
// let's make (b_deq = b_s / a_s) and (req_s = a_s / out_s)
//        = (a_int8 + b_int8 * b_deq - a_zp - b_zp * b_deq) * req_s + out_zp
//        = (a_int8 + b_int8 * b_deq) * req_s + out_zp - (a_zp + b_zp * b_deq) *
//        req_s
// we could compute [(a_zp + b_zp * b_deq) * req_s + out_zp] in advance
// so the kernel only nee to compute:
//        (a_int8 + b_int8 * b_deq) * req_s
template <bool kReluFused>
Tensor QAdd(
    Tensor qa,
    Tensor qb,
    double output_scale,
    int64_t output_zero_point) {
  c10::musa::MUSAGuard device_guard(qa.device());
  if (qa.numel() == 0) {
    return Tensor{};
  }

  TORCH_CHECK(
      qa.sizes() == qb.sizes(),
      "quantized::add currently expects both input tensors to be the same shape");
  TORCH_CHECK(
      qa.scalar_type() == qb.scalar_type(),
      "quantized::add currently expects both input tensors to be the same dtype");
  TORCH_CHECK(
      qa.suggest_memory_format() == qb.suggest_memory_format(),
      "quantized::add expects both input tenseors to be the same memory format");
  at::ScalarType output_type = qa.scalar_type();

  CheckInputs(qa, qb);

  double a_scale = qa.q_scale();
  double b_scale = qb.q_scale();
  int64_t a_zp = qa.q_zero_point();
  int64_t b_zp = qb.q_zero_point();

  double b_dequant_scale = b_scale / a_scale;
  double requant_scale = a_scale / output_scale;
  double bias_optional = 0.f;
  if (b_zp != 0 || a_zp != 0 || output_zero_point != 0) {
    bias_optional =
        output_zero_point - (a_zp + b_zp * b_dequant_scale) * requant_scale;
  }

  at::Tensor quantized_output = at::_empty_affine_quantized(
      qa.sizes(),
      qa.options().dtype(output_type),
      output_scale,
      output_zero_point,
      qa.suggest_memory_format());

  QuantizedAddImpl<kReluFused>(
      quantized_output, qa, qb, b_dequant_scale, requant_scale, bias_optional);

  return quantized_output;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::add"), TORCH_FN(QAdd<false>));
  m.impl(TORCH_SELECTIVE_NAME("quantized::add_relu"), TORCH_FN(QAdd<true>));
}

} // namespace
} // namespace musa
} // namespace at
