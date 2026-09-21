#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/aten/quantized/mudnn/QConv.h"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <iostream>
#include "musa.h"
#include "musa_runtime.h"
#include "musa_runtime_api.h"

namespace at::native {
template <typename T>
__global__ void W8A8Conv2dKernel_FLOAT(
    const uint8_t* input,
    const int8_t* weight,
    T* output,
    int B,
    int IC,
    int IH,
    int IW,
    int OC,
    int OH,
    int OW,
    int KH,
    int KW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    float input_scale,
    int input_zp,
    const float* weight_scales,
    const int32_t* weight_zps,
    const float* bias_data,
    bool use_relu) {
  int total_pixels = B * OC * OH * OW;

  int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;

  for (int i = start_idx; i < total_pixels; i += grid_stride) {
    int ow_idx = i % OW;
    int oh_idx = (i / OW) % OH;
    int oc_idx = (i / (OW * OH)) % OC;
    int b_idx = i / (OW * OH * OC);

    int ic_per_group = IC / groups;
    int oc_per_group = OC / groups;
    int g_idx = oc_idx / oc_per_group;
    int ic_start = g_idx * ic_per_group;

    int32_t acc = 0;
    const int in_h_base = oh_idx * stride_h - pad_h;
    const int in_w_base = ow_idx * stride_w - pad_w;

    for (int c = 0; c < ic_per_group; ++c) {
      int current_ic = ic_start + c;

      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          int cur_h = in_h_base + kh * dilation_h;
          int cur_w = in_w_base + kw * dilation_w;

          if (cur_h >= 0 && cur_h < IH && cur_w >= 0 && cur_w < IW) {
            int32_t a_val =
                input[((b_idx * IC + current_ic) * IH + cur_h) * IW + cur_w];

            int weight_idx = ((oc_idx * ic_per_group + c) * KH + kh) * KW + kw;
            int32_t b_val = weight[weight_idx] - weight_zps[oc_idx];

            acc += (a_val - input_zp) * b_val;
          }
        }
      }
    }
    float out_val =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      out_val += bias_data[oc_idx];
    }
    if (use_relu) {
      out_val = fmaxf(0.0, out_val);
    }
    output[i] = static_cast<T>(out_val);
  }
}

template <typename T>
__global__ void W8A8Conv2dKernel_UINT(
    const uint8_t* input,
    const int8_t* weight,
    T* output,
    int B,
    int IC,
    int IH,
    int IW,
    int OC,
    int OH,
    int OW,
    int KH,
    int KW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    float input_scale,
    int input_zp,
    const float* weight_scales,
    const int32_t* weight_zps,
    float output_scale,
    int output_zero_point,
    const float* bias_data,
    bool use_relu) {
  int total_pixels = B * OC * OH * OW;

  int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;

  for (int i = start_idx; i < total_pixels; i += grid_stride) {
    int ow_idx = i % OW;
    int oh_idx = (i / OW) % OH;
    int oc_idx = (i / (OW * OH)) % OC;
    int b_idx = i / (OW * OH * OC);

    int ic_per_group = IC / groups;
    int oc_per_group = OC / groups;
    int g_idx = oc_idx / oc_per_group;
    int ic_start = g_idx * ic_per_group;

    int32_t acc = 0;
    const int in_h_base = oh_idx * stride_h - pad_h;
    const int in_w_base = ow_idx * stride_w - pad_w;

    for (int c = 0; c < ic_per_group; ++c) {
      int current_ic = ic_start + c;

      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          int cur_h = in_h_base + kh * dilation_h;
          int cur_w = in_w_base + kw * dilation_w;

          if (cur_h >= 0 && cur_h < IH && cur_w >= 0 && cur_w < IW) {
            int32_t a_val =
                input[((b_idx * IC + current_ic) * IH + cur_h) * IW + cur_w];

            int weight_idx = ((oc_idx * ic_per_group + c) * KH + kh) * KW + kw;
            int32_t b_val = weight[weight_idx] - weight_zps[oc_idx];

            acc += (a_val - input_zp) * b_val;
          }
        }
      }
    }
    float out_val =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      out_val += bias_data[oc_idx];
    }
    if (use_relu) {
      out_val = fmaxf(0.0, out_val);
    }
    float quantized =
        roundf(out_val / output_scale) + static_cast<float>(output_zero_point);
    float min_bound = 0.0f;
    float max_bound = 255.0f;
    float clamped = fmaxf(min_bound, fminf(quantized, max_bound));
    output[i] = static_cast<T>(clamped);
  }
}

template <typename T>
__global__ void W8A8Conv2dKernel_INT(
    const uint8_t* input,
    const int8_t* weight,
    T* output,
    int B,
    int IC,
    int IH,
    int IW,
    int OC,
    int OH,
    int OW,
    int KH,
    int KW,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int dilation_h,
    int dilation_w,
    int groups,
    float input_scale,
    int input_zp,
    const float* weight_scales,
    const int32_t* weight_zps,
    float output_scale,
    int output_zero_point,
    const float* bias_data,
    bool use_relu) {
  int total_pixels = B * OC * OH * OW;

  int start_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int grid_stride = blockDim.x * gridDim.x;

  for (int i = start_idx; i < total_pixels; i += grid_stride) {
    int ow_idx = i % OW;
    int oh_idx = (i / OW) % OH;
    int oc_idx = (i / (OW * OH)) % OC;
    int b_idx = i / (OW * OH * OC);

    int ic_per_group = IC / groups;
    int oc_per_group = OC / groups;
    int g_idx = oc_idx / oc_per_group;
    int ic_start = g_idx * ic_per_group;

    int32_t acc = 0;
    const int in_h_base = oh_idx * stride_h - pad_h;
    const int in_w_base = ow_idx * stride_w - pad_w;

    for (int c = 0; c < ic_per_group; ++c) {
      int current_ic = ic_start + c;

      for (int kh = 0; kh < KH; ++kh) {
        for (int kw = 0; kw < KW; ++kw) {
          int cur_h = in_h_base + kh * dilation_h;
          int cur_w = in_w_base + kw * dilation_w;

          if (cur_h >= 0 && cur_h < IH && cur_w >= 0 && cur_w < IW) {
            int32_t a_val =
                input[((b_idx * IC + current_ic) * IH + cur_h) * IW + cur_w];

            int weight_idx = ((oc_idx * ic_per_group + c) * KH + kh) * KW + kw;
            int32_t b_val = weight[weight_idx] - weight_zps[oc_idx];

            acc += (a_val - input_zp) * b_val;
          }
        }
      }
    }
    float out_val =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      out_val += bias_data[oc_idx];
    }
    if (use_relu) {
      out_val = fmaxf(0.0, out_val);
    }
    float quantized =
        roundf(out_val / output_scale) + static_cast<float>(output_zero_point);
    float min_bound = -128.0f;
    float max_bound = 127.0f;
    float clamped = fmaxf(min_bound, fminf(quantized, max_bound));
    output[i] = static_cast<T>(clamped);
  }
}

#define GEN_CONV_FUNC(_TYPE)                                                \
  [](at::Tensor input,                                                      \
     double input_scale,                                                    \
     int64_t input_zp,                                                      \
     at::Tensor weight,                                                     \
     at::Tensor weight_scales,                                              \
     at::Tensor weight_zps,                                                 \
     at::Tensor bias,                                                       \
     at::Tensor& output,                                                    \
     IntArrayRef stride,                                                    \
     IntArrayRef padding,                                                   \
     IntArrayRef dilation,                                                  \
     int64_t groups,                                                        \
     double output_scale,                                                   \
     int64_t output_zero_point,                                             \
     bool use_relu) {                                                       \
    auto stream = c10::musa::getCurrentMUSAStream();                        \
    int B = input.size(0);                                                  \
    int IC = input.size(1);                                                 \
    int IH = input.size(2);                                                 \
    int IW = input.size(3);                                                 \
    int OC = weight.size(0);                                                \
    int KH = weight.size(2);                                                \
    int KW = weight.size(3);                                                \
    int OH = output.size(2);                                                \
    int OW = output.size(3);                                                \
    int total_pixels = B * OC * OH * OW;                                    \
    int threadsPerBlock = 1024;                                             \
    int blocksPerGrid = std::min(                                           \
        (total_pixels + threadsPerBlock - 1) / threadsPerBlock, INT32_MAX); \
    float* bias_ptr = nullptr;                                              \
    if (bias.defined() && bias.numel() > 0) {                               \
      bias_ptr = bias.data_ptr<float>();                                    \
    }                                                                       \
    if constexpr (std::is_same_v<_TYPE, uint8_t>) {                         \
      W8A8Conv2dKernel_UINT<_TYPE>                                          \
          <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(                  \
              static_cast<const uint8_t*>(input.data_ptr()),                \
              static_cast<const int8_t*>(weight.data_ptr()),                \
              static_cast<_TYPE*>(output.data_ptr()),                       \
              B,                                                            \
              IC,                                                           \
              IH,                                                           \
              IW,                                                           \
              OC,                                                           \
              OH,                                                           \
              OW,                                                           \
              KH,                                                           \
              KW,                                                           \
              stride[0],                                                    \
              stride[1],                                                    \
              padding[0],                                                   \
              padding[1],                                                   \
              dilation[0],                                                  \
              dilation[1],                                                  \
              groups,                                                       \
              (float)input_scale,                                           \
              (int)input_zp,                                                \
              static_cast<float*>(weight_scales.data_ptr()),                \
              static_cast<int32_t*>(weight_zps.data_ptr()),                 \
              (float)output_scale,                                          \
              (int)output_zero_point,                                       \
              bias_ptr,                                                     \
              use_relu);                                                    \
    } else if constexpr (std::is_same_v<_TYPE, int8_t>) {                   \
      W8A8Conv2dKernel_INT<_TYPE>                                           \
          <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(                  \
              static_cast<const uint8_t*>(input.data_ptr()),                \
              static_cast<const int8_t*>(weight.data_ptr()),                \
              static_cast<_TYPE*>(output.data_ptr()),                       \
              B,                                                            \
              IC,                                                           \
              IH,                                                           \
              IW,                                                           \
              OC,                                                           \
              OH,                                                           \
              OW,                                                           \
              KH,                                                           \
              KW,                                                           \
              stride[0],                                                    \
              stride[1],                                                    \
              padding[0],                                                   \
              padding[1],                                                   \
              dilation[0],                                                  \
              dilation[1],                                                  \
              groups,                                                       \
              (float)input_scale,                                           \
              (int)input_zp,                                                \
              static_cast<float*>(weight_scales.data_ptr()),                \
              static_cast<int32_t*>(weight_zps.data_ptr()),                 \
              (float)output_scale,                                          \
              (int)output_zero_point,                                       \
              bias_ptr,                                                     \
              use_relu);                                                    \
    } else {                                                                \
      W8A8Conv2dKernel_FLOAT<_TYPE>                                         \
          <<<blocksPerGrid, threadsPerBlock, 0, stream>>>(                  \
              static_cast<const uint8_t*>(input.data_ptr()),                \
              static_cast<const int8_t*>(weight.data_ptr()),                \
              static_cast<_TYPE*>(output.data_ptr()),                       \
              B,                                                            \
              IC,                                                           \
              IH,                                                           \
              IW,                                                           \
              OC,                                                           \
              OH,                                                           \
              OW,                                                           \
              KH,                                                           \
              KW,                                                           \
              stride[0],                                                    \
              stride[1],                                                    \
              padding[0],                                                   \
              padding[1],                                                   \
              dilation[0],                                                  \
              dilation[1],                                                  \
              groups,                                                       \
              (float)input_scale,                                           \
              (int)input_zp,                                                \
              static_cast<float*>(weight_scales.data_ptr()),                \
              static_cast<int32_t*>(weight_zps.data_ptr()),                 \
              bias_ptr,                                                     \
              use_relu);                                                    \
    }                                                                       \
  }

// type to CTYPE
#define REGISTER_CONV_KERNEL(_TYPE, _CTYPE) \
  conv_kernels[(int)_TYPE] = GEN_CONV_FUNC(_CTYPE);

struct ConvKernelTable {
  using KernelFunc = std::function<void(
      at::Tensor,
      double,
      int64_t,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor,
      at::Tensor&,
      IntArrayRef,
      IntArrayRef,
      IntArrayRef,
      int64_t,
      double,
      int64_t,
      bool)>;

#define REGISTER_KERNEL_DTYPE                            \
  REGISTER_CONV_KERNEL(at::ScalarType::Half, float16_t); \
  REGISTER_CONV_KERNEL(at::ScalarType::Float, float);    \
  REGISTER_CONV_KERNEL(at::ScalarType::Char, int8_t);    \
  REGISTER_CONV_KERNEL(at::ScalarType::Byte, uint8_t);

  ConvKernelTable() {
    REGISTER_KERNEL_DTYPE;
  }

#undef REGISTER_KERNEL_DTYPE

  void launch(
      at::Tensor input,
      double input_scale,
      int64_t input_zp,
      at::Tensor weight,
      at::Tensor weight_scales,
      at::Tensor weight_zps,
      at::Tensor bias,
      at::Tensor& output,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      c10::ScalarType output_dtype,
      const std::string_view& attr) {
    auto& func = conv_kernels[(int)output_dtype];
    bool use_relu = attr == "relu" ? true : false;
    func(
        input,
        input_scale,
        input_zp,
        weight,
        weight_scales,
        weight_zps,
        bias,
        output,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        use_relu);
  }

  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  KernelFunc conv_kernels[nr_dtype];
};

void conv_kernel(
    at::Tensor input,
    double input_scale,
    int64_t input_zp,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zps,
    at::Tensor bias,
    at::Tensor& output,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype,
    const std::string_view& attr) {
  static ConvKernelTable kernels;
  kernels.launch(
      input,
      input_scale,
      input_zp,
      weight,
      weight_scales,
      weight_zps,
      bias,
      output,
      stride,
      padding,
      dilation,
      groups,
      output_scale,
      output_zero_point,
      output_dtype,
      attr);
}

REGISTER_MUSA_DISPATCH(conv_stub, &conv_kernel);
} // namespace at::native
