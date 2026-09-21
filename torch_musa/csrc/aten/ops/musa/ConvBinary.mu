#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/native/TensorIterator.h>

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/musa/MUSADtype.muh"
#include "torch_musa/csrc/aten/musa/MUSAMath.muh"
#include "torch_musa/csrc/core/MUSAStream.h"

#include <iostream>
#include "musa.h"
#include "musa_runtime.h"
#include "musa_runtime_api.h"
#include "torch_musa/csrc/aten/quantized/mudnn/QConv.h"

namespace at::native {
template <typename T, typename T_acc>
__global__ void W8A8Conv2dBinaryKernel_FLOAT(
    const uint8_t* input,
    const int8_t* weight,
    const T_acc* accum,
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
    float accum_scale,
    int64_t accum_zp,
    float alpha,
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
    float conv_f32 =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      conv_f32 += bias_data[oc_idx];
    }

    float accum_fp32 = static_cast<float>(accum[i] - accum_zp) * accum_scale;

    float out_val = conv_f32 + alpha * accum_fp32;

    if (use_relu) {
      out_val = fmaxf(0.0, out_val);
    }

    output[i] = static_cast<T>(out_val);
  }
}

template <typename T, typename T_acc>
__global__ void W8A8Conv2dBinaryKernel_UINT(
    const uint8_t* input,
    const int8_t* weight,
    const T_acc* accum,
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
    float accum_scale,
    int64_t accum_zp,
    float alpha,
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

    float conv_f32 =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      conv_f32 += bias_data[oc_idx];
    }

    float accum_fp32 = static_cast<float>(accum[i] - accum_zp) * accum_scale;

    float out_val = conv_f32 + alpha * accum_fp32;

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

template <typename T, typename T_acc>
__global__ void W8A8Conv2dBinaryKernel_INT(
    const uint8_t* input,
    const int8_t* weight,
    const T_acc* accum,
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
    float accum_scale,
    int64_t accum_zp,
    float alpha,
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
    float conv_f32 =
        static_cast<float>(acc) * input_scale * weight_scales[oc_idx];
    if (bias_data) {
      conv_f32 += bias_data[oc_idx];
    }

    float accum_fp32 = static_cast<float>(accum[i] - accum_zp) * accum_scale;

    float out_val = conv_f32 + alpha * accum_fp32;

    if (use_relu) {
      out_val = fmaxf(0, out_val);
    }

    float quantized =
        roundf(out_val / output_scale) + static_cast<float>(output_zero_point);
    float min_bound = -128.0f;
    float max_bound = 127.0f;
    float clamped = fmaxf(min_bound, fminf(quantized, max_bound));
    output[i] = static_cast<T>(clamped);
  }
}

#define GEN_CONV_BINARY_FUNC(_TYPE)                                                                       \
  [](at::Tensor input,                                                                                    \
     double input_scale,                                                                                  \
     int64_t input_zp,                                                                                    \
     at::Tensor weight,                                                                                   \
     at::Tensor weight_scales,                                                                            \
     at::Tensor weight_zps,                                                                               \
     at::Tensor accum,                                                                                    \
     at::Tensor bias,                                                                                     \
     at::Tensor& output,                                                                                  \
     IntArrayRef stride,                                                                                  \
     IntArrayRef padding,                                                                                 \
     IntArrayRef dilation,                                                                                \
     int64_t groups,                                                                                      \
     double output_scale,                                                                                 \
     int64_t output_zero_point,                                                                           \
     const std::string_view& attr,                                                                        \
     double accum_scale,                                                                                  \
     int64_t accum_zp,                                                                                    \
     double alpha,                                                                                        \
     bool use_relu) {                                                                                     \
    auto stream = c10::musa::getCurrentMUSAStream();                                                      \
    int B = input.size(0);                                                                                \
    int IC = input.size(1);                                                                               \
    int IH = input.size(2);                                                                               \
    int IW = input.size(3);                                                                               \
    int OC = weight.size(0);                                                                              \
    int KH = weight.size(2);                                                                              \
    int KW = weight.size(3);                                                                              \
    int OH = output.size(2);                                                                              \
    int OW = output.size(3);                                                                              \
    int total_pixels = B * OC * OH * OW;                                                                  \
    int threadsPerBlock = 1024;                                                                           \
    int blocksPerGrid = std::min(                                                                         \
        (total_pixels + threadsPerBlock - 1) / threadsPerBlock, INT32_MAX);                               \
    c10::ScalarType accum_type = accum.scalar_type();                                                     \
    float* bias_ptr = nullptr;                                                                            \
    if (bias.defined() && bias.numel() > 0) {                                                             \
      bias_ptr = bias.data_ptr<float>();                                                                  \
    }                                                                                                     \
    AT_DISPATCH_ALL_TYPES(accum_type, "W8A8 convolution", [&] {                                           \
      if constexpr (std::is_same_v<_TYPE, uint8_t>) {                                                     \
        W8A8Conv2dBinaryKernel_UINT<_TYPE, scalar_t>                                                      \
            <<<blocksPerGrid,                                                                             \
               threadsPerBlock,                                                                           \
               0,                                                                                         \
               stream>>>(  \      
                    static_cast<const uint8_t*>(input.data_ptr()),                                        \
                           static_cast<const int8_t*>(weight.data_ptr()),                                 \
                           static_cast<const scalar_t*>(accum.data_ptr()),                                \
                           static_cast<_TYPE*>(output.data_ptr()),                                        \
                           B,                                                                             \
                           IC,                                                                            \
                           IH,                                                                            \
                           IW,                                                                            \
                           OC,                                                                            \
                           OH,                                                                            \
                           OW,                                                                            \
                           KH,                                                                            \
                           KW,                                                                            \
                           (int)stride[0],                                                                \
                           (int)stride[1],                                                                \
                           (int)padding[0],                                                               \
                           (int)padding[1],                                                               \
                           (int)dilation[0],                                                              \
                           (int)dilation[1],                                                              \
                           (int)groups,                                                                   \
                           (float)input_scale,                                                            \
                           (int)input_zp,                                                                 \
                           static_cast<float*>(weight_scales.data_ptr()),                                 \
                           static_cast<int32_t*>(weight_zps.data_ptr()),                                  \
                           (float)output_scale,                                                           \
                           (int)output_zero_point,                                                        \
                           bias_ptr,                                                                      \
                           (float)accum_scale,                                                            \
                           (int)accum_zp,                                                                 \
                           (float)alpha,                                                                  \
                           use_relu     \                 
                );                                                                                        \
      } else if constexpr (std::is_same_v<_TYPE, int8_t>) {                                               \
        W8A8Conv2dBinaryKernel_INT<_TYPE, scalar_t>                                                       \
            <<<blocksPerGrid,                                                                             \
               threadsPerBlock,                                                                           \
               0,                                                                                         \
               stream>>>(   \   
                    static_cast<const uint8_t*>(input.data_ptr()),                                        \
                            static_cast<const int8_t*>(weight.data_ptr()),                                \
                            static_cast<const scalar_t*>(accum.data_ptr()),                               \
                            static_cast<_TYPE*>(output.data_ptr()),                                       \
                            B,                                                                            \
                            IC,                                                                           \
                            IH,                                                                           \
                            IW,                                                                           \
                            OC,                                                                           \
                            OH,                                                                           \
                            OW,                                                                           \
                            KH,                                                                           \
                            KW,                                                                           \
                            (int)stride[0],                                                               \
                            (int)stride[1],                                                               \
                            (int)padding[0],                                                              \
                            (int)padding[1],                                                              \
                            (int)dilation[0],                                                             \
                            (int)dilation[1],                                                             \
                            (int)groups,                                                                  \
                            (float)input_scale,                                                           \
                            (int)input_zp,                                                                \
                            static_cast<float*>(weight_scales.data_ptr()),                                \
                            static_cast<int32_t*>(weight_zps.data_ptr()),                                 \
                            (float)output_scale,                                                          \
                            (int)output_zero_point,                                                       \
                            bias_ptr,                                                                     \
                            (float)accum_scale,                                                           \
                            (int)accum_zp,                                                                \
                            (float)alpha,                                                                 \
                            use_relu      \                  
                );                                                                                        \
      } else {                                                                                            \
        W8A8Conv2dBinaryKernel_FLOAT<_TYPE, scalar_t><<<blocksPerGrid, threadsPerBlock, 0, stream>>>( \  
                    static_cast<const uint8_t*>(input.data_ptr()),                                            \
                    static_cast<const int8_t*>(weight.data_ptr()),                                            \
                    static_cast<const scalar_t*>(accum.data_ptr()),                                           \
                    static_cast<_TYPE*>(output.data_ptr()),                                                   \
                    B, IC, IH, IW, OC, OH, OW, KH, KW,                                                        \ 
                    (int)stride[0], (int)stride[1], (int)padding[0], (int)padding[1],                         \
                    (int)dilation[0], (int)dilation[1], (int)groups,                                          \
                    (float)input_scale, (int)input_zp,                                                        \
                    static_cast<float*>(weight_scales.data_ptr()),                                            \
                    static_cast<int32_t*>(weight_zps.data_ptr()),                                             \
                    bias_ptr, (float)accum_scale, (int)accum_zp, (float)alpha, use_relu                      \                  
                ); \
      }                                                                                                   \
    });                                                                                                   \
  }

// type to CTYPE
#define REGISTER_CONV_BINARY_KERNEL(_TYPE, _CTYPE) \
  conv_binary_kernels[(int)_TYPE] = GEN_CONV_BINARY_FUNC(_CTYPE);

struct ConvBinaryKernelTable {
  using KernelFunc = std::function<void(
      at::Tensor,
      double,
      int64_t,
      at::Tensor,
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
      const std::string_view&,
      double,
      int64_t,
      double,
      bool)>;

#define REGISTER_KERNEL_DTYPE                                   \
  REGISTER_CONV_BINARY_KERNEL(at::ScalarType::Half, float16_t); \
  REGISTER_CONV_BINARY_KERNEL(at::ScalarType::Float, float);    \
  REGISTER_CONV_BINARY_KERNEL(at::ScalarType::Char, int8_t);    \
  REGISTER_CONV_BINARY_KERNEL(at::ScalarType::Byte, uint8_t);

  ConvBinaryKernelTable() {
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
      at::Tensor accum,
      at::Tensor bias,
      at::Tensor& output,
      IntArrayRef stride,
      IntArrayRef padding,
      IntArrayRef dilation,
      int64_t groups,
      double output_scale,
      int64_t output_zero_point,
      c10::ScalarType output_dtype,
      const std::string_view& attr,
      double accum_scale,
      int64_t accum_zero_point,
      double alpha_value,
      const std::string_view& unary_attr) {
    auto& func = conv_binary_kernels[(int)output_dtype];
    bool use_relu = unary_attr == "relu" ? true : false;
    func(
        input,
        input_scale,
        input_zp,
        weight,
        weight_scales,
        weight_zps,
        accum,
        bias,
        output,
        stride,
        padding,
        dilation,
        groups,
        output_scale,
        output_zero_point,
        attr,
        accum_scale,
        accum_zero_point,
        alpha_value,
        use_relu);
  }

  static constexpr int nr_dtype = (int)at::ScalarType::NumOptions;
  KernelFunc conv_binary_kernels[nr_dtype];
};

void conv_binary_kernel(
    at::Tensor input,
    double input_scale,
    int64_t input_zp,
    at::Tensor weight,
    at::Tensor weight_scales,
    at::Tensor weight_zps,
    at::Tensor accum,
    at::Tensor bias,
    at::Tensor& output,
    IntArrayRef stride,
    IntArrayRef padding,
    IntArrayRef dilation,
    int64_t groups,
    double output_scale,
    int64_t output_zero_point,
    c10::ScalarType output_dtype,
    const std::string_view& attr,
    double accum_scale,
    int64_t accum_zero_point,
    double alpha_value,
    const std::string_view& unary_attr) {
  static ConvBinaryKernelTable kernels;
  kernels.launch(
      input,
      input_scale,
      input_zp,
      weight,
      weight_scales,
      weight_zps,
      accum,
      bias,
      output,
      stride,
      padding,
      dilation,
      groups,
      output_scale,
      output_zero_point,
      output_dtype,
      attr,
      accum_scale,
      accum_zero_point,
      alpha_value,
      unary_attr);
};

REGISTER_MUSA_DISPATCH(conv_binary_stub, &conv_binary_kernel);
} // namespace at::native