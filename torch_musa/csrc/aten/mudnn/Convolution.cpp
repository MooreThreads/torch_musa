#include "torch_musa/csrc/aten/mudnn/Convolution.h"
#include "torch_musa/csrc/aten/mudnn/Exception.h"
#include "torch_musa/csrc/aten/utils/Context.h"

namespace at::musa {

#ifdef TORCH_MUSA_USE_MUDNN_C_API

namespace {

using ActivationDescGuard = Descriptor<
    mudnnActivationStruct,
    mudnnCreateActivationDescriptor,
    mudnnDestroyActivationDescriptor>;

mudnnTensorFormat_t InferFormatFromTensor(const int64_t* strideA, int ndims) {
  if (ndims == 4) {
    if (strideA[1] == 1 && strideA[3] > 1) {
      return MUDNN_TENSOR_NHWC;
    }
    return MUDNN_TENSOR_NCHW;
  } else if (ndims == 5) {
    if (strideA[1] == 1 && strideA[4] > 1) {
      return MUDNN_TENSOR_NDHWC;
    }
    return MUDNN_TENSOR_NCDHW;
  }
  return MUDNN_TENSOR_NCHW;
}

void BuildFilterDesc(mudnnFilterDescriptor_t fdesc, const muTensor& t) {
  mudnnDataType_t dtype;
  mudnnTensorFormat_t format;
  int ndims;
  const int64_t* dims;
  const int64_t* strides;
  float scale;
  TORCH_INTERNAL_ASSERT(
      t.GetDescriptorInfoIfSet(
          &dtype, &format, &ndims, &dims, &strides, &scale),
      "filter tensor descriptor info is unavailable before BuildFilterDesc");

  if (format == MUDNN_TENSOR_NCHW) {
    format = InferFormatFromTensor(strides, ndims);
  }

  int filterDimA[MUDNN_DIM_MAX] = {};
  for (int i = 0; i < ndims; ++i) {
    filterDimA[i] = static_cast<int>(dims[i]);
  }

  CHECK_MUDNN_STATUS(
      mudnnSetFilterNdDescriptorWithScale(
          fdesc, dtype, format, ndims, filterDimA, scale),
      "mudnnSetFilterNdDescriptorWithScale");
}

mudnnConvolutionFwdAlgo_t ToMudnnFwdAlgo(Convolution::Algorithm algo) {
  switch (algo) {
    case Convolution::Algorithm::IMPLICIT_GEMM:
      return MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    case Convolution::Algorithm::DIRECT:
      return MUDNN_CONVOLUTION_FWD_ALGO_DIRECT;
    case Convolution::Algorithm::WINOGRAD_NONFUSED:
      return MUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
    case Convolution::Algorithm::GEMM:
      return MUDNN_CONVOLUTION_FWD_ALGO_GEMM;
    default:
      return MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  }
}

Convolution::Algorithm FromMudnnFwdAlgo(mudnnConvolutionFwdAlgo_t algo) {
  switch (algo) {
    case MUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM:
      return Convolution::Algorithm::IMPLICIT_GEMM;
    case MUDNN_CONVOLUTION_FWD_ALGO_DIRECT:
      return Convolution::Algorithm::DIRECT;
    case MUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED:
      return Convolution::Algorithm::WINOGRAD_NONFUSED;
    case MUDNN_CONVOLUTION_FWD_ALGO_GEMM:
      return Convolution::Algorithm::GEMM;
    default:
      return Convolution::Algorithm::IMPLICIT_GEMM;
  }
}

mudnnConvolutionBwdDataAlgo_t ToMudnnBwdDataAlgo(
    Convolution::AlgorithmBwdData algo) {
  switch (algo) {
    case Convolution::AlgorithmBwdData::IMPLICIT_GEMM:
      return MUDNN_CONVOLUTION_BWD_DATA_ALGO_IMPLICIT_GEMM;
    case Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED:
      return MUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED;
    case Convolution::AlgorithmBwdData::GEMM:
      return MUDNN_CONVOLUTION_BWD_DATA_ALGO_GEMM;
    case Convolution::AlgorithmBwdData::DIRECT:
      return MUDNN_CONVOLUTION_BWD_DATA_ALGO_DIRECT;
    default:
      return MUDNN_CONVOLUTION_BWD_DATA_ALGO_IMPLICIT_GEMM;
  }
}

Convolution::AlgorithmBwdData FromMudnnBwdDataAlgo(
    mudnnConvolutionBwdDataAlgo_t algo) {
  switch (algo) {
    case MUDNN_CONVOLUTION_BWD_DATA_ALGO_IMPLICIT_GEMM:
      return Convolution::AlgorithmBwdData::IMPLICIT_GEMM;
    case MUDNN_CONVOLUTION_BWD_DATA_ALGO_WINOGRAD_NONFUSED:
      return Convolution::AlgorithmBwdData::WINOGRAD_NONFUSED;
    case MUDNN_CONVOLUTION_BWD_DATA_ALGO_GEMM:
      return Convolution::AlgorithmBwdData::GEMM;
    case MUDNN_CONVOLUTION_BWD_DATA_ALGO_DIRECT:
      return Convolution::AlgorithmBwdData::DIRECT;
    default:
      return Convolution::AlgorithmBwdData::IMPLICIT_GEMM;
  }
}

mudnnConvolutionBwdFilterAlgo_t ToMudnnBwdFilterAlgo(
    Convolution::AlgorithmBwdFilter algo) {
  switch (algo) {
    case Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM:
      return MUDNN_CONVOLUTION_BWD_FILTER_ALGO_IMPLICIT_GEMM;
    case Convolution::AlgorithmBwdFilter::WINOGRAD_NONFUSED:
      return MUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED;
    case Convolution::AlgorithmBwdFilter::GEMM:
      return MUDNN_CONVOLUTION_BWD_FILTER_ALGO_GEMM;
    case Convolution::AlgorithmBwdFilter::DIRECT:
      return MUDNN_CONVOLUTION_BWD_FILTER_ALGO_DIRECT;
    default:
      return MUDNN_CONVOLUTION_BWD_FILTER_ALGO_IMPLICIT_GEMM;
  }
}

Convolution::AlgorithmBwdFilter FromMudnnBwdFilterAlgo(
    mudnnConvolutionBwdFilterAlgo_t algo) {
  switch (algo) {
    case MUDNN_CONVOLUTION_BWD_FILTER_ALGO_IMPLICIT_GEMM:
      return Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM;
    case MUDNN_CONVOLUTION_BWD_FILTER_ALGO_WINOGRAD_NONFUSED:
      return Convolution::AlgorithmBwdFilter::WINOGRAD_NONFUSED;
    case MUDNN_CONVOLUTION_BWD_FILTER_ALGO_GEMM:
      return Convolution::AlgorithmBwdFilter::GEMM;
    case MUDNN_CONVOLUTION_BWD_FILTER_ALGO_DIRECT:
      return Convolution::AlgorithmBwdFilter::DIRECT;
    default:
      return Convolution::AlgorithmBwdFilter::IMPLICIT_GEMM;
  }
}

mudnnActivationMode_t ToMudnnActivationMode(
    Convolution::FusedActivationDesc::Mode mode) {
  switch (mode) {
    case Convolution::FusedActivationDesc::Mode::IDENTITY:
      return MUDNN_ACTIVATION_IDENTITY;
    case Convolution::FusedActivationDesc::Mode::SIGMOID:
      return MUDNN_ACTIVATION_SIGMOID;
    case Convolution::FusedActivationDesc::Mode::RELU:
      return MUDNN_ACTIVATION_RELU;
    case Convolution::FusedActivationDesc::Mode::TANH:
      return MUDNN_ACTIVATION_TANH;
    case Convolution::FusedActivationDesc::Mode::CLIPPED_RELU:
      return MUDNN_ACTIVATION_CLIPPED_RELU;
    case Convolution::FusedActivationDesc::Mode::ELU:
      return MUDNN_ACTIVATION_ELU;
    case Convolution::FusedActivationDesc::Mode::SILU:
      return MUDNN_ACTIVATION_SILU;
    default:
      return MUDNN_ACTIVATION_IDENTITY;
  }
}

mudnnStatus_t SetGroupsImpl(Convolution& conv, int groups) {
  return mudnnSetConvolutionGroupCount(conv.Desc(), groups);
}

mudnnStatus_t SetNdInfoImpl(
    Convolution& conv,
    int length,
    const int* pad,
    const int* stride,
    const int* dilation) {
  return mudnnSetConvolutionNdDescriptor(
      conv.Desc(),
      length,
      const_cast<int*>(pad),
      const_cast<int*>(stride),
      const_cast<int*>(dilation),
      MUDNN_CROSS_CORRELATION,
      MUDNN_DATA_FLOAT);
}

mudnnStatus_t SetComputeModeImpl(Convolution& conv, ComputeMode mode) {
  return mudnnSetConvolutionMathType(conv.Desc(), mode);
}

} // anonymous namespace

mudnnFilterDescriptor_t Convolution::FilterDescFor(
    const muTensor& filter) const {
  if (!filter_desc_) {
    filter_desc_ = std::make_unique<FilterDesc>();
    BuildFilterDesc(filter_desc_->Desc(), filter);
  }
  return filter_desc_->Desc();
}

mudnnStatus_t Convolution::SetGroups(int groups) {
  return SetGroupsImpl(*this, groups);
}

mudnnStatus_t Convolution::SetNdInfo(
    std::initializer_list<int> pad,
    std::initializer_list<int> stride,
    std::initializer_list<int> dilation) {
  return SetNdInfoImpl(
      *this,
      static_cast<int>(pad.size()),
      pad.begin(),
      stride.begin(),
      dilation.begin());
}

mudnnStatus_t Convolution::SetNdInfo(
    int length,
    const int* pad,
    const int* stride,
    const int* dilation) {
  return SetNdInfoImpl(*this, length, pad, stride, dilation);
}

mudnnStatus_t Convolution::SetComputeMode(ComputeMode mode) {
  return SetComputeModeImpl(*this, mode);
}

mudnnStatus_t Convolution::GetRecommendForwardAlgorithm(
    muHandle& h,
    Algorithm& algo,
    const muTensor& out,
    const muTensor& in,
    const muTensor& filter) {
  in.Build();
  out.Build();

  auto fdesc = FilterDescFor(filter);

  int returned = 0;
  mudnnConvolutionFwdAlgoPerf_t perf;
  auto status = mudnnGetConvolutionForwardAlgorithm_v7(
      h, in.Desc(), fdesc, Desc(), out.Desc(), 1, &returned, &perf);
  if (status == MUDNN_STATUS_SUCCESS && returned > 0) {
    algo = FromMudnnFwdAlgo(perf.algo);
  }
  return status;
}

mudnnStatus_t Convolution::GetRecommendBackwardDataAlgorithm(
    muHandle& h,
    AlgorithmBwdData& algo,
    const muTensor& gin,
    const muTensor& gout,
    const muTensor& filter) {
  gin.Build();
  gout.Build();

  auto fdesc = FilterDescFor(filter);

  int returned = 0;
  mudnnConvolutionBwdDataAlgoPerf_t perf;
  auto status = mudnnGetConvolutionBackwardDataAlgorithm_v7(
      h, fdesc, gout.Desc(), Desc(), gin.Desc(), 1, &returned, &perf);
  if (status == MUDNN_STATUS_SUCCESS && returned > 0) {
    algo = FromMudnnBwdDataAlgo(perf.algo);
  }
  return status;
}

mudnnStatus_t Convolution::GetRecommendBackwardFilterAlgorithm(
    muHandle& h,
    AlgorithmBwdFilter& algo,
    const muTensor& gw,
    const muTensor& in,
    const muTensor& gout) {
  in.Build();
  gout.Build();

  auto fdesc = FilterDescFor(gw);

  int returned = 0;
  mudnnConvolutionBwdFilterAlgoPerf_t perf;
  auto status = mudnnGetConvolutionBackwardFilterAlgorithm_v7(
      h, in.Desc(), gout.Desc(), Desc(), fdesc, 1, &returned, &perf);
  if (status == MUDNN_STATUS_SUCCESS && returned > 0) {
    algo = FromMudnnBwdFilterAlgo(perf.algo);
  }
  return status;
}

mudnnStatus_t Convolution::Run(
    muHandle& h,
    muTensor& out,
    const muTensor& data,
    const muTensor& filter,
    Algorithm algo,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  data.Build();

  auto fdesc = FilterDescFor(filter);

  auto c_algo = ToMudnnFwdAlgo(algo);

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetConvolutionForwardWorkspaceSize(
          h, data.Desc(), fdesc, Desc(), out.Desc(), c_algo, &ws_size),
      "mudnnGetConvolutionForwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  float alpha = 1.0f, beta = 0.0f;
  return mudnnConvolutionForward(
      h,
      &alpha,
      MUDNN_UNPACK_TENSOR(data),
      fdesc,
      filter.DataPtr(),
      Desc(),
      c_algo,
      ws.get(),
      ws_size,
      &beta,
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t Convolution::RunFusion(
    muHandle& h,
    muTensor& out,
    const muTensor& data,
    const muTensor& filter,
    const muTensor& bias,
    const muTensor& add,
    const FusedActivationDesc& act,
    Algorithm algo,
    const MemoryMaintainer& maintainer) const {
  out.Build();
  data.Build();
  bias.Build();
  add.Build();

  auto fdesc = FilterDescFor(filter);

  ActivationDescGuard adesc;
  CHECK_MUDNN_STATUS(
      mudnnSetActivationDescriptor(
          adesc.Desc(),
          ToMudnnActivationMode(act.mode),
          MUDNN_NOT_PROPAGATE_NAN,
          act.params[0]),
      "mudnnSetActivationDescriptor");

  auto c_algo = ToMudnnFwdAlgo(algo);

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetConvolutionForwardWorkspaceSize(
          h, data.Desc(), fdesc, Desc(), out.Desc(), c_algo, &ws_size),
      "mudnnGetConvolutionForwardWorkspaceSize");

  auto ws = maintainer(ws_size);
  float alpha1 = 1.0f, alpha2 = 0.0f;
  return mudnnConvolutionBiasActivationForward(
      h,
      &alpha1,
      MUDNN_UNPACK_TENSOR(data),
      fdesc,
      filter.DataPtr(),
      Desc(),
      c_algo,
      ws.get(),
      ws_size,
      &alpha2,
      MUDNN_UNPACK_TENSOR(add),
      MUDNN_UNPACK_TENSOR(bias),
      adesc.Desc(),
      MUDNN_UNPACK_TENSOR(out));
}

mudnnStatus_t Convolution::RunBwdData(
    muHandle& h,
    muTensor& gin,
    const muTensor& gout,
    const muTensor& filter,
    AlgorithmBwdData algo,
    const MemoryMaintainer& maintainer) const {
  gin.Build();
  gout.Build();

  auto fdesc = FilterDescFor(filter);

  auto c_algo = ToMudnnBwdDataAlgo(algo);

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetConvolutionBackwardDataWorkspaceSize(
          h, fdesc, gout.Desc(), Desc(), gin.Desc(), c_algo, &ws_size),
      "mudnnGetConvolutionBackwardDataWorkspaceSize");

  auto ws = maintainer(ws_size);
  float alpha = 1.0f, beta = 0.0f;
  return mudnnConvolutionBackwardData(
      h,
      &alpha,
      fdesc,
      filter.DataPtr(),
      MUDNN_UNPACK_TENSOR(gout),
      Desc(),
      c_algo,
      ws.get(),
      ws_size,
      &beta,
      MUDNN_UNPACK_TENSOR(gin));
}

mudnnStatus_t Convolution::RunBwdFilter(
    muHandle& h,
    muTensor& gw,
    const muTensor& in,
    const muTensor& gout,
    AlgorithmBwdFilter algo,
    const MemoryMaintainer& maintainer) const {
  in.Build();
  gout.Build();

  auto fdesc = FilterDescFor(gw);

  auto c_algo = ToMudnnBwdFilterAlgo(algo);

  size_t ws_size = 0;
  CHECK_MUDNN_STATUS(
      mudnnGetConvolutionBackwardFilterWorkspaceSize(
          h, in.Desc(), gout.Desc(), Desc(), fdesc, c_algo, &ws_size),
      "mudnnGetConvolutionBackwardFilterWorkspaceSize");

  auto ws = maintainer(ws_size);
  float alpha = 1.0f, beta = 0.0f;
  return mudnnConvolutionBackwardFilter(
      h,
      &alpha,
      MUDNN_UNPACK_TENSOR(in),
      MUDNN_UNPACK_TENSOR(gout),
      Desc(),
      c_algo,
      ws.get(),
      ws_size,
      &beta,
      fdesc,
      gw.DataPtr());
}

mudnnStatus_t Convolution::GetForwardWorkspaceSize(
    muHandle& h,
    size_t& size_in_bytes,
    const muTensor& out,
    const muTensor& data,
    const muTensor& filter,
    const Algorithm& algo) const {
  out.Build();
  data.Build();

  auto fdesc = FilterDescFor(filter);

  return mudnnGetConvolutionForwardWorkspaceSize(
      h,
      data.Desc(),
      fdesc,
      Desc(),
      out.Desc(),
      ToMudnnFwdAlgo(algo),
      &size_in_bytes);
}

mudnnStatus_t Convolution::GetBackwardDataWorkspaceSize(
    muHandle& h,
    size_t& size_in_bytes,
    const muTensor& gin,
    const muTensor& gout,
    const muTensor& filter,
    const AlgorithmBwdData& algo) const {
  gin.Build();
  gout.Build();

  auto fdesc = FilterDescFor(filter);

  return mudnnGetConvolutionBackwardDataWorkspaceSize(
      h,
      fdesc,
      gout.Desc(),
      Desc(),
      gin.Desc(),
      ToMudnnBwdDataAlgo(algo),
      &size_in_bytes);
}

mudnnStatus_t Convolution::GetBackwardFilterWorkspaceSize(
    muHandle& h,
    size_t& size_in_bytes,
    const muTensor& gw,
    const muTensor& in,
    const muTensor& gout,
    const AlgorithmBwdFilter& algo) const {
  in.Build();
  gout.Build();

  auto fdesc = FilterDescFor(gw);

  return mudnnGetConvolutionBackwardFilterWorkspaceSize(
      h,
      in.Desc(),
      gout.Desc(),
      Desc(),
      fdesc,
      ToMudnnBwdFilterAlgo(algo),
      &size_in_bytes);
}

#endif // TORCH_MUSA_USE_MUDNN_C_API
} // namespace at::musa
