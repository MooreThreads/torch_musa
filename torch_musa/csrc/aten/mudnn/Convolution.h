#ifndef TORCH_MUSA_CSRC_ATEN_MUDNN_CONVOLUTION_H_
#define TORCH_MUSA_CSRC_ATEN_MUDNN_CONVOLUTION_H_

#include <array>
#include <initializer_list>
#include <memory>

#include <mudnn.h>

#ifdef TORCH_MUSA_USE_MUDNN_C_API

#include "torch_musa/csrc/aten/mudnn/Handle.h"
#include "torch_musa/csrc/aten/mudnn/Tensor.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at::musa {

class Convolution : public Descriptor<
                        mudnnConvolutionStruct,
                        mudnnCreateConvolutionDescriptor,
                        mudnnDestroyConvolutionDescriptor> {
 private:
  using FilterDesc = Descriptor<
      mudnnFilterStruct,
      mudnnCreateFilterDescriptor,
      mudnnDestroyFilterDescriptor>;

  mutable std::unique_ptr<FilterDesc> filter_desc_;

  mudnnFilterDescriptor_t FilterDescFor(const muTensor& filter) const;

 public:
  enum class Algorithm {
    IMPLICIT_GEMM,
    DIRECT,
    WINOGRAD_NONFUSED,
    GEMM,
  };

  enum class AlgorithmBwdData {
    IMPLICIT_GEMM,
    WINOGRAD_NONFUSED,
    GEMM,
    DIRECT,
  };

  enum class AlgorithmBwdFilter {
    IMPLICIT_GEMM,
    WINOGRAD_NONFUSED,
    GEMM,
    DIRECT,
  };

  struct FusedActivationDesc {
    enum class Mode {
      IDENTITY,
      SIGMOID,
      RELU,
      TANH,
      CLIPPED_RELU,
      ELU,
      HARD_SWISH,
      SILU,
    };

    mudnnStatus_t SetMode(Mode m) {
      mode = m;
      return MUDNN_STATUS_SUCCESS;
    }

    mudnnStatus_t SetCoef(double a, double b, double g) {
      params = {a, b, g};
      return MUDNN_STATUS_SUCCESS;
    }

    Mode mode = Mode::IDENTITY;
    std::array<double, 3> params = {0.0, 0.0, 0.0};
  };

  mudnnStatus_t SetGroups(int groups);

  mudnnStatus_t SetNdInfo(
      std::initializer_list<int> pad,
      std::initializer_list<int> stride,
      std::initializer_list<int> dilation);

  mudnnStatus_t SetNdInfo(
      int length,
      const int* pad,
      const int* stride,
      const int* dilation);

  mudnnStatus_t SetComputeMode(ComputeMode mode);

  mudnnStatus_t GetRecommendForwardAlgorithm(
      muHandle& h,
      Algorithm& algo,
      const muTensor& out,
      const muTensor& in,
      const muTensor& filter);

  mudnnStatus_t GetRecommendBackwardDataAlgorithm(
      muHandle& h,
      AlgorithmBwdData& algo,
      const muTensor& gin,
      const muTensor& gout,
      const muTensor& filter);

  mudnnStatus_t GetRecommendBackwardFilterAlgorithm(
      muHandle& h,
      AlgorithmBwdFilter& algo,
      const muTensor& gw,
      const muTensor& in,
      const muTensor& gout);

  mudnnStatus_t Run(
      muHandle& h,
      muTensor& out,
      const muTensor& data,
      const muTensor& filter,
      Algorithm algo,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunFusion(
      muHandle& h,
      muTensor& out,
      const muTensor& data,
      const muTensor& filter,
      const muTensor& bias,
      const muTensor& add,
      const FusedActivationDesc& act,
      Algorithm algo,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunBwdData(
      muHandle& h,
      muTensor& gin,
      const muTensor& gout,
      const muTensor& filter,
      AlgorithmBwdData algo,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t RunBwdFilter(
      muHandle& h,
      muTensor& gw,
      const muTensor& in,
      const muTensor& gout,
      AlgorithmBwdFilter algo,
      const MemoryMaintainer& maintainer) const;

  mudnnStatus_t GetForwardWorkspaceSize(
      muHandle& h,
      size_t& size_in_bytes,
      const muTensor& out,
      const muTensor& data,
      const muTensor& filter,
      const Algorithm& algo) const;

  mudnnStatus_t GetBackwardDataWorkspaceSize(
      muHandle& h,
      size_t& size_in_bytes,
      const muTensor& gin,
      const muTensor& gout,
      const muTensor& filter,
      const AlgorithmBwdData& algo) const;

  mudnnStatus_t GetBackwardFilterWorkspaceSize(
      muHandle& h,
      size_t& size_in_bytes,
      const muTensor& gw,
      const muTensor& in,
      const muTensor& gout,
      const AlgorithmBwdFilter& algo) const;
};

} // namespace at::musa

namespace musa::dnn {
using Convolution = class at::musa::Convolution;
} // namespace musa::dnn

#else

namespace at::musa {
using ::musa::dnn::Convolution;
} // namespace at::musa

#endif // TORCH_MUSA_USE_MUDNN_C_API
#endif // TORCH_MUSA_CSRC_ATEN_MUDNN_CONVOLUTION_H_
