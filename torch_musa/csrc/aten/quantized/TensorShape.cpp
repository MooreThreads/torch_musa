#include <ATen/NamedTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/Tensor.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorShape.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/util/irange.h>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include <torch/library.h>
#include "torch_musa/csrc/aten/quantized/QTensor.h"
#include "torch_musa/csrc/aten/quantized/Quantizer.h"
#include "torch_musa/csrc/core/MUSAGuard.h"

namespace at {
namespace musa {
namespace {
Tensor MakeQtensor(
    const Tensor& self,
    IntArrayRef size,
    IntArrayRef stride,
    QuantizerPtr quantizer) {
  auto result = at::detail::make_tensor<QTensorImpl>(
      c10::TensorImpl::VIEW,
      Storage(self.storage()),
      self.key_set(),
      self.dtype(),
      quantizer);
  at::native::setStrided(result, size, stride, self.storage_offset());
  return result;
}

std::tuple<SymDimVector, SymDimVector> InferSqueezeGeometry(
    const Tensor& tensor,
    std::bitset<dim_bitset_size> dim_mask) {
  const auto ndim = tensor.dim();
  const auto sym_sizes = tensor.sym_sizes();
  const auto sym_strides = tensor.sym_strides();

  SymDimVector out_sizes, out_strides;
  for (const auto d : c10::irange(ndim)) {
    if (!dim_mask.test(d) || sym_sizes[d] != 1) {
      out_sizes.push_back(sym_sizes[d]);
      out_strides.push_back(sym_strides[d]);
    }
  }
  return std::make_tuple(std::move(out_sizes), std::move(out_strides));
}

// Named type instead of a pair/tuple so that we can be sure to
// construct the vectors in place and get NRVO.
struct InferUnsqueezeGeometryResult {
  DimVector sizes;
  DimVector strides;
  InferUnsqueezeGeometryResult(
      IntArrayRef tensor_sizes,
      IntArrayRef tensor_strides)
      : sizes(tensor_sizes.begin(), tensor_sizes.end()),
        strides(tensor_strides.begin(), tensor_strides.end()) {}
};

InferUnsqueezeGeometryResult InferUnsqueezeGeometry(
    const Tensor& tensor,
    int64_t dim) {
  InferUnsqueezeGeometryResult result(tensor.sizes(), tensor.strides());
  int64_t new_stride =
      dim >= tensor.dim() ? 1 : result.sizes[dim] * result.strides[dim];
  result.sizes.insert(result.sizes.begin() + dim, 1);
  result.strides.insert(result.strides.begin() + dim, new_stride);

  return result;
}
} // namespace

// dim is present if squeezing a single dimension and absent if squeezing all
// dimensions
Tensor SqueezeQtensor(const Tensor& self, c10::OptionalIntArrayRef dims) {
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  SymDimVector sizes;
  SymDimVector strides;
  const auto ndim = self.dim();
  auto mask = dims.has_value()
      ? dim_list_to_bitset(dims, self.dim())
      : std::bitset<dim_bitset_size>((1ull << self.dim()) - 1);
  std::tie(sizes, strides) = InferSqueezeGeometry(self, mask);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::MusaPerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    int64_t shift = 0;
    for (const auto d : c10::irange(ndim)) {
      if (mask.test(d) && self.sizes()[d] == 1) {
        TORCH_CHECK(
            axis != d,
            "Squeeze is only possible on non-axis dimension for Per-Channel Quantized Tensors.");
        if (d < axis) {
          ++shift;
        }
      }
    }
    axis -= shift;
    quantizer = MakePerChannelAffineQuantizer(
        per_channel_quantizer->scales(),
        per_channel_quantizer->zero_points(),
        axis,
        quantizer->scalar_type());
  }
  // TODO(@fan.mo): quantized Tensor support for SymInt needs to be added but
  // basic building blocks are missing for now.
  auto result = MakeQtensor(
      self,
      C10_AS_INTARRAYREF_SLOW(sizes),
      C10_AS_INTARRAYREF_SLOW(strides),
      std::move(quantizer));
  auto maybe_outnames = namedinference::compute_squeeze_outnames(self, mask);
  namedinference::propagate_names_if_nonempty(result, maybe_outnames);
  return result;
}

Tensor SqueezeQuantized(const Tensor& self) {
  // DeviceGuard omitted
  return SqueezeQtensor(self, c10::nullopt);
}

Tensor SqueezeQuantizedDim(const Tensor& self, int64_t dim) {
  // DeviceGuard omitted
  return SqueezeQtensor(self, dim);
}

Tensor SqueezeQuantizedDims(const Tensor& self, IntArrayRef dim) {
  // DeviceGuard omitted
  return SqueezeQtensor(self, dim);
}

Tensor UnsqueezeQuantized(const Tensor& self, int64_t dim) {
  // DeviceGuard omitted
  dim = maybe_wrap_dim(dim, self.dim() + 1);
  auto geometry = InferUnsqueezeGeometry(self, dim);
  auto quantizer = at::GetQTensorImpl(self)->quantizer();
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::MusaPerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    quantizer = MakePerChannelAffineQuantizer(
        per_channel_quantizer->scales(),
        per_channel_quantizer->zero_points(),
        axis,
        quantizer->scalar_type());
  }
  return MakeQtensor(
      self, geometry.sizes, geometry.strides, std::move(quantizer));
}

Tensor CatQuantized(const ITensorListRef& qxs, int64_t dim) {
  c10::optional<Device> common_device = nullopt;
  (void)common_device; // Suppress unused variable warning
  c10::impl::check_and_update_common_device(
      common_device, qxs, "CatQuantized", "qxs");
  auto materialized = qxs.materialize();
  at::native::check_cat_no_zero_dim(materialized);
  dim = legacy_cat_wrap_dim(dim, materialized);

  auto input = materialized[0].get();
  const auto x_dtype = input.scalar_type();
  const auto x_qscheme = input.qscheme();

  std::vector<Tensor> xs;
  xs.reserve(qxs.size());
  for (const at::Tensor& qx : qxs) {
    TORCH_CHECK(x_dtype == qx.scalar_type(), "All dtypes must be the same.");
    TORCH_CHECK(
        x_qscheme == qx.qscheme(), "Quantization schemes must be the same.");
    xs.push_back(qx.dequantize());
  }
  Tensor output;
  const Tensor output_fp32 = at::cat(xs, dim);
  // return QuantizePerTensor(y, x_scale, x_zero_point, x_dtype);
  if (x_qscheme == kPerTensorAffine) {
    output = at::musa::QuantizePerTensor(
        output_fp32, input.q_scale(), input.q_zero_point(), x_dtype);
  } else if (
      x_qscheme == kPerChannelAffine ||
      x_qscheme == kPerChannelAffineFloatQParams) {
    output = at::musa::QuantizePerChannel(
        output_fp32,
        input.q_per_channel_scales(),
        input.q_per_channel_zero_points(),
        input.q_per_channel_axis(),
        x_dtype);
  } else {
    TORCH_CHECK(
        false,
        "QScheme not supported by upsample_nearest2d:",
        toString(x_qscheme));
  }
  return output;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("squeeze", TORCH_FN(SqueezeQuantized));
  m.impl("squeeze.dim", TORCH_FN(SqueezeQuantizedDim));
  m.impl("squeeze.dims", TORCH_FN(SqueezeQuantizedDims));
  m.impl("unsqueeze", TORCH_FN(UnsqueezeQuantized));
  m.impl("cat", TORCH_FN(CatQuantized));
}
} // namespace musa
} // namespace at
