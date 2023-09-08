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

#include <mudnn.h>

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
  const auto& materialized = qxs.materialize();
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
    TORCH_CHECK(false, "QScheme not supported by cat", toString(x_qscheme));
  }
  return output;
}

Tensor QuantizedCatOut(const c10::List<Tensor>& qxs, int64_t dim, Tensor out) {
  TORCH_CHECK(
      out.qscheme() == kPerTensorAffine || out.qscheme() == kPerTensorSymmetric,
      "Quantized cat only supports quantize-per-tensor");
  if (dim >= 0) {
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong dim: ", dim);
  } else {
    dim = dim + qxs[0].dim();
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong din: ", dim);
  }
  float scale = static_cast<float>(out.q_scale());
  unsigned int zero_point = static_cast<unsigned int>(out.q_zero_point());

  std::vector<at::musa::muTensor> mu_tensors;
  mu_tensors.reserve(qxs.size());

  for (const Tensor& qx : qxs) {
    at::musa::muTensor mu_qx;
    if (!qx.is_contiguous()) {
      mu_qx = at::musa::CreateMUTensor(qx.contiguous());
    } else {
      mu_qx = at::musa::CreateMUTensor(qx);
    }
    CHECK_MUDNN_STATUS(
        mu_qx.SetQuantizationInfo(1, &scale, &zero_point),
        "Set quantization info");
    CHECK_MUDNN_STATUS(
        mu_qx.SetType(at::musa::muTensor::Type::QINT8),
        "Set quantization dtype");
    mu_tensors.emplace_back(mu_qx);
  }
  at::musa::muTensor out_ = at::musa::CreateMUTensor(out);
  CHECK_MUDNN_STATUS(
      out_.SetQuantizationInfo(1, &scale, &zero_point),
      "Set quantization info");
  CHECK_MUDNN_STATUS(
      out_.SetType(at::musa::muTensor::Type::QINT8), "Set quantization dtype");

  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Concat op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "Set concat axis");
  CHECK_MUDNN_STATUS(
      op.Run(h, out_, qxs.size(), mu_tensors.data()), "Run concat");

  return out;
}

Tensor QuantizedCat(
    const c10::List<Tensor>& qxs,
    int64_t dim,
    c10::optional<double> scale,
    c10::optional<int64_t> zero_point) {
  const at::Tensor& qx0 = qxs[0];
  if (dim >= 0) {
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong dim: ", dim);
  } else {
    dim = dim + qxs[0].dim();
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong dim: ", dim);
  }
  TORCH_CHECK(
      qx0.qscheme() == kPerTensorAffine || qx0.qscheme() == kPerTensorSymmetric,
      "Quantized cat only supports quantize-per-tensor");
  double scale_ = scale.has_value() ? scale.value() : qx0.q_scale();
  int64_t zero_point_ =
      zero_point.has_value() ? zero_point.value() : qx0.q_zero_point();
  float scale_m = static_cast<float>(scale_);
  unsigned int zero_point_m = static_cast<unsigned int>(zero_point_);
  auto dtype = qx0.scalar_type();

  std::vector<at::musa::muTensor> mu_tensors;
  mu_tensors.reserve(qxs.size());

  std::vector<int64_t> output_shape(
      qx0.sizes().data(), qx0.sizes().data() + qx0.dim());
  output_shape[dim] = 0;

  for (const Tensor& qx : qxs) {
    at::musa::muTensor mu_qx;
    output_shape[dim] += qx.size(dim);
    if (!qx.is_contiguous()) {
      mu_qx = at::musa::CreateMUTensor(qx.contiguous());
    } else {
      mu_qx = at::musa::CreateMUTensor(qx);
    }
    CHECK_MUDNN_STATUS(
        mu_qx.SetQuantizationInfo(1, &scale_m, &zero_point_m),
        "Set quantization info");
    CHECK_MUDNN_STATUS(
        mu_qx.SetType(at::musa::muTensor::Type::QINT8),
        "Set quantization dtype");
    mu_tensors.emplace_back(mu_qx);
  }

  Tensor output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(dtype),
      scale_,
      zero_point_,
      c10::MemoryFormat::Contiguous);

  at::musa::muTensor out_ = at::musa::CreateMUTensor(output);
  CHECK_MUDNN_STATUS(
      out_.SetQuantizationInfo(1, &scale_m, &zero_point_m),
      "Set quantization info");
  CHECK_MUDNN_STATUS(
      out_.SetType(at::musa::muTensor::Type::QINT8), "Set quantization dtype");

  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Concat op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "Set concat axis");
  CHECK_MUDNN_STATUS(
      op.Run(h, out_, qxs.size(), mu_tensors.data()), "Run concat");

  return output;
}

TORCH_LIBRARY_IMPL(aten, QuantizedPrivateUse1, m) {
  m.impl("squeeze", TORCH_FN(SqueezeQuantized));
  m.impl("squeeze.dim", TORCH_FN(SqueezeQuantizedDim));
  m.impl("squeeze.dims", TORCH_FN(SqueezeQuantizedDims));
  m.impl("unsqueeze", TORCH_FN(UnsqueezeQuantized));
  m.impl("cat", TORCH_FN(CatQuantized));
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(QuantizedCat));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(QuantizedCatOut));
}

} // namespace musa
} // namespace at
