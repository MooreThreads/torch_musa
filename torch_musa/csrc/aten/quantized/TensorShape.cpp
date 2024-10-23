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
#include "torch_musa/csrc/aten/utils/Utils.h"
#include "torch_musa/csrc/core/MUSAGuard.h"
#include "torch_musa/csrc/utils/register_wrapper.h"

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
  auto quantizer = at::get_qtensorimpl(self)->quantizer();
  SymDimVector sizes;
  SymDimVector strides;
  const auto ndim = self.dim();
  auto mask = dims.has_value()
      ? dim_list_to_bitset(dims, self.dim())
      : std::bitset<dim_bitset_size>((1ull << self.dim()) - 1);
  std::tie(sizes, strides) = InferSqueezeGeometry(self, mask);
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
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
    quantizer = at::make_per_channel_affine_quantizer(
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
  auto quantizer = at::get_qtensorimpl(self)->quantizer();
  if (quantizer->qscheme() == QScheme::PER_CHANNEL_AFFINE) {
    const auto* per_channel_quantizer =
        static_cast<at::PerChannelAffineQuantizer*>(quantizer.get());
    auto axis = per_channel_quantizer->axis();
    if (axis >= dim) {
      axis += 1;
    }
    quantizer = at::make_per_channel_affine_quantizer(
        per_channel_quantizer->scales(),
        per_channel_quantizer->zero_points(),
        axis,
        quantizer->scalar_type());
  }
  return MakeQtensor(
      self, geometry.sizes, geometry.strides, std::move(quantizer));
}

bool isValidQuantizationScheme(const Tensor& t) {
  const auto qtype = t.qscheme();
  return (qtype == kPerTensorAffine) || (qtype == kPerTensorSymmetric);
}

bool allInputsSharingQParams(const MaterializedITensorListRef& qxs) {
  bool is_valid = true;
  for (const auto i : c10::irange(1, qxs.size())) {
    is_valid |= qxs[0].get().is_quantized();
    is_valid |= qxs[i].get().is_quantized() == qxs[0].get().is_quantized();
    is_valid |= qxs[i].get().qscheme() == qxs[0].get().qscheme();
    is_valid |= qxs[i].get().dtype() == qxs[0].get().dtype();
    if (qxs[0].get().qscheme() == kPerTensorAffine) {
      is_valid |= qxs[i].get().q_scale() == qxs[0].get().q_scale();
      is_valid |= qxs[i].get().q_zero_point() == qxs[0].get().q_zero_point();
    } else if (qxs[0].get().qscheme() == kPerChannelAffine) {
      is_valid |= qxs[i].get().q_per_channel_scales().equal(
          qxs[0].get().q_per_channel_scales());
      is_valid |= qxs[i].get().q_per_channel_zero_points().equal(
          qxs[0].get().q_per_channel_zero_points());
    } else {
      TORCH_CHECK(false, "Unsupported quantization scheme");
    }
  }
  return is_valid;
}

// Quantized concatenation
Tensor CatQuantizedMusaImpl(
    const MaterializedITensorListRef& qxs,
    int64_t dim,
    double scale,
    int64_t zero_point) {
  const Tensor& qx0 = qxs[0].get();
  if (qxs.size() == 1) {
    return qx0;
  }

  float scale_m = static_cast<float>(scale);
  unsigned int zero_point_m = static_cast<unsigned int>(zero_point);
  auto dtype = qx0.scalar_type();

  // rt_tensors holds contiguous cat candidates
  std::vector<Tensor> rt_tensors;
  int elements = 0;
  rt_tensors.reserve(qxs.size());
  // TODO(@fan.mo) this nhwc could be wrapped into utils functions
  bool is_nhwc =
      qx0.dim() == 4 && qx0.is_contiguous(at::MemoryFormat::ChannelsLast);

  std::vector<int64_t> output_shape(
      qx0.sizes().data(), qx0.sizes().data() + qx0.dim());
  output_shape[dim] = 0;

  for (int idx = 0; idx < qxs.size(); ++idx) {
    if (qxs[idx].get().numel() > 0) {
      Tensor qx = qxs[idx].get();
      output_shape[dim] += qx.size(dim);
      if (is_nhwc) {
        qx = qx.permute({0, 2, 3, 1});
      }
      rt_tensors.emplace_back(qx.contiguous());
      elements++;
    }
  }

  // mu_tensors holds muTensors that create from rt_tensors
  std::vector<at::musa::muTensor> mu_tensors;
  mu_tensors.reserve(elements);

  // set output tensor's shape and create muTensor
  for (const Tensor& qx : rt_tensors) {
    at::musa::muTensor mu_qx = at::musa::CreateMUTensor(qx);
    if (is_nhwc) {
      mu_qx.SetFormat(muTensor::Format::NHWC);
    }
    // set q-scale and q-zero-point
    SetMudnnQuantizationInfo(mu_qx, qx.q_scale(), qx.q_zero_point());
    mu_tensors.emplace_back(mu_qx);
  }

  // create output tensor
  Tensor output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(dtype),
      scale,
      zero_point,
      qx0.suggest_memory_format());
  at::musa::muTensor out_;
  if (is_nhwc) {
    output = output.permute({0, 2, 3, 1});
    out_ = at::musa::CreateMUTensor(output);
    out_.SetFormat(muTensor::Format::NHWC);
    // set dim to match nhwc format tensor
    if (dim == 1) {
      dim = 3;
    } else {
      dim = dim == 0 ? dim : dim - 1;
    }
  } else {
    out_ = at::musa::CreateMUTensor(output);
  }

  CHECK_MUDNN_STATUS(
      out_.SetQuantizationInfo(1, &scale_m, &zero_point_m),
      "Set quantization info");

  // run mudnn op
  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Concat op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "Set concat axis");
  CHECK_MUDNN_STATUS(
      op.Run(h, out_, elements, mu_tensors.data()), "Run concat");

  if (is_nhwc) {
    // permute to NCHW shape
    output = output.permute({0, 3, 1, 2});
  }
  return output;
}

Tensor CatQuantizedMusa(const ITensorListRef& qxs, int64_t dim) {
  const auto& materialized = qxs.materialize();
  TORCH_CHECK(
      isValidQuantizationScheme(materialized[0]),
      "Only per-tensor quantization is supported in `cat`");
  TORCH_CHECK(
      allInputsSharingQParams(materialized),
      "All inputs must have the same quantization parameters");
  at::native::check_cat_no_zero_dim(materialized);
  dim = legacy_cat_wrap_dim(dim, materialized);

  double _scale = materialized[0].get().q_scale();
  int64_t _zero_point = materialized[0].get().q_zero_point();

  return CatQuantizedMusaImpl(materialized, dim, _scale, _zero_point);
}

Tensor& CatOutQuantizedMusa(
    const ITensorListRef& qxs,
    int64_t dim,
    Tensor& out) {
  const auto& materialized = qxs.materialize();
  TORCH_CHECK(
      isValidQuantizationScheme(materialized[0]),
      "Only per-tensor quantization is supported in `cat`");
  TORCH_CHECK(
      isValidQuantizationScheme(out),
      "Only per-tensor quantization is supported in `cat`");
  at::native::check_cat_no_zero_dim(materialized);
  dim = legacy_cat_wrap_dim(dim, materialized);

  auto out_ = CatQuantizedMusaImpl(
      materialized, dim, out.q_scale(), out.q_zero_point());
  out.copy_(out_);

  return out;
}

/* Quantized concatenation (under quantized schema)
 *
 * Note: This function directly concat quantized tensors
 */
Tensor QuantizedCatOut(const c10::List<Tensor>& qxs, int64_t dim, Tensor out) {
  TORCH_CHECK(
      out.qscheme() == kPerTensorAffine || out.qscheme() == kPerTensorSymmetric,
      "Quantized cat only supports quantize-per-tensor");
  TORCH_CHECK(
      out.scalar_type() == c10::kQInt8,
      "Quantize cat out should be qint8 dtype");
  if (dim >= 0) {
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong dim: ", dim);
  } else {
    dim = dim + qxs[0].dim();
    TORCH_CHECK(dim < qxs[0].dim(), "Wrong din: ", dim);
  }
  float scale = static_cast<float>(out.q_scale());
  unsigned int zero_point = static_cast<unsigned int>(out.q_zero_point());

  // rt_tensors holds contiguous cat candidates
  std::vector<Tensor> rt_tensors;
  int elements = 0;
  rt_tensors.reserve(qxs.size());
  for (int idx = 0; idx < qxs.size(); ++idx) {
    if (qxs[idx].numel() > 0) {
      rt_tensors.emplace_back(qxs[idx].contiguous());
      elements++;
    }
  }

  // mu_tensors holds muTensors that create from rt_tensors
  std::vector<at::musa::muTensor> mu_tensors;
  mu_tensors.reserve(elements);

  // create muTensor
  for (const Tensor& qx : rt_tensors) {
    // set q-scale and q-zero-point
    at::musa::muTensor mu_qx = at::musa::CreateMUTensor(qx);
    SetMudnnQuantizationInfo(mu_qx, qx.q_scale(), qx.q_zero_point());
    mu_tensors.emplace_back(mu_qx);
  }

  at::musa::muTensor out_m = at::musa::CreateMUTensor(out);
  SetMudnnQuantizationInfo(out_m, out.q_scale(), out.q_zero_point());

  // run mudnn op
  at::musa::muHandle& h = at::GetMudnnHandle();
  ::musa::dnn::Concat op;
  CHECK_MUDNN_STATUS(op.SetAxis(dim), "Set concat axis");
  CHECK_MUDNN_STATUS(
      op.Run(h, out_m, elements, mu_tensors.data()), "Run concat");

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
  auto dtype = qx0.scalar_type();

  std::vector<int64_t> output_shape(
      qx0.sizes().data(), qx0.sizes().data() + qx0.dim());
  output_shape[dim] = 0;

  for (const Tensor& qx : qxs) {
    output_shape[dim] += qx.size(dim);
  }

  Tensor output = at::_empty_affine_quantized(
      output_shape,
      at::device(at::kPrivateUse1).dtype(dtype),
      scale_,
      zero_point_,
      c10::MemoryFormat::Contiguous);

  output = QuantizedCatOut(qxs, dim, output);

  return output;
}

TORCH_LIBRARY_IMPL(quantized, QuantizedPrivateUse1, m) {
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat"), TORCH_FN(QuantizedCat));
  m.impl(TORCH_SELECTIVE_NAME("quantized::cat_out"), TORCH_FN(QuantizedCatOut));
}

} // namespace musa
} // namespace at
