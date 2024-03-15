#include <ATen/ATen.h>
#include <ATen/core/Array.h>
#include <ATen/core/Tensor.h>

#include <musa_fp16.h>
#include "torch_musa/csrc/aten/ops/musa/IndexUtils.muh"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {
constexpr int MAX_DIM = kMaxDim;

// if the non-null indices are not all adjacent, transpose self and indices
// together so that they're adjacent at the front
bool HasContiguousSubspace(
    int& indices_num,
    const std::vector<Tensor>& indices) {
  auto IsNull = [](const int& i) { return i == 0; };
  auto IsNotNull = [](const int& i) { return i == 1; };
  std::vector<int> inds;
  for (int i = 0; i < indices_num; ++i) {
    indices[i].numel() == 0 ? inds.emplace_back(0) : inds.emplace_back(1);
  }
  auto start = std::find_if(inds.begin(), inds.end(), IsNotNull);
  auto end = std::find_if(inds.rbegin(), inds.rend(), IsNotNull);
  auto it = std::find_if(start, end.base(), IsNull);
  return it == end.base();
}

void IndicesBroadCast(
    int num,
    const std::vector<Tensor>& indice,
    at::detail::Array<int, MAX_DIM>& bcast_shape,
    int& bcast_indice_ndim) {
  TORCH_CHECK(
      num <= MAX_DIM,
      "Tensor's ndim should be less than ",
      MAX_DIM + 1,
      " now");

  for (int i = 0; i < num; ++i) {
    if (indice[i].numel() == 0) {
      continue;
    }
    TORCH_CHECK(
        indice[i].dim() <= MAX_DIM,
        "Indice's ndim should be less than ",
        MAX_DIM + 1,
        " now");

    int cur_ndim = indice[i].dim();

    if (i == 0) {
      bcast_indice_ndim = cur_ndim;
      std::copy(
          indice[i].sizes().data(),
          indice[i].sizes().data() + cur_ndim,
          std::begin(bcast_shape.data));
    } else {
      int max_ndim =
          cur_ndim > bcast_indice_ndim ? cur_ndim : bcast_indice_ndim;
      std::vector<int> temp_shape(max_ndim);

      for (int j = max_ndim - 1; j >= 0; --j) {
        int offset = max_ndim - 1 - j;
        int dim_a = cur_ndim - 1 - offset;
        int dim_b = bcast_indice_ndim - 1 - offset;
        size_t size_a = dim_a >= 0 ? indice[i].size(dim_a) : 1;
        size_t size_b = dim_b >= 0 ? bcast_shape[dim_b] : 1;
        TORCH_CHECK(
            size_a == size_b || size_a == 1 || size_b == 1,
            "The indice tensor cannot be broadcast");
        temp_shape[j] = size_a == 1 ? size_b : size_a;
      }

      std::copy(
          temp_shape.begin(), temp_shape.end(), std::begin(bcast_shape.data));

      bcast_indice_ndim =
          bcast_indice_ndim < max_ndim ? max_ndim : bcast_indice_ndim;
    }
  }
}

void ValueBroadcastableCheck(
    const Tensor& v,
    at::detail::Array<int, MAX_DIM>& shape,
    int bcast_ndim) {
  int value_ndim = v.dim();
  TORCH_CHECK(
      value_ndim <= bcast_ndim, "value has larger ndim than target shape");
  for (int i = bcast_ndim - 1; i >= 0; --i) {
    int offset = bcast_ndim - 1 - i;
    int dim_a = bcast_ndim - 1 - offset;
    int dim_b = value_ndim - 1 - offset;
    size_t size_a = dim_a >= 0 ? shape[dim_a] : 1;
    size_t size_b = dim_b >= 0 ? v.sizes()[dim_b] : 1;
    TORCH_CHECK(
        size_a == size_b || size_a == 1 || size_b == 1,
        "Value tensor cannot be broadcast");
  }
}

void IndicesStrides(
    at::detail::Array<int, MAX_DIM * MAX_DIM>& stride_inds,
    at::detail::Array<int, MAX_DIM * MAX_DIM>& shape_inds,
    at::detail::Array<int, MAX_DIM>& bcast_indice_shape,
    at::detail::Array<int, MAX_DIM>& task_shape,
    int num,
    const std::vector<Tensor>& indice,
    int bcast_indice_ndim,
    int shape_ndim,
    int num_of_null_in_front) {
  // update strides of indices
  // eg, we have configuration: indice0: (3, 1), indice1: (2),
  // value: (3, 1, 5, 1), out: (3, 4, 5, 8), task_shape: (3, 2, 5, 8)
  // step0: calculate original strides of indices: (1, 1) and (1)
  // step1: calculate broadcast strides of indices: (1, 0) and (0, 1)
  // step2: fill zero to align with shape_ndim:
  // (1, 0, 0, 0) and (0, 1, 0, 0)

  // step0
  for (int i = 0; i < num; ++i) {
    for (int j = indice[i].dim() - 1; j >= 0; --j) {
      stride_inds[i * MAX_DIM + j] = indice[i].strides()[j];
    }
  }
  // step1
  for (int i = 0; i < num; ++i) {
    int ind_ndim = indice[i].dim();
    auto ind_dim = indice[i].sizes().vec();
    for (int j = bcast_indice_ndim - 1; j >= 0; --j) {
      int offset = bcast_indice_ndim - 1 - j;
      int dim_a = bcast_indice_ndim - 1 - offset;
      int dim_b = ind_ndim - 1 - offset;
      bool bcast_dim =
          (dim_b < 0 ||
           (bcast_indice_shape[dim_a] != 1 && ind_dim[dim_b] == 1));
      if (bcast_dim) {
        shape_inds[i * MAX_DIM + j] = 1;
        stride_inds[i * MAX_DIM + j] = 0;
      } else {
        stride_inds[i * MAX_DIM + j] = stride_inds[i * MAX_DIM + dim_b];
        shape_inds[i * MAX_DIM + j] = ind_dim[dim_b];
      }
    }
    for (int j = bcast_indice_ndim; j < shape_ndim; ++j) {
      shape_inds[i * MAX_DIM + j] = 1;
    }
  }
  // step2
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < shape_ndim; ++j) {
      if (task_shape[j] != 1 && shape_inds[i * MAX_DIM + j] == 1) {
        stride_inds[i * MAX_DIM + j] = 0;
      }
    }
  }
  // step3 (optional)
  if (num_of_null_in_front > 0) {
    for (int i = 0; i < num; ++i) {
      for (int j = MAX_DIM - 1; j >= num_of_null_in_front; --j) {
        stride_inds[i * MAX_DIM + j] =
            stride_inds[i * MAX_DIM + j - num_of_null_in_front];
      }
      for (int j = 0; j < num_of_null_in_front; ++j) {
        stride_inds[i * MAX_DIM + j] = 0;
      }
    }
  }
}

void InputStrides(
    at::detail::Array<int, MAX_DIM>& stride_in,
    const Tensor& in,
    int indices_num,
    const std::vector<Tensor>& indices,
    bool has_contiguous_subspace) {
  // update strides of input, treat input as MAX_DIM dimension
  // step0: calculate original strides of input
  // step1: if !has_contiguous_subspace, for example:
  // input_shape: (1, 2, 3, 4, 5, 6)
  // [indice0: (7, 1), indice1: (), indice2: (7), indice3: (), indice4: (7, 7)]
  // output_shape: (7, 7, 2, 4, 6)
  // we need to rearrange the strides for input from (720, 360, 120, 30, 6, 1)
  // to (720, 120, 6, 360, 30, 1)
  for (int i = 0; i < MAX_DIM; ++i) {
    stride_in[i] = 0;
  }
  for (int i = in.dim() - 1; i >= 0; --i) {
    stride_in[i] = in.strides()[i];
  }

  if (!has_contiguous_subspace) {
    std::vector<int> indices_jump;
    int idx = 0;
    for (int i = 0; i < indices_num; ++i) {
      if (indices[i].numel() == 0) {
        indices_jump.emplace_back(stride_in[i]);
      } else {
        stride_in[idx++] = stride_in[i];
      }
    }
    for (auto iter : indices_jump) {
      stride_in[idx++] = iter;
    }
    for (int i = indices_num; i < in.dim(); ++i) {
      stride_in[idx++] = stride_in[i];
    }
  }
}

void ValueStrides(
    at::detail::Array<int, MAX_DIM>& stride_v,
    at::detail::Array<int, MAX_DIM>& shape_v,
    at::detail::Array<int, MAX_DIM>& task_shape,
    const Tensor& value,
    int shape_ndim) {
  // update strides of indices, similar proceedure as IndicesStrides
  int v_ndim = value.dim();
  auto v_dim = value.sizes().vec();
  for (int i = v_ndim - 1; i >= 0; --i) {
    if (i == v_ndim - 1) {
      stride_v[i] = 1;
    } else {
      stride_v[i] = stride_v[i + 1] * v_dim[i + 1];
    }
  }
  for (int j = shape_ndim - 1; j >= 0; --j) {
    int offset = shape_ndim - 1 - j;
    int dim_a = shape_ndim - 1 - offset;
    int dim_b = v_ndim - 1 - offset;
    bool bcast_dim = (dim_b < 0 || task_shape[dim_a] != v_dim[dim_b]);
    if (bcast_dim) {
      shape_v[j] = 1;
      stride_v[j] = 0;
    } else {
      shape_v[j] = v_dim[dim_b];
      stride_v[j] = stride_v[dim_b];
    }
  }

  for (int j = 0; j < shape_ndim; ++j) {
    if (task_shape[j] != 1 && shape_v[j] == 1) {
      stride_v[j] = 0;
    }
  }
}

void OutputStrides(
    at::detail::Array<int, MAX_DIM>& stride_o,
    const Tensor& out) {
  for (int i = out.dim() - 1; i >= 0; --i) {
    if (i == out.dim() - 1) {
      stride_o[i] = 1;
    } else {
      stride_o[i] = stride_o[i + 1] * out.sizes()[i + 1];
    }
  }
}

} // namespace native
} // namespace at