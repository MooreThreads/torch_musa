#pragma once

#include <ATen/Context.h>
#include <ATen/core/Tensor.h>

#include <ATen/native/TransposeType.h>
#include <ATen/native/musa/MiscUtils.h>

namespace at::musa {
void lu_factor_looped_musolver(
    const Tensor& self,
    const Tensor& pivots,
    const Tensor& infos,
    bool get_pivots);

void cholesky_helper_musolver(
    const Tensor& input,
    bool upper,
    const Tensor& info);
Tensor& cholesky_inverse_kernel_impl_musolver(
    Tensor& result,
    Tensor& infos,
    bool upper);

void lu_solve_looped_musolver(
    const Tensor& LU,
    const Tensor& pivots,
    const Tensor& B,
    at::native::TransposeType transpose);
} // namespace at::musa
