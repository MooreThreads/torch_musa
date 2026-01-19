#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/DispatchStub.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/Functions.h>
#else
#include <ATen/native/BatchLinearAlgebra.h>
#include <ATen/native/LinearAlgebraUtils.h>
#include <ATen/ops/_linalg_check_errors.h>
#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/_linalg_solve_ex_native.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_cholesky_ex_native.h>
#include <ATen/ops/linalg_inv_ex_native.h>
#include <ATen/ops/linalg_inv_ex_ops.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <ATen/ops/linalg_lu_factor_ex.h>
#include <ATen/ops/linalg_lu_solve.h>
#include <ATen/ops/linalg_lu_solve_native.h>
#include <ATen/ops/linalg_solve_ex.h>
#endif

#include "torch_musa/csrc/aten/linalg/BatchLinearAlgebraLib.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

Tensor& CholeskyInverseKernelImpl(Tensor& result, Tensor& infos, bool upper) {
  // This function calculates the inverse matrix in-place
  // result should be in column major order and contain matrices to invert
  // the content of result is overwritten by 'apply_cholesky_inverse'
  return at::musa::cholesky_inverse_kernel_impl_musolver(result, infos, upper);
}

static void LUFactor(
    const Tensor& input,
    const Tensor& pivots,
    const Tensor& infos,
    bool compute_pivots) {
  auto batch_size = at::native::batchCount(input);
  (void)batch_size; // Silence unused warning in some builds
  auto m = input.size(-2);
  auto n = input.size(-1);
  TORCH_CHECK(
      compute_pivots == true,
      "musolver doesnot support pivots == false, musolver 1.4.0, while cusolver support");
  const auto lu_factor_solver = [batch_size, m, n](
                                    const Tensor& input,
                                    const Tensor& pivots,
                                    const Tensor& infos,
                                    bool compute_pivots) {
    // CUDA provides cusolver and cublas for LU factorization, but MUSA only
    // provides musolver.
    at::musa::lu_factor_looped_musolver(input, pivots, infos, compute_pivots);
  };

  lu_factor_solver(input, pivots, infos, compute_pivots);

  // We return the trivial permutation of pivots starting with 1 (FORTRAN
  // indexing)
  if (!compute_pivots) {
    auto k = std::min(input.size(-2), input.size(-1));
    auto pivots_tmp = at::arange(1, k + 1, input.options().dtype(at::kInt));
    pivots.copy_(pivots_tmp);
  }
}

REGISTER_MUSA_DISPATCH(lu_factor_stub, &LUFactor);
REGISTER_MUSA_DISPATCH(cholesky_inverse_stub, &CholeskyInverseKernelImpl);

} // namespace native
namespace musa {

/* TODO(@mt-ai): linalg operators here are all walkarounded by moving tensors to
 * CPU. To implement these ops on MUSA, some APIs from muBlas are requested,
 * while many of them are missing. We already file an issue (on JIRA) to blas
 * team of this feature request, check:
 * https://jira.mthreads.com/browse/SW-30954
 */

// TODO(@ai-infra): Implement with musolver
::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> LinalgLstsqOut(
    const at::Tensor& self,
    const at::Tensor& b,
    c10::optional<double> rcond,
    c10::optional<c10::string_view> driver,
    at::Tensor& solution,
    at::Tensor& residuals,
    at::Tensor& rank,
    at::Tensor& singular_values) {
  c10::optional<Device> common_device = nullopt;
  c10::impl::check_and_update_common_device(
      common_device, solution, "LinalgLstsqOut", "solution");
  c10::impl::check_and_update_common_device(
      common_device, residuals, "LinalgLstsqOut", "residuals");
  c10::impl::check_and_update_common_device(
      common_device, rank, "LinalgLstsqOut", "rank");
  c10::impl::check_and_update_common_device(
      common_device, singular_values, "LinalgLstsqOut", "singular_values");
  c10::impl::check_and_update_common_device(
      common_device, self, "LinalgLstsqOut", "self");
  c10::impl::check_and_update_common_device(
      common_device, b, "LinalgLstsqOut", "b");
  const OptionalDeviceGuard device_guard(device_of(self));

  auto cpu_self =
      at::empty(self.sizes(), self.options().device(DeviceType::CPU));
  auto cpu_b = at::empty(b.sizes(), b.options().device(DeviceType::CPU));
  auto cpu_solution =
      at::empty(solution.sizes(), solution.options().device(DeviceType::CPU));
  auto cpu_residuals =
      at::empty(residuals.sizes(), residuals.options().device(DeviceType::CPU));
  auto cpu_rank =
      at::empty(rank.sizes(), rank.options().device(DeviceType::CPU));
  auto cpu_singular_values = at::empty(
      singular_values.sizes(),
      singular_values.options().device(DeviceType::CPU));

  cpu_self.copy_(self);
  cpu_b.copy_(b);
  cpu_solution.copy_(solution);
  cpu_residuals.copy_(residuals);
  cpu_rank.copy_(rank);
  cpu_singular_values.copy_(singular_values);

  at::native::linalg_lstsq_out(
      cpu_self,
      cpu_b,
      rcond,
      driver,
      cpu_solution,
      cpu_residuals,
      cpu_rank,
      cpu_singular_values);

  solution = at::empty(
      cpu_solution.sizes(),
      cpu_solution.options().device(DeviceType::PrivateUse1));
  solution.copy_(cpu_solution);
  residuals = at::empty(
      cpu_residuals.sizes(),
      cpu_residuals.options().device(DeviceType::PrivateUse1));
  residuals.copy_(cpu_residuals);
  rank = at::empty(
      cpu_rank.sizes(), cpu_rank.options().device(DeviceType::PrivateUse1));
  rank.copy_(cpu_rank);
  singular_values = at::empty(
      cpu_singular_values.sizes(),
      cpu_singular_values.options().device(DeviceType::PrivateUse1));
  singular_values.copy_(cpu_singular_values);

  return ::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>{
      solution, residuals, rank, singular_values};
  ;
}

Tensor _CholeskySolveHelperMusa(
    const Tensor& self,
    const Tensor& A,
    bool upper) {
  return _cholesky_solve_helper_musolver(self, A, upper);
}

} // namespace musa
} // namespace at
