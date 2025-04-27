#include <ATen/Config.h>
#include <ATen/core/op_registration/adaption.h>
#include <ATen/native/Resize.h>

#include <ATen/ops/_linalg_solve_ex.h>
#include <ATen/ops/cholesky_inverse.h>
#include <ATen/ops/linalg_cholesky_ex.h>
#include <ATen/ops/linalg_inv_ex_ops.h>
#include <ATen/ops/linalg_lstsq_native.h>
#include <torch/library.h>
#include "torch_musa/csrc/aten/ops/LinearAlgebra.h"
#include "torch_musa/csrc/aten/ops/TensorFactory.h"
#include "torch_musa/csrc/aten/utils/Utils.h"

namespace at {
namespace native {

DEFINE_DISPATCH(inverse_stub);
REGISTER_NO_CPU_DISPATCH(inverse_stub);

} // namespace native
namespace musa {

/* TODO(@mt-ai): linalg operators here are all walkarounded by moving tensors to
 * CPU. To implement these ops on MUSA, some APIs from muBlas are requested,
 * while many of them are missing. We already file an issue (on JIRA) to blas
 * team of this feature request, check:
 * https://jira.mthreads.com/browse/SW-30954
 */

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

::std::tuple<at::Tensor&, at::Tensor&> LinalgInvExOutInverse(
    const at::Tensor& A,
    bool check_errors,
    at::Tensor& inverse,
    at::Tensor& info) {
  // TODO(@mt-ai): this kernel is doing the same arithmetic as the
  // one below (LinalgInverse), but pytorch got a hacky way to impl
  // such functions. For example, torch.inverse should be an alias
  // of torch.linalg.inv, while we impl them seperately.
  auto cpu_inverse =
      at::empty(inverse.sizes(), inverse.options().device(DeviceType::CPU));
  auto cpu_info =
      at::empty(info.sizes(), info.options().device(DeviceType::CPU));
  auto cpu_A = at::empty(A.sizes(), A.options().device(DeviceType::CPU));

  cpu_inverse.copy_(inverse);
  cpu_info.copy_(info);
  cpu_A.copy_(A);

  at::_ops::linalg_inv_ex_inverse::call(
      cpu_A, check_errors, cpu_inverse, cpu_info);
  inverse = at::empty(
      cpu_inverse.sizes(),
      cpu_inverse.options().device(DeviceType::PrivateUse1));
  inverse.copy_(cpu_inverse);
  info = at::empty(
      cpu_info.sizes(), cpu_info.options().device(DeviceType::PrivateUse1));
  info.copy_(cpu_info);
  return ::std::tuple<at::Tensor&, at::Tensor&>{inverse, info};
}

at::Tensor LinalgInverse(const at::Tensor& A) {
  // TODO(@mt-ai): kernel provided by mudnn now only support
  // dtypes of fp32 and fp64
  // TODO(@mt-ai): kernel internal optimization of non-continuous issues
  TORCH_CHECK(
      A.scalar_type() == at::ScalarType::Float ||
          A.scalar_type() == at::ScalarType::Double,
      "Inverse currently only supports float32/64 dtype, got ",
      A.scalar_type());
  c10::musa::MUSAGuard device_guard(A.device());

  at::Tensor A_contiguous = A.contiguous();
  at::Tensor result = at::empty_like(A_contiguous);
  at::native::inverse_stub(kMUSA, result, A_contiguous);
  return result;
}

::std::tuple<at::Tensor, at::Tensor> LinalgCholeskyEx(
    const at::Tensor& self,
    bool upper,
    bool check_errors) {
  at::Tensor self_cpu = self.to(kCPU);
  std::tuple<at::Tensor, at::Tensor> rst =
      at::linalg_cholesky_ex(self_cpu, upper, check_errors);
  at::Tensor& rst0 = std::get<0>(rst);
  at::Tensor& rst1 = std::get<1>(rst);
  rst0 = rst0.to(kMUSA);
  rst1 = rst1.to(kMUSA);
  return rst;
}

::std::tuple<at::Tensor&, at::Tensor&> LinalgCholeskyExOut(
    const at::Tensor& self,
    bool upper,
    bool check_errors,
    at::Tensor& L,
    at::Tensor& info) {
  at::Tensor L_cpu = L.to(kCPU);
  at::Tensor info_cpu = info.to(kCPU);
  at::Tensor self_cpu = self.to(kCPU);
  std::tuple<at::Tensor&, at::Tensor&> rst = at::linalg_cholesky_ex_out(
      L_cpu, info_cpu, self_cpu, upper, check_errors);
  at::Tensor& rst0 = std::get<0>(rst);
  at::Tensor& rst1 = std::get<1>(rst);
  rst0 = rst0.to(kMUSA);
  rst1 = rst1.to(kMUSA);
  return rst;
}

at::Tensor& CholeskyInverseOut(
    const at::Tensor& input,
    bool upper,
    at::Tensor& result) {
  at::Tensor input_cpu = input.to(kCPU);
  at::Tensor result_cpu = result.to(kCPU);
  result_cpu = at::cholesky_inverse_out(result_cpu, input_cpu, upper);
  result = result_cpu.to(kMUSA);
  return result;
}

at::Tensor CholeskyInverse(const at::Tensor& input, bool upper) {
  at::Tensor result = at::empty({0}, input.options());
  result = CholeskyInverseOut(input, upper, result);
  return result;
}

::std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&>
LinalgSolveExResult(
    const at::Tensor& A,
    const at::Tensor& B,
    bool left,
    bool check_errors,
    at::Tensor& result,
    at::Tensor& LU,
    at::Tensor& pivots,
    at::Tensor& info) {
  at::Tensor A_cpu = A.to(kCPU);
  at::Tensor B_cpu = B.to(kCPU);
  at::Tensor result_cpu = result.to(kCPU);
  at::Tensor LU_cpu = LU.to(kCPU);
  at::Tensor pivots_cpu = pivots.to(kCPU);
  at::Tensor info_cpu = info.to(kCPU);
  std::tuple<at::Tensor&, at::Tensor&, at::Tensor&, at::Tensor&> rst =
      at::_linalg_solve_ex_out(
          result_cpu,
          LU_cpu,
          pivots_cpu,
          info_cpu,
          A_cpu,
          B_cpu,
          left,
          check_errors);
  at::Tensor& rst0 = std::get<0>(rst);
  at::Tensor& rst1 = std::get<1>(rst);
  at::Tensor& rst2 = std::get<2>(rst);
  at::Tensor& rst3 = std::get<3>(rst);
  rst0 = rst0.to(kMUSA);
  rst1 = rst1.to(kMUSA);
  rst2 = rst2.to(kMUSA);
  rst3 = rst3.to(kMUSA);
  return rst;
}

::std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> LinalgSolveEx(
    const at::Tensor& A,
    const at::Tensor& B,
    bool left,
    bool check_errors) {
  at::Tensor A_cpu = A.to(kCPU);
  at::Tensor B_cpu = B.to(kCPU);
  std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> rst =
      at::_linalg_solve_ex(A_cpu, B_cpu, left, check_errors);
  at::Tensor& rst0 = std::get<0>(rst);
  at::Tensor& rst1 = std::get<1>(rst);
  at::Tensor& rst2 = std::get<2>(rst);
  at::Tensor& rst3 = std::get<3>(rst);
  rst0 = rst0.to(kMUSA);
  rst1 = rst1.to(kMUSA);
  rst2 = rst2.to(kMUSA);
  rst3 = rst3.to(kMUSA);
  return rst;
}

} // namespace musa
} // namespace at
