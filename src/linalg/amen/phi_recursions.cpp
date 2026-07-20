#include "ttnte/linalg/amen/phi_recursions.hpp"

namespace ttnte::linalg::amen {

torch::Tensor compute_phi_bck_A(const torch::Tensor& phi_now,
  const torch::Tensor& core_left, const torch::Tensor& core_a,
  const torch::Tensor& core_right)
{
  torch::Tensor phi = torch::tensordot(core_left, phi_now, {2}, {0});
  phi = torch::tensordot(phi, core_a, {1, 2}, {1, 3});
  return torch::tensordot(phi, core_right, {1, 3}, {2, 1});
}

torch::Tensor compute_phi_fwd_A(const torch::Tensor& phi_now,
  const torch::Tensor& core_left, const torch::Tensor& core_a,
  const torch::Tensor& core_right)
{
  torch::Tensor phi_next = torch::tensordot(core_left, phi_now, {0}, {0});
  phi_next = torch::tensordot(phi_next, core_a, {0, 2}, {1, 0});
  return torch::tensordot(phi_next, core_right, {1, 2}, {0, 1});
}

torch::Tensor compute_phi_bck_rhs(
  const torch::Tensor& phi_now, const torch::Tensor& core_b,
  const torch::Tensor& core)
{
  torch::Tensor phi = torch::tensordot(core_b, phi_now, {2}, {0});
  return torch::tensordot(phi, core, {1, 2}, {1, 2});
}

torch::Tensor compute_phi_fwd_rhs(
  const torch::Tensor& phi_now, const torch::Tensor& core_b,
  const torch::Tensor& core)
{
  torch::Tensor phi_next = torch::tensordot(phi_now, core_b, {0}, {0});
  return torch::tensordot(phi_next, core, {0, 1}, {0, 1});
}

} // namespace ttnte::linalg::amen
