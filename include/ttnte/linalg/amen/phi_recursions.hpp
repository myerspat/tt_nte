#pragma once

// Interface-tensor ("Phi") recursions used by the AMEn sweep to maintain the
// reduced (Galerkin-projected) representation of `A` and `b` against the
// current solution/residual TT bases. Same contractions as torchTT's
// `compute_phi_{bck,fwd}_{A,rhs}` (external/torchTT/torchtt/cpp/amen_solve.h)
// -- an independent reimplementation against ttnte's own tensor types (no
// code copied) so the native sweep driver isn't required to include the
// vendored torchTT headers.

#include <torch/extension.h>

namespace ttnte::linalg::amen {

/// @brief Backward Phi recursion for `dot(left, A @ right)`.
/// @param phi_now Shape `[r1_k+1, R_k+1, r2_k+1]`.
/// @param core_left Shape `[r1_k, N_k, r1_k+1]`.
/// @param core_a Shape `[R_k, N_k, N_k, R_k+1]`.
/// @param core_right Shape `[r2_k, N_k, r2_k+1]`.
/// @return Shape `[r1_k, R_k, r2_k]`.
torch::Tensor compute_phi_bck_A(const torch::Tensor& phi_now,
  const torch::Tensor& core_left, const torch::Tensor& core_a,
  const torch::Tensor& core_right);

/// @brief Forward Phi recursion for `dot(left, A @ right)`.
/// @param phi_now Shape `[r1_k, R_k, r2_k]`.
/// @return Shape `[r1_k+1, R_k+1, r2_k+1]`.
torch::Tensor compute_phi_fwd_A(const torch::Tensor& phi_now,
  const torch::Tensor& core_left, const torch::Tensor& core_a,
  const torch::Tensor& core_right);

/// @brief Backward Phi recursion for `dot(left, b)`.
/// @param phi_now Shape `[rb_k+1, r_k+1]`.
/// @param core_b Shape `[rb_k, N_k, rb_k+1]`.
/// @param core Shape `[r_k, N_k, r_k+1]`.
/// @return Shape `[rb_k, r_k]`.
torch::Tensor compute_phi_bck_rhs(
  const torch::Tensor& phi_now, const torch::Tensor& core_b,
  const torch::Tensor& core);

/// @brief Forward Phi recursion for `dot(left, b)`.
/// @param phi_now Shape `[rb_k, r_k]`.
/// @return Shape `[rb_k+1, r_k+1]`.
torch::Tensor compute_phi_fwd_rhs(
  const torch::Tensor& phi_now, const torch::Tensor& core_b,
  const torch::Tensor& core);

} // namespace ttnte::linalg::amen
