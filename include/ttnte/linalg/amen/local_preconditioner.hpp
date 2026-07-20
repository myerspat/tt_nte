#pragma once

// Local (per-core) Jacobi-style preconditioner for the AMEn inner GMRES
// solve.
//
// Algorithmic background: torchTT's baseline (external/torchTT/torchtt/cpp/
// matvecs.h, `AMENsolveMV::setter`/`apply_prec`) supports two local
// preconditioners, built from the *diagonal* of the interface tensors
// (Phi_left/Phi_right) -- a cheap, per-core Jacobi-style approximation of
// the local operator, not a global one (compare `Rank1Preconditioner`,
// which preconditions the whole system once before sweeping):
//
//   Jl = tensordot(diagonal(Phi_left, dim0=0, dim1=2), A_core, {0},{0})
//
//   LOCAL_C_PREC: Jr = diagonal(Phi_right, dim0=0, dim1=2)
//                 Jt = tensordot(Jl, Jr, {3},{0}).permute({0,3,1,2})
//                 -- small n x n block per rank index; cheap.
//   LOCAL_R_PREC: Jt = tensordot(Jl, Phi_right, {3},{1})
//                      .permute({0,1,3,2,4})
//                 -- keeps the right interface's full (non-diagonal)
//                 structure; bigger, more accurate block.
//
// The baseline explicitly inverts this block (`J = inv(Jt)`) and uses it as
// right-preconditioning of a *correction* equation (`A dx = drhs`, zero
// initial guess in preconditioned space, `dx = apply_prec(y)` recovered
// after GMRES). This reimplementation differs in two deliberate ways: (1)
// `Jt` is kept un-inverted and `apply_inverse` uses `torch::linalg_solve`
// instead of materializing an explicit inverse, avoiding the numerical
// error amplification of small-matrix inversion; (2) since `apply_forward`
// is available (the baseline never needed it), the native GMRES solve
// preconditions the system directly (`(A M^{-1}) y = rhs`, `y0 =
// apply_forward(x0)`, `x = apply_inverse(y)`) instead of going through a
// correction-equation reformulation -- mathematically equivalent, simpler
// to wire into `gmres_solve`'s existing warm-start API. Independent
// reimplementation against ttnte's own tensor types (no code copied).

#include "ttnte/linalg/amen/amen_config.hpp"
#include <torch/extension.h>

namespace ttnte::linalg::amen {

/// A lightweight value type, deliberately not following the codebase's usual
/// factory + `Ptr = shared_ptr<...>` pattern (see `ttnte/CLAUDE.md`): one is
/// built and discarded per AMEn core, every sweep -- heap allocation +
/// refcounting per instance would be real overhead in that hot loop.
class LocalPreconditioner {
public:
  /// @brief Build the local preconditioner from one core's interface
  /// tensors, matching the same inputs `FoldedLocalOperator::build` takes.
  /// @param phi_left Shape `[l, s, r]` (`l == r` required -- the
  /// preconditioner is only defined for a square local operator, same
  /// requirement as `gmres_solve`).
  /// @param a_core Shape `[s, m, n, S]` (`m == n` required).
  /// @param phi_right Shape `[L, S, R]` (`L == R` required).
  /// @param mode `AMEnPreconditioner::LOCAL_C_PREC` (cheap; right interface
  /// also diagonalized) or `AMEnPreconditioner::LOCAL_R_PREC` (right
  /// interface kept full; more accurate/expensive). `NONE`/`RANK1` are
  /// invalid here (`RANK1` is a global preconditioner, handled elsewhere).
  static LocalPreconditioner build(const torch::Tensor& phi_left,
    const torch::Tensor& a_core, const torch::Tensor& phi_right,
    AMEnPreconditioner mode);

  /// @brief Apply the forward map `M x` (the un-inverted small block).
  /// @param x Shape `[r, n, R]`.
  /// @return Shape `[r, n, R]`.
  torch::Tensor apply_forward(const torch::Tensor& x) const;

  /// @brief Apply the inverse map `M^{-1} y` via `torch::linalg_solve`
  /// (never materializes an explicit inverse).
  /// @param y Shape `[r, n, R]`.
  /// @return Shape `[r, n, R]`.
  torch::Tensor apply_inverse(const torch::Tensor& y) const;

private:
  AMEnPreconditioner mode_ = AMEnPreconditioner::NONE;
  torch::Tensor
    Jt_; // LOCAL_C_PREC: [d, d, n, n]; LOCAL_R_PREC: [d, m, L, n, R]
  int64_t r_ = 0, n_ = 0, R_ = 0, d_ = 0, m_ = 0, L_ = 0;
};

} // namespace ttnte::linalg::amen
