#pragma once

// Folded local-operator apply.
//
// Algorithmic background: Roehrig-Zoellner, Becklas, Thies, Basermann (2025),
// "Performance of linear solvers in tensor-train format on current multicore
// architectures", Sec. 5.3. Independent reimplementation against ttnte's own
// tensor types (no code copied).
//
// The AMEn local subproblem's operator apply is, in torchTT's baseline
// (external/torchTT/torchtt/cpp/amen_solve.h, `local_product`), the einsum
//   lsr,smnS,LSR,rnR->lmL
// i.e. Phi_left[l,s,r] (x) A_core[s,m,n,S] (x) Phi_right[L,S,R] applied to a
// candidate core Y[r,n,R]. Implemented as three chained `at::tensordot`
// calls, each of which internally permutes/reshapes its operands -- with no
// reuse of that permuted layout across repeated calls (e.g. every GMRES
// Krylov iteration for the same core).
//
// `FoldedLocalOperator::build()` reorders Phi_left/A_core/Phi_right's
// dimensions ONCE per core so that contracted index pairs and free index
// pairs each flatten into a single dimension; `apply()` then reduces to
// exactly three direct `torch::matmul` calls with no further permutes,
// amortizing the one-time layout cost across every iteration that reuses
// the same core's operator (GMRES Krylov vectors, residual checks, etc).
//
// Deliberately NOT pre-contracting A_core and Phi_right (or Phi_left and
// A_core) together at build time into a single combined operand: although
// that would let apply() skip its two runtime permutes entirely, it
// collapses the low-rank factorization the shared operator-rank index (S)
// provides. A_core/Phi_right stored separately cost O(s*n*m*S + R*S*L); a
// combined `G[s,m,n,L,R] = sum_S A_core[s,m,n,S] * Phi_right[L,S,R]` costs
// O(s*n*m*R*L) -- multiplicative in the STATE's TT ranks R,L (which reach
// into the hundreds under AMEn's adaptive enrichment; e.g. `max_rank=500` in
// scripts/ans26/c5g7_pincell_2d.py) instead of additive. Tried and reverted
// -- see memory.

#include <torch/extension.h>

namespace ttnte::linalg::amen {

/// A lightweight value type, deliberately not following the codebase's usual
/// factory + `Ptr = shared_ptr<...>` pattern (see `ttnte/CLAUDE.md`): one is
/// built and discarded per AMEn core, every sweep, and reused across every
/// GMRES Krylov iteration for that core -- heap allocation + refcounting per
/// instance would be real overhead in that hot loop.
class FoldedLocalOperator {
public:
  /// @brief Precompute the folded GEMM operands for one AMEn core's local
  /// operator.
  /// @param phi_left Left interface tensor, shape `[l, s, r]` (`l`/`r` are
  /// the bra/ket x-ranks, `s` is A's left rank).
  /// @param a_core The operator's core at this position, shape
  /// `[s, m, n, S]`.
  /// @param phi_right Right interface tensor, shape `[L, S, R]`.
  /// @param regularization Proximal (Tikhonov-style) diagonal shift added by
  /// `apply()`/`to_dense()`: `regularization * y` / `regularization * I`.
  /// Zero (default) reproduces the original unregularized operator exactly.
  /// Only meaningful when the operator is square (`l == r`, `m == n`,
  /// `L == R`) -- see `AMEnNativeOptions::proximal_regularization`.
  static FoldedLocalOperator build(const torch::Tensor& phi_left,
    const torch::Tensor& a_core, const torch::Tensor& phi_right,
    double regularization = 0.0);

  /// @brief Apply the local operator to a candidate core.
  /// @param y Shape `[r, n, R]` (flattened or not; reshaped internally).
  /// @return Shape `[l, m, L]`. Includes `+ regularization * y` when this
  /// operator was built with a nonzero `regularization`.
  torch::Tensor apply(const torch::Tensor& y) const;

  /// @brief Materialize the local operator as a dense `[l*m*L, r*n*R]`
  /// matrix, for the small-local-problem direct-solve path. Computed via two
  /// folded GEMMs rather than the baseline's tensordot chain. Includes
  /// `+ regularization * I` when this operator was built with a nonzero
  /// `regularization`.
  torch::Tensor to_dense() const;

  /// @brief A float32 copy of this operator, rebuilt from the same raw
  /// interface/core tensors cast down. For `AMEnNativeOptions::
  /// gmres_mixed_precision`'s inner-Krylov-loop cast boundary -- the
  /// original (float64) operator is kept alongside this one, not replaced,
  /// so residual/truncation logic elsewhere continues to see full precision.
  FoldedLocalOperator to_float32() const;

  int64_t l() const noexcept { return l_; }
  int64_t m() const noexcept { return m_; }
  int64_t n() const noexcept { return n_; }
  int64_t r() const noexcept { return r_; }
  int64_t L() const noexcept { return L_; }
  int64_t R() const noexcept { return R_; }

private:
  torch::Tensor phi_left_mat_;  // [l*s, r]
  torch::Tensor a_core_mat_;    // [s*n, m*S]
  torch::Tensor phi_right_mat_; // [R*S, L]
  torch::Tensor phi_left_raw_;  // [l, s, r], kept for to_dense()
  torch::Tensor a_core_raw_;    // [s, m, n, S], kept for to_dense()
  torch::Tensor phi_right_raw_; // [L, S, R], kept for to_dense()
  int64_t l_ = 0, s_ = 0, r_ = 0, m_ = 0, n_ = 0, S_ = 0, R_ = 0, L_ = 0;
  double regularization_ = 0.0;
};

} // namespace ttnte::linalg::amen
