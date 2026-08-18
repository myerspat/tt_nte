#pragma once

// Global TT-rank-1 preconditioner.
//
// Algorithmic background: Roehrig-Zoellner, Becklas, Thies, Basermann (2025),
// "Performance of linear solvers in tensor-train format on current multicore
// architectures", Sec. 3.4.1. Independent reimplementation against ttnte's
// own tensor types (no code copied).
//
// Given an operator A_TT, truncate it to TT-rank 1 (`A_tilde`), SVD each of
// A_tilde's (now dense, per-core) m_i x n_i factors as `U_i S_i V_i^T`, and
// build:
//   P_left  = kron_i (S_i^{-1/2} U_i^T)
//   P_right = kron_i (V_i S_i^{-1/2})
// so that the preconditioned system is
//   (P_left A_TT P_right) y = P_left b,   x = P_right y.
// Because P_left/P_right are rank-1 Kronecker operators, applying them is a
// per-core matmul independent of TT rank, and -- the key property motivating
// this specific preconditioner -- they never increase the TT rank of the
// vectors they're applied to.

#include "ttnte/linalg/tt_engine.hpp"
#include <vector>

namespace ttnte::linalg::amen {

/// A lightweight value type, deliberately not following the codebase's usual
/// factory + `Ptr = shared_ptr<...>` pattern (see `ttnte/CLAUDE.md`): it's an
/// internal solve-time detail (built once per `amen_solve_dispatch` call, not
/// part of the public solver API), so the extra heap indirection buys
/// nothing here.
class Rank1Preconditioner {
public:
  /// @brief Build the preconditioner from the TT-rank-1 truncation of `A`.
  /// @param A The (generally full-rank) operator to precondition.
  /// @param sv_floor_ratio Singular values of each core's dense factor below
  /// `sv_floor_ratio * max_singular_value` are clamped up to that floor
  /// before inversion, to keep `S_i^{-1/2}` bounded on near-singular cores.
  static Rank1Preconditioner build(
    const TTEngine& A, double sv_floor_ratio = 1e-10);

  /// @brief Apply `P_left` to a TT-vector's cores (transforms each core's
  /// m-mode from the operator's output-space size to the preconditioner's
  /// canonical rank). Cores use the plain 3-D vector-core convention
  /// `[r_l, m_i, r_r]` (no trivial trailing mode), matching
  /// `FoldedLocalOperator`/the native sweep drivers -- NOT `TTEngine`'s 4-D
  /// `[r_l, m_i, 1, r_r]` convention.
  std::vector<torch::Tensor> apply_left(
    const std::vector<torch::Tensor>& vector_cores) const;

  /// @brief Apply `P_right` to a TT-vector's cores (inverse direction of
  /// `apply_left`, mapping the canonical rank back to the operator's
  /// input-space size). Same 3-D core convention as `apply_left`.
  std::vector<torch::Tensor> apply_right(
    const std::vector<torch::Tensor>& vector_cores) const;

  /// @brief Apply `P_right^{-1}` to a TT-vector's cores: maps a vector
  /// already living in the operator's input space (e.g. a user-supplied
  /// warm-start `x0`) into the preconditioned/canonical space, so it can be
  /// used to warm-start a solve of the preconditioned system. Same 3-D core
  /// convention as `apply_left`.
  std::vector<torch::Tensor> apply_right_inverse(
    const std::vector<torch::Tensor>& vector_cores) const;

  /// @brief Form `P_left A P_right`, applied to the (full-rank) operator's
  /// cores. Only transforms each core's m/n mode sizes -- bond ranks are
  /// untouched, so this does not change `A`'s TT rank.
  std::vector<torch::Tensor> sandwich_operator(
    const std::vector<torch::Tensor>& operator_cores) const;

private:
  /// L_i = S_i^{-1/2} U_i^T, shape [rank_i, m_i].
  std::vector<torch::Tensor> left_factors_;
  /// R_i = V_i S_i^{-1/2}, shape [n_i, rank_i].
  std::vector<torch::Tensor> right_factors_;
  /// R_i^{-1} = S_i^{1/2} V_i^T, shape [rank_i, n_i].
  std::vector<torch::Tensor> right_inverse_factors_;
};

} // namespace ttnte::linalg::amen
