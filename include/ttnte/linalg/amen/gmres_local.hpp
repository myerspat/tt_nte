#pragma once

// Native local GMRES for the AMEn inner (per-core) linear solve.
//
// Structural reference: include/ttnte/linalg/old/gmres.hpp (a JAX
// `scipy.sparse.linalg`-derived port already vendored in this codebase,
// citation preserved there), which already contains two solve strategies:
// `gmres_incremental` (per-iteration Givens rotations + early-stop residual
// check -- cheap on CPU, no device round-trip) and `gmres_batched` (build the
// full fixed-budget Krylov basis with NO incremental least-squares tracking,
// then solve the small Hessenberg least-squares problem once at the end --
// avoids the per-iteration host sync that an incremental method forces on
// GPU). This is an independent reimplementation against `FoldedLocalOperator`
// (matrix-free, no dense global operator) instead of the old dense-global
// `Operator::apply()`, with two fixes: (1) all scalar/tensor bookkeeping runs
// in the input's own dtype instead of being hardcoded to `double`; (2) the
// GPU strategy's final small least-squares solve uses `at::linalg_lstsq`
// instead of the old code's normal-equations (`H @ H.T`) solve, since normal
// equations square the condition number of `H` -- exactly what to avoid when
// the local system is ill-conditioned.

#include "ttnte/linalg/amen/local_operator.hpp"
#include "ttnte/linalg/amen/local_preconditioner.hpp"

namespace ttnte::linalg::amen {

/// @brief Restarted GMRES against a (square) `FoldedLocalOperator`.
/// Dispatches to `gmres_solve_cpu()` or `gmres_solve_gpu()` based on the
/// device of `rhs` -- unless `prefer_incremental` is set, in which case
/// `gmres_solve_cpu()`'s incremental-Givens strategy is used regardless of
/// device.
/// @param op The local operator; must have matching domain/codomain shape
/// (`op.l() == op.r()`, `op.m() == op.n()`, `op.L() == op.R()`).
/// @param rhs Right-hand side, shape `[l, m, L]`.
/// @param x0 Initial guess, same shape as `rhs`. When `prec` is given, this
/// must already live in the *preconditioned* space (i.e. the caller passes
/// `prec->apply_forward(true_x0)`, and the returned solution is likewise in
/// preconditioned space -- the caller applies `prec->apply_inverse(...)` to
/// recover the true solution). `gmres_solve` itself never transforms
/// `x0`/the return value; it only uses `prec` to precondition the operator
/// applies inside the Krylov iteration.
/// @param max_iterations Krylov subspace dimension before a restart.
/// @param restarts Maximum number of restarts.
/// @param rel_tol Relative residual tolerance (`||b - Ax|| <= rel_tol *
/// ||b||`).
/// @param prefer_incremental When true, always use `gmres_solve_cpu()`'s
/// strategy, even for CUDA tensors. `gmres_solve_cpu()`'s implementation is
/// plain device-agnostic ATen ops, so this is correct on GPU too -- it just
/// pays the per-iteration host-sync cost that `gmres_solve_gpu()` exists to
/// avoid. Safer default: the fixed-budget "batched" GPU strategy never
/// early-stops within a Krylov expansion (always builds the full
/// `max_iterations`-sized basis), which risks orthogonality loss for larger
/// `max_iterations` or harder local systems -- see
/// `AMEnNativeOptions::use_gpu_batched_gmres`.
/// @param prec Optional local (right-)preconditioner: each operator apply
/// inside the Krylov iteration becomes `op.apply(prec->apply_inverse(v))`
/// instead of `op.apply(v)`, i.e. solves `(A M^-1) y = rhs`. `nullptr`
/// (default) means unpreconditioned, byte-identical to the original
/// behavior.
/// @param check_interval Forwarded to `gmres_solve_gpu()` -- see there.
/// Ignored by `gmres_solve_cpu()`, which already checks every iteration.
/// @param gmres_mixed_precision Forwarded to whichever strategy is
/// dispatched to -- see `AMEnNativeOptions::gmres_mixed_precision`.
/// @return The approximate solution, same shape as `rhs` (in preconditioned
/// space if `prec` was given -- see the `x0` note above).
torch::Tensor gmres_solve(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, bool prefer_incremental = false,
  const LocalPreconditioner* prec = nullptr, int check_interval = 8,
  bool gmres_mixed_precision = false);

/// @brief CPU strategy: incremental Givens rotations with a per-iteration
/// early-stop residual check.
/// @param gmres_mixed_precision When true, runs the inner Arnoldi/Gram-
/// Schmidt/Givens build in float32 (on either CPU or CUDA `rhs`) while
/// keeping `x`/`rhs`/the true per-restart residual in float64 -- see
/// `AMEnNativeOptions::gmres_mixed_precision`.
torch::Tensor gmres_solve_cpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec = nullptr,
  bool gmres_mixed_precision = false);

/// @brief GPU strategy: fixed-budget Arnoldi expansion (no incremental
/// least-squares tracking), with a residual check amortized every
/// `check_interval` iterations instead of every iteration (down from
/// `gmres_solve_cpu()`'s per-iteration host sync, while still bounding
/// wasted work -- past convergence -- to at most `check_interval - 1` Krylov
/// steps instead of always burning the full `max_iterations` budget). Each
/// check solves the small least-squares problem on the Krylov subspace built
/// so far (`H[:k+2, :k+1]`, `beta[:k+2]`) and reads back its residual norm --
/// one host sync per check, not per iteration. On break (early or at the
/// budget), the solution uses the last solved `y`/subspace, so no final
/// re-solve is needed.
/// @param check_interval Check convergence (and update `x` if it holds)
/// every this many Krylov iterations, plus always on the final iteration of
/// a restart. Smaller values check more often (more syncs, less wasted
/// work); larger values check less often (fewer syncs, more wasted work
/// past convergence). Must be >= 1.
/// @param gmres_mixed_precision When true, runs the inner Arnoldi build in
/// float32 while keeping `x`/`rhs`/the true per-restart residual in
/// float64 -- see `AMEnNativeOptions::gmres_mixed_precision` and
/// `gmres_solve_cpu`'s identical parameter.
torch::Tensor gmres_solve_gpu(const FoldedLocalOperator& op,
  const torch::Tensor& rhs, const torch::Tensor& x0, int max_iterations,
  int restarts, double rel_tol, const LocalPreconditioner* prec = nullptr,
  int check_interval = 8, bool gmres_mixed_precision = false);

} // namespace ttnte::linalg::amen
