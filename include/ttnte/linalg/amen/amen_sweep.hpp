#pragma once

// The core AMEn sweep and its thin TTEngine-level wrapper, used by
// `amen_solve_native()`. The sweep control flow itself (backward
// orthogonalization pass, forward local-solve/truncate/enrich pass) is
// device-agnostic -- every device-specific tuning decision already lives
// one layer down, inside the primitives it calls (FoldedLocalOperator,
// gmres_solve's CPU/GPU strategy split, qless_orthogonalize). See
// src/linalg/amen/amen_sweep.cpp for the algorithmic citations.

#include "ttnte/linalg/amen/amen_config.hpp"
#include "ttnte/linalg/tt_engine.hpp"
#include <vector>

namespace ttnte::linalg::amen {

/// @brief The core AMEn sweep, operating on the classic 3-D-core convention:
/// vector cores `[r_l, N_k, r_r]`, operator cores `[R_l, N_k, N_k, R_r]`.
/// @param A_cores Operator TT cores, 4-D `[R_l, N_k, N_k, R_r]` per core.
/// @param b_cores Right-hand-side TT cores, 3-D `[r_l, N_k, r_r]` per core.
/// @param x_cores Initial guess / warm start TT cores, same convention as
/// `b_cores`; overwritten in place sweep-by-sweep and returned as the
/// solution.
/// @param N Mode sizes per core.
/// @param nswp Maximum number of forward+backward half-sweep pairs.
/// @param eps Target relative residual tolerance; a sweep whose `max_res`
/// drops below this becomes the final ("last", enrichment-skipped)
/// polishing sweep.
/// @param max_rank Maximum allowed TT rank per bond.
/// @param max_full Local-subproblem size below which a dense direct solve
/// (`torch::linalg_solve`) is used instead of `gmres_solve`.
/// @param kickrank Adaptive-SVD enrichment rank (`FULL`/`SIMPLIFIED` modes).
/// @param kick2 Extra random-enrichment columns added alongside `kickrank`.
/// @param local_iterations Krylov subspace dimension per GMRES restart.
/// @param resets Maximum number of GMRES restarts per local solve.
/// @param verbose When true, prints a per-sweep progress summary (max
/// residual, current TT ranks) to stdout.
/// @param preconditioner Local preconditioner selector; only `NONE`,
/// `LOCAL_C_PREC`, and `LOCAL_R_PREC` are valid here -- `RANK1` is handled
/// one layer up, in `amen_solve_dispatch`, before this function is called.
/// @param tsqr_block_size Block size for `qless_orthogonalize`'s blocked TSQR.
/// @param use_qless_tsqr Whether to use Q-less blocked TSQR instead of plain
/// `torch::linalg_qr` for orthogonalization steps.
/// @param mode Enrichment strategy; see `AMEnEnrichmentMode`.
/// @param als_residual_rank Fixed enrichment-basis rank, only used when
/// `mode == AMEnEnrichmentMode::ALS_FIXED_RANK`. A value of 0 (under
/// `ALS_FIXED_RANK`) or `kickrank == 0 && kick2 == 0` (otherwise) disables
/// enrichment entirely, collapsing the sweep to pure ALS (local solve +
/// truncation only; rank can shrink but never grow past the warm start).
/// @param use_local_forcing Whether to loosen each core's GMRES target
/// tolerance based on its own current residual (inexact-Newton style)
/// instead of always solving to `eps`.
/// @param gmres_forcing_ceiling Upper bound on the loosened per-core GMRES
/// tolerance when `use_local_forcing` is enabled.
/// @param use_gpu_batched_gmres Whether local solves on CUDA tensors use
/// the fixed-budget "batched" GPU GMRES strategy (see `gmres_solve`'s
/// `prefer_incremental`) rather than the incremental CPU-style strategy.
/// @param proximal_regularization See
/// `AMEnNativeOptions::proximal_regularization`. Only actually applied while
/// enrichment is disabled (pure ALS) -- ignored otherwise, regardless of the
/// value passed.
/// @return The solution's TT cores, same convention as `x_cores`.
std::vector<torch::Tensor> amen_sweep(std::vector<torch::Tensor> A_cores,
  std::vector<torch::Tensor> b_cores, std::vector<torch::Tensor> x_cores,
  const std::vector<int64_t>& N, int nswp, double eps, int64_t max_rank,
  int64_t max_full, int64_t kickrank, int64_t kick2, int local_iterations,
  int resets, bool verbose, AMEnPreconditioner preconditioner,
  int tsqr_block_size, bool use_qless_tsqr, AMEnEnrichmentMode mode,
  int64_t als_residual_rank, bool use_local_forcing,
  double gmres_forcing_ceiling, bool use_gpu_batched_gmres,
  double proximal_regularization = 0.0);

/// @brief Shared TTEngine-level wrapper around `amen_sweep`: validates
/// shapes, optionally applies the rank-1 preconditioner, converts to/from
/// the 3-D core convention, and calls `amen_sweep`. Used by
/// `amen_solve_native()` -- the sweep itself dispatches to the right
/// device-specific strategy internally (e.g. `gmres_solve`'s CPU/GPU
/// split), so this wrapper's logic doesn't need to differ by device.
/// @param A Operator in `TTEngine`/4-D-core form (`[R_l, m, n, R_r]`);
/// requires `m == n` per core (square).
/// @param b Right-hand side in `TTEngine`/`m_mode == 1` TT-vector form.
/// @param x0 Optional initial guess / warm start, same convention as `b`;
/// defaults to an all-ones rank-1 TT when absent.
/// @param nswp,eps,max_rank,max_full,kickrank,kick2,local_iterations,resets,
/// verbose See the matching parameters of `amen_sweep`.
/// @param preconditioner Preconditioner selector; unlike `amen_sweep`, all
/// four `AMEnPreconditioner` values are accepted here -- `RANK1` triggers
/// the global TT-rank-1 sandwiching of `A`/`b`/`x0` before dispatching to
/// `amen_sweep` (with `preconditioner` downgraded to `NONE` for that call,
/// since the two preconditioning strategies are never combined).
/// @param native_opts Native-backend tuning knobs; see `AMEnNativeOptions`.
/// @param caller Fully-qualified call site, used in error messages.
/// @return The solution as a `TTEngine` TT-vector.
TTEngine amen_solve_dispatch(const TTEngine& A, const TTEngine& b,
  std::optional<TTEngine> x0, int nswp, double eps, int max_rank, int max_full,
  int kickrank, int kick2, int local_iterations, int resets, bool verbose,
  AMEnPreconditioner preconditioner, const AMEnNativeOptions& native_opts,
  const char* caller);

} // namespace ttnte::linalg::amen
