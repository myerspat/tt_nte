#pragma once

#include <cstdint>

namespace ttnte::linalg {

/// @brief Selects which AMEn implementation `amen_solve()` dispatches to.
enum class AMEnBackend : uint8_t {
  /// The original vendored torchTT C++ implementation.
  TORCHTT = 0,
  /// The native ttnte implementation (rank-1 preconditioner, Q-less TSQR,
  /// ALS enrichment heuristic, folded cuBLAS contractions, fused CUDA
  /// kernels).
  NATIVE = 1
};

/// @brief Selects how the native AMEn solver tracks and enriches the
/// residual basis between sweeps.
enum class AMEnEnrichmentMode : uint8_t {
  /// Full AMEn: recompute the exact global-residual enrichment every step
  /// (matches the torchTT baseline's algorithm; used for validation).
  FULL = 0,
  /// Same SVD-truncated-to-`kickrank` basis extraction as `FULL`, but skips
  /// the backward half-sweep's residual recomputation (which would be
  /// estimating a core's residual before it's actually been re-solved this
  /// sweep) -- the only residual computed is the forward half-sweep's,
  /// immediately after a core's fresh local solve. Cheaper per sweep than
  /// `FULL`; the tradeoff is one fewer (stale) residual re-estimate per
  /// core per sweep, not a different algorithm.
  SIMPLIFIED = 1,
  /// Maintain a fixed-rank approximate residual TT, updated cheaply each
  /// half-sweep (pitts' "AMEn+ALS", Algorithm 5 of Roehrig-Zoellner et al.
  /// 2025).
  ALS_FIXED_RANK = 2
};

/// @brief Selects which preconditioner (if any) the AMEn solver applies.
/// `LOCAL_C_PREC`/`LOCAL_R_PREC` are cheap, per-core Jacobi-style local
/// preconditioners applied inside each core's GMRES solve (see
/// `amen::LocalPreconditioner`); `RANK1` is a global TT-rank-1
/// preconditioner applied once to the whole system before sweeping (see
/// `amen::Rank1Preconditioner`) and is only supported by `AMEnBackend::NATIVE`
/// (the vendored torchTT backend has no equivalent). Exactly one strategy is
/// active at a time -- they are not combined.
enum class AMEnPreconditioner : uint8_t {
  /// No preconditioning.
  NONE = 0,
  /// Cheap local preconditioner: both interfaces (left and right) are
  /// diagonalized -- one small `n x n` block per rank-index pair.
  LOCAL_C_PREC = 1,
  /// More accurate, more expensive local preconditioner: the right
  /// interface's full (non-diagonal) structure is kept.
  LOCAL_R_PREC = 2,
  /// Global TT-rank-1 preconditioner (Roehrig-Zoellner et al. 2025,
  /// Sec. 3.4.1), `AMEnBackend::NATIVE`-only.
  RANK1 = 3
};

// Empirically tuned (see benchmarks -- isolated `qless_orthogonalize` on
// synthetic matrices up to 50,000 rows, both devices): a single whole-matrix
// Cholesky-QR was *always* faster than any blocked tree-reduction at every
// size tested, on both CPU and GPU (e.g. CPU, 20,000 rows: 11ms unblocked
// vs. 31ms at block_size=2048; GPU, 50,000 rows: 10ms unblocked vs. 19ms at
// block_size=8192). Blocking exists for communication-avoiding/distributed
// or out-of-core scenarios (the TSQR literature's original motivation) --
// neither applies to ttnte's actual usage (single device, matrices that
// comfortably fit in memory), so it's pure overhead here. Set well above any
// row count a realistic AMEn sweep would produce (mode size x bond rank),
// so blocking effectively never triggers, while still capping memory if
// something unusually large is ever passed.
inline constexpr int DEFAULT_TSQR_BLOCK_SIZE = 8192;

/// @brief Tuning knobs specific to `AMEnBackend::NATIVE`.
struct AMEnNativeOptions {
  /// Use the Q-less, blocked TSQR for core orthogonalization/truncation
  /// instead of a plain dense QR/SVD.
  bool use_qless_tsqr = true;
  /// How the residual basis is tracked/enriched between sweeps.
  AMEnEnrichmentMode enrichment_mode = AMEnEnrichmentMode::ALS_FIXED_RANK;
  /// Fixed rank of the approximate residual TT when `enrichment_mode ==
  /// ALS_FIXED_RANK`.
  int als_residual_rank = 4;
  /// Row-block size for the Q-less TSQR reduction.
  int tsqr_block_size = DEFAULT_TSQR_BLOCK_SIZE;
  /// Use an inexact-Newton-style forcing term for the local GMRES
  /// tolerance: relax it (up to `gmres_forcing_ceiling`) when the local
  /// residual is still far from converged, tightening toward the per-core
  /// target as the sweep approaches `eps`. When off, every core's GMRES
  /// call targets the fixed per-core tolerance directly (matching the
  /// torchTT baseline's behavior).
  ///
  /// EXPERIMENTAL, NOT RECOMMENDED -- defaults to off and stays off.
  /// Confirmed via a real 2D neutron-transport regression case: with forcing
  /// on, the outer power iteration plateaus at ~2% error instead of
  /// converging to ~1e-6 (torchTT and this same native FULL-mode sweep with
  /// forcing off both converge cleanly). The original cause identified --
  /// the adaptive rank-truncation loop's acceptance threshold reusing the
  /// just-achieved (deliberately loosened) residual as a floor for how much
  /// rank to keep -- has been fixed (the truncation floor no longer uses
  /// `res_new` for a core whose solve was forced loose), but that fix did
  /// NOT resolve the plateau, so a second, not-yet-understood interaction
  /// remains (most likely: a deliberately under-solved core's *values*, not
  /// just its rank, persist unfixed through the rest of that sweep's
  /// `Phis`/`Phiz` interface tensors and the enrichment step derived from
  /// them). Do not enable without re-validating end-to-end.
  bool use_local_forcing = false;
  /// Upper bound on the forcing-relaxed local GMRES tolerance -- never
  /// relax past this, even when the local residual is very large. Ignored
  /// when `use_local_forcing` is false.
  double gmres_forcing_ceiling = 1e-1;
  /// On CUDA tensors, use the fixed-budget "batched" Arnoldi+lstsq GMRES
  /// strategy (no per-iteration host sync) instead of the incremental
  /// Givens-rotation strategy CPU always uses. Defaults to false (i.e.
  /// GPU uses the same incremental strategy as CPU by default): the batched
  /// strategy always builds the full `local_iterations`-sized Krylov basis
  /// with no early stop, which risks loss of orthogonality for larger
  /// `local_iterations` or harder local systems -- it trades that risk for
  /// avoiding host syncs. Turn on only after validating end-to-end accuracy
  /// on your own problems; not yet proven adequate on this project's
  /// regression tests.
  bool use_gpu_batched_gmres = false;
  /// Run the GMRES inner Arnoldi/Gram-Schmidt/Givens build in float32
  /// instead of the input's own (float64) dtype -- a mixed-precision
  /// iterative-refinement scheme, not a blanket precision switch: `x`, `b`,
  /// and the true residual recomputed at the top of every restart (`r = b -
  /// local_apply(x)`) all stay in float64 throughout, so any float32-
  /// rounding error introduced by one restart's inner loop is corrected by
  /// the next restart's full-precision residual recompute rather than
  /// accumulating. Follows the pattern in Haque, Shontz & Tu, "GPU-
  /// Accelerated Mixed Precision GMRES(m) with Varied Restarts" (HPEC 2025),
  /// Algorithm 6/7.
  ///
  /// Targets the confirmed dominant real-workload cost (dense GEMM/GEMV
  /// inside the per-core local operator apply, profiled via `torch.profiler`
  /// CUDA-activity tracing on the C5G7 pincell benchmark) on hardware with a
  /// large FP64:FP32 throughput gap (e.g. RTX Ada workstation GPUs, ~1/64).
  /// Applies to both `gmres_solve_cpu` (CPU and CUDA tensors alike) and
  /// `gmres_solve_gpu` -- i.e. every GMRES strategy this codebase has,
  /// regardless of device or `use_gpu_batched_gmres`. The FP64:FP32 gap that
  /// motivates this is far smaller on general-purpose CPUs (~2:1) than on
  /// RTX Ada GPUs, so the CPU win is expected to be modest -- validate
  /// before assuming it's worth enabling there.
  ///
  /// NOT YET VALIDATED against this project's real regression/benchmark
  /// accuracy tolerances at the time this option was added -- defaults to
  /// off. In particular, watch for the incremental Givens-rotation residual
  /// estimate (`beta[k+1]`, updated each iteration rather than recomputed
  /// from scratch) drifting from the true residual under reduced precision:
  /// Jang, Jolivet & Mary, "Mixed Precision Augmented GMRES" (NLAA 2026),
  /// show a structurally similar cheap-update-derived-from-an-exact-
  /// arithmetic-identity pattern (GMRES-DR's restart construction) can
  /// silently stagnate at low accuracy under reduced precision, though their
  /// specific failure mode (a collinearity property used for augmented-
  /// subspace restarts) doesn't apply here since this GMRES has no
  /// augmentation/deflation.
  bool gmres_mixed_precision = false;
  /// Proximal regularization weight for the per-core ALS solve: when active,
  /// each core's local system `B x = rhs` is solved as `(B + w I) x = rhs +
  /// w x_prev` instead (`w` = this field), damping the step toward the
  /// previous iterate -- the standard remedy for ALS "swamping" (Tomasi &
  /// Bro-style regularized ALS). Convergence bookkeeping (`max_res`, the
  /// rank-truncation floor, the eps-forcing term) is still measured against
  /// the true, unshifted residual `B x - rhs`; this only changes how the
  /// step itself is computed, never what counts as converged.
  ///
  /// Only takes effect once enrichment is disabled (i.e. pure ALS -- either
  /// an `AMEnSolver` `EnrichmentPolicy` disabled it, or the caller set zero
  /// enrichment directly); ignored otherwise. Rationale: while enrichment is
  /// active,
  /// rank is still adapting to reach the tightening truncation target, so
  /// the local solves aren't structurally swamped -- damping them there
  /// would just slow down otherwise-legitimate large corrections. The
  /// swamping this targets shows up specifically once a frozen rank is asked
  /// for a truncation tolerance tighter than it can represent: the per-core
  /// solve becomes ill-conditioned enough that its residual *increases*
  /// sweep-to-sweep instead of decreasing (confirmed via a real fixed-source
  /// regression case).
  ///
  /// Disabled (0.0) by default.
  double proximal_regularization = 0.0;
  /// Solves/truncates each core to `eps/(sqrt(d)*resid_damp)` rather than
  /// `eps` itself (Dolgov & Savostyanov's own `resid_damp` parameter,
  /// TT-Toolbox `amen_solve2.m` -- ttnte previously hardcoded this to `2.0`
  /// instead of exposing it). The forward sweep's termination check
  /// (`max_res < eps`) still compares against the raw, undivided `eps`, so
  /// this is the only knob controlling the margin between what a core's
  /// local solve/truncation targets and what the termination check needs to
  /// see. Confirmed via direct measurement on two independent problems
  /// (KAIST cruciform fixed-source benchmark, C5G7 pincell eigenvalue
  /// benchmark) that the default of `2.0` can leave a real gap: local solves
  /// can plateau at a residual roughly 20-25x (median, both problems) above
  /// what the per-core target actually is, preventing `max_res < eps` from
  /// ever firing even after the full sweep budget. Matches Dolgov's own
  /// default (`2.0`) unless raised; per his own documentation, raising it
  /// "may reduce a spurious noise from inexact local solutions, but increase
  /// CPU time."
  double resid_damp = 2.0;
};

} // namespace ttnte::linalg
