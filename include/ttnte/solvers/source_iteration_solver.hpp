#pragma once

#include "ttnte/linalg/amen/amen_config.hpp"
#include "ttnte/linalg/format_type.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/utils/exception.hpp"
#include <limits>

namespace ttnte::solvers {

/// @brief Local solver that iterates the scattering source explicitly
/// (classical "source iteration" -- Picard iteration on scattering) instead
/// of inverting it jointly with streaming/removal in one linear system, the
/// way AMEnSolver does by default. Requires a LinearSystem assembled with
/// `physics::DGTransportAssemblerConfig::source_iterate_scattering = true`
/// (`LinearSystem::get_scatter_op()` defined, scattering NOT folded into
/// `get_interior_op()`) -- see `LocalSolver::handles_scatter_source()`,
/// which this class overrides to `true`, and the mismatch guard in
/// `LocalSolver::presolve()`.
///
/// Each `solve()` call runs source-iteration (SI) sweeps to full internal
/// convergence before returning control to the outer DD/eigenvalue
/// iteration: sweep `m` solves `A*psi = fixed_rhs + scatter_op*psi^(m-1)`
/// via an AMEn sub-solve, warm-started from `psi^(m-1)` -- the same
/// warm-starting a direct `AMEnSolver` solve already gets from its previous
/// outer iterate. This is architecturally distinct from defect correction
/// (see the `use_defect_correction` dead end, memory
/// `project_amen_defect_correction_dead_end`): that reformulation cold-started
/// its correction from zero every outer iteration, which turned out to be
/// the dominant cost on a marginal DD map. Source iteration's `psi` warm-starts
/// across sweeps the whole time, so it doesn't inherit that failure mode.
///
/// This class does not compose an `AMEnSolver` for its inner sub-solves --
/// `AMEnSolver::solve()` only accepts a full `LinearSystem::Ptr`, and
/// constructing/packing one per SI sweep would repeat expensive flat-buffer
/// work every sweep for no benefit. It calls `linalg::amen_solve()` directly
/// instead -- the same free function `AMEnSolver::solve()` itself calls --
/// which does mean this class's AMEn sub-solve tuning knobs
/// (`nswp`/`max_rank`/`kickrank`/... below) duplicate `AMEnSolver`'s own
/// constructor surface rather than sharing it; a small, deliberate tradeoff
/// for a robust, non-hacky per-sweep solve.
///
/// Uses a two-level (nested) forcing sequence, tightening from loose to
/// tight in both directions:
///   - Outer, DD-level: `eps_ = max(eps_floor_, eps_forcing_ * min_error_)`,
///     identical in shape to `AMEnSolver`'s own mechanism -- `min_error_` is
///     the running-best value passed to `update_convergence_criteria()` by
///     the outer DD/eigenvalue loop.
///   - Inner, SI-level: `eps_si = max(eps_, eps_forcing_si_ * min_error_si_)`
///     -- `eps_` (the outer target) is the floor SI tightens down to, never
///     over-converging past what the outer loop actually needs this
///     iteration; `min_error_si_` is the running-best SI residual so far
///     *within this one `solve()` call*. Resolving `A*psi = fixed +
///     scatter_op*psi^(m-1)` tightly while `psi^(m-1)` is still a poor
///     estimate wastes effort on stale data -- the same argument that
///     justifies the outer mechanism, one level down.
///
/// SI convergence is checked against the TRUE SI residual
/// `norm(A*psi - fixed_rhs - scatter_op*psi) / norm(fixed_rhs +
/// scatter_op*psi)`, not successive-iterate agreement -- the latter has the
/// same blind spot already diagnosed for the DD-level boundary-only Schwarz
/// metric (a self-consistent-but-wrong plateau looks identical to real
/// convergence).
class SourceIterationSolver : public LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<SourceIterationSolver>;

protected:
  // =================================================================
  // Protected data
  /// Maximum number of AMEn sweeps per SI sub-solve.
  int nswp_;
  /// Maximum allowed rank.
  int max_rank_;
  /// The largest local-subproblem size before switching from GMRES to a
  /// direct solver.
  int max_full_;
  /// The rank enrichment size.
  int kickrank_;
  /// ALS enrichment size.
  int kick2_;
  /// Number of GMRES iterations per local subproblem.
  int local_iterations_;
  /// The number of restarts in GMRES.
  int resets_;
  /// Show per-sweep AMEn output for each SI sub-solve.
  bool verbose_;
  /// Local preconditioner for each AMEn sub-solve.
  linalg::AMEnPreconditioner preconditioner_;
  /// Which AMEn implementation each sub-solve dispatches to.
  linalg::AMEnBackend backend_;
  /// Tuning knobs specific to `AMEnBackend::NATIVE`.
  linalg::AMEnNativeOptions native_opts_;

  /// Outer, DD-level truncation tolerance -- see
  /// `update_convergence_criteria()`.
  double eps_;
  /// Minimum allowed value of `eps_`.
  double eps_floor_;
  /// Forcing for `eps_` toward `eps_floor_` as the outer loop's `min_error_`
  /// improves.
  double eps_forcing_;

  /// Maximum number of source-iteration sweeps per `solve()` call.
  int max_si_sweeps_;
  /// Forcing for the inner, SI-level `eps_si` toward `eps_` as SI's own
  /// `min_error_si_` improves.
  double eps_forcing_si_;
  /// Running-best SI residual seen so far within the current `solve()`
  /// call. Reset at the start of every `solve()`.
  double min_error_si_ = 1.0;
  /// Number of SI sweeps the last `solve()` call actually ran (diagnostic).
  int last_si_sweeps_ = 0;
  /// SI residual after the last sweep of the last `solve()` call
  /// (diagnostic).
  double last_si_residual_ = std::numeric_limits<double>::max();

  // =================================================================
  // Protected constructor
  SourceIterationSolver(int nswp = 22, double eps = 1e-10,
    double eps_forcing = 0.01, int max_rank = std::numeric_limits<int>::max(),
    int max_full = 500, int kickrank = 4, int kick2 = 0,
    int local_iterations = 40, int resets = 2, bool verbose = false,
    linalg::AMEnPreconditioner preconditioner =
      linalg::AMEnPreconditioner::NONE,
    linalg::AMEnBackend backend = linalg::AMEnBackend::NATIVE,
    linalg::AMEnNativeOptions native_opts = linalg::AMEnNativeOptions {},
    int max_si_sweeps = 25, double eps_forcing_si = 0.1)
    : nswp_(nswp), max_rank_(max_rank), max_full_(max_full),
      kickrank_(kickrank), kick2_(kick2), local_iterations_(local_iterations),
      resets_(resets), verbose_(verbose), preconditioner_(preconditioner),
      backend_(backend), native_opts_(native_opts), eps_(eps), eps_floor_(eps),
      eps_forcing_(eps_forcing), max_si_sweeps_(max_si_sweeps),
      eps_forcing_si_(eps_forcing_si)
  {
    if (nswp_ < 1 || eps_ < 0 || max_rank_ < 1 || max_full < 0 ||
        kickrank < 0 || kick2 < 0 || local_iterations_ < 1 || resets_ < 1 ||
        max_si_sweeps_ < 1 || eps_forcing_si_ <= 0) {
      throw utils::runtime_error(
        "ttnte::solvers::SourceIterationSolver::SourceIterationSolver",
        "`nswp`, `max_rank`, `local_iterations`, `resets`, and "
        "`max_si_sweeps` must be greater than or equal to 1, `eps`, "
        "`max_full`, `kickrank`, and `kick2` must be greater than or equal "
        "to 0, and `eps_forcing_si` must be strictly positive");
    }
  }

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new SourceIterationSolver instance.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new SourceIterationSolver(std::forward<Args>(args)...));
  }

  /// @brief Solve the local linear system by source-iterating scattering to
  /// full internal convergence -- see the class documentation.
  /// @param local_system The local linear system to be solved. Must have
  /// been assembled with `source_iterate_scattering = true`
  /// (`get_scatter_op()` defined) -- see the guard in
  /// `LocalSolver::presolve()`.
  void solve(const linalg::LinearSystem::Ptr& local_system) override final;

  /// @brief Always true -- this solver consumes
  /// `LinearSystem::get_scatter_op()` itself. See
  /// `LocalSolver::handles_scatter_source()`.
  bool handles_scatter_source() const noexcept override final { return true; }

  /// @brief Update `min_error_` (via LocalSolver), then force the outer,
  /// DD-level `eps_` toward `eps_floor_` as `min_error_` improves --
  /// identical in shape to `AMEnSolver::update_convergence_criteria()`.
  void update_convergence_criteria(
    double error, double rank_metric = 0.0) override
  {
    LocalSolver::update_convergence_criteria(error, rank_metric);
    eps_ = std::max(eps_floor_, eps_forcing_ * min_error_);
  }

  // =================================================================
  // Public getters / setters
  /// @return The current outer, DD-level truncation tolerance -- the floor
  /// each `solve()` call's inner SI-level forcing tightens down to.
  double get_eps() const override final { return eps_; }
  /// @return The maximum rank.
  int64_t get_max_rank() const override final
  {
    return static_cast<int64_t>(max_rank_);
  }
  /// @return Number of SI sweeps the last `solve()` call actually ran.
  int get_last_si_sweeps() const noexcept { return last_si_sweeps_; }
  /// @return The true SI residual after the last sweep of the last
  /// `solve()` call.
  double get_last_si_residual() const noexcept { return last_si_residual_; }

  /// @return Always FormatType::TENSOR_TRAIN.
  linalg::FormatType get_state_format() override final
  {
    return linalg::FormatType::TENSOR_TRAIN;
  }
};

} // namespace ttnte::solvers
