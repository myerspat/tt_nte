#pragma once

#include "ttnte/linalg/linear_system.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/solvers/solver.hpp"
#include <memory>
#include <tuple>

namespace ttnte::solvers {

/// @brief The local solver base class.
class LocalSolver : public Solver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<LocalSolver>;

protected:
  // =================================================================
  // Protected data
  Systems local_systems_;
  double min_error_ = 1.0;

  /// Whether to round via round_conserved() (moment-preserving) instead of
  /// a plain round_() wherever this base class rounds a State -- currently
  /// just presolve()'s RHS. Concrete solvers (e.g. AMEnSolver) expose this
  /// as a constructor parameter and apply it themselves too, at their own
  /// solution-round call site.
  bool preserve_moments_ = false;
  /// Multiplicative relaxation applied to get_eps() for the non-moment
  /// remainder's truncation tolerance when preserve_moments_ is set:
  /// remainder_eps = moment_remainder_relaxation_ * get_eps(). Default 1.0
  /// -- identical to get_eps() itself, the safe no-op starting point.
  /// Deliberately relative rather than an absolute tolerance: get_eps()
  /// itself is not constant (e.g. AMEnSolver tightens it every outer
  /// iteration via eps_forcing_ * min_error_), so a fixed absolute
  /// remainder eps would drift stale -- too loose early on (when get_eps()
  /// itself starts loose) or, worse, effectively too TIGHT once forcing has
  /// driven get_eps() below it, silently giving back none of the intended
  /// slack right when the solve is working hardest. A relaxation factor
  /// stays proportionally looser than whatever get_eps() currently demands,
  /// tracking the same forcing schedule instead of fighting it.
  double moment_remainder_relaxation_ = 1.0;
  /// Truncation tolerance for the moment part itself; < 0 means "reuse
  /// get_eps()". The moment part is structurally low-rank already, so this
  /// is expected to stay tight regardless.
  double moment_eps_ = -1.0;
  /// Maximum rank for the moment part; < 0 means "reuse get_max_rank()".
  int64_t moment_max_rank_ = -1;

  // =================================================================
  // Protected methods
  /// @brief Round `x`, preserving its projection onto `moment_projector`
  /// exactly when preserve_moments_ is set and `moment_projector` is
  /// defined (see linalg::round_conserved()); otherwise falls back to a
  /// plain round_() at get_eps()/get_max_rank(), identical to previous
  /// behavior.
  /// @param x The state to round.
  /// @param moment_projector The LinearSystem's moment projector (may be
  /// undefined).
  /// @return The rounded state.
  linalg::State round_conserved(
    linalg::State x, const linalg::Operator& moment_projector) const
  {
    if (!preserve_moments_ || !moment_projector.defined()) {
      x.round_(get_eps(), get_max_rank());
      return x;
    }

    const double remainder_eps = moment_remainder_relaxation_ * get_eps();
    const double m_eps = moment_eps_ < 0 ? get_eps() : moment_eps_;
    const int64_t m_max_rank =
      moment_max_rank_ < 0 ? get_max_rank() : moment_max_rank_;

    return linalg::round_conserved(std::move(x), moment_projector,
      remainder_eps, get_max_rank(), m_eps, m_max_rank);
  }

public:
  virtual ~LocalSolver() = default;

  // =================================================================
  // Public methods
  /// @brief Whether this solver consumes LinearSystem::get_scatter_op()
  /// itself (i.e. iterates scattering explicitly -- see
  /// solvers::SourceIterationSolver) rather than expecting it already folded
  /// into the interior operator. False for every solver except one that
  /// overrides it (SourceIterationSolver). presolve() uses this to guard
  /// against a mismatch between how a LinearSystem was assembled
  /// (source_iterate_scattering) and which solver is being used to solve
  /// it -- a mismatch is silent, not a crash (scattering either vanishes
  /// from the physics or gets double-counted), so it's checked explicitly.
  virtual bool handles_scatter_source() const noexcept { return false; }

  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  virtual void solve(const linalg::LinearSystem::Ptr& local_system) = 0;

  /// @brief Compute the right-hand-side of the linear system based on the new
  /// boundary conditions.
  /// @param local_system The local linear system to be solved.
  /// @return A tuple of A, b, x0.
  std::tuple<linalg::Operator, linalg::State, linalg::State> presolve(
    const linalg::LinearSystem::Ptr& sys) const;

  /// @brief Post solve cleanup.
  /// @param local_system The local linear system to be solved.
  void postsolve(
    const linalg::LinearSystem::Ptr& sys, const linalg::State& x) const;

  /// @brief Store the systems to be solved (one per patch).
  void init(const Systems& local_systems) override;
  /// @brief Solve every registered system in turn.
  void step() override;
  /// @brief Track the best (smallest) error observed so far. Concrete
  /// solvers (e.g. AMEnSolver) override this to also derive their own
  /// forcing from min_error_ and consult an EnrichmentPolicy with
  /// `rank_metric`.
  void update_convergence_criteria(
    double error, double rank_metric = 0.0) override
  {
    if (error < min_error_ && error > 0) {
      min_error_ = error;
    }
  }

  // =================================================================
  // Public getters / setters
  const Systems& get_local_systems() const override { return local_systems_; }
  /// @brief Equivalent to init() -- re-registers the systems to be solved.
  void set_local_systems(const Systems& local_systems) override
  {
    init(local_systems);
  }
};

} // namespace ttnte::solvers
