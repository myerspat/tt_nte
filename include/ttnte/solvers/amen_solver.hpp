#pragma once

#include "ttnte/linalg/amen/amen_config.hpp"
#include "ttnte/linalg/format_type.hpp"
#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/utils/exception.hpp"

namespace ttnte::solvers {

/// @brief The AMEn local solver. Dispatches to either ttnte's own native
/// AMEn implementation (`AMEnBackend::NATIVE`, the default) or the vendored
/// torchTT implementation (`AMEnBackend::TORCHTT`), see `backend_`.
class AMEnSolver : public LocalSolver {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<AMEnSolver>;

protected:
  // =================================================================
  // Protected data
  /// Number of sweeps
  int nswp_;
  /// Relative residual.
  double eps_;
  /// Minimum allowed truncation tolerance.
  double eps_floor_;
  /// Forcing for truncation tolerance in an inexact solver.
  double eps_forcing_;
  /// Maximum allowed rank.
  int max_rank_;
  /// The largest size before switching from a direct solver to GMRES.
  int max_full_;
  /// The rank enrichment size.
  int kickrank_;
  /// ALS enrichment size.
  int kick2_;
  /// Number of GMRES iterations for each subproblem.
  int local_iterations_;
  /// The number of restarts in GMRES.
  int resets_;
  /// Show output.
  bool verbose_;
  /// What preconditioner to use.
  linalg::AMEnPreconditioner preconditioner_;
  /// Which AMEn implementation to dispatch to.
  linalg::AMEnBackend backend_;
  /// Tuning knobs specific to `AMEnBackend::NATIVE`.
  linalg::AMEnNativeOptions native_opts_;
  /// True once `native_opts_.rank_freeze_eps` has triggered (sticky --
  /// never resets).
  bool rank_frozen_ = false;

  // =================================================================
  // Protected constructors
  AMEnSolver(int nswp = 22, double eps = 1e-10, double eps_forcing = 0.01,
    int max_rank = std::numeric_limits<int>::max(), int max_full = 500,
    int kickrank = 4, int kick2 = 0, int local_iterations = 40, int resets = 2,
    bool verbose = false,
    linalg::AMEnPreconditioner preconditioner = linalg::AMEnPreconditioner::NONE,
    linalg::AMEnBackend backend = linalg::AMEnBackend::NATIVE,
    linalg::AMEnNativeOptions native_opts = linalg::AMEnNativeOptions{})
    : nswp_(nswp), eps_(eps), max_rank_(max_rank), max_full_(max_full),
      kickrank_(kickrank), kick2_(kick2), local_iterations_(local_iterations),
      resets_(resets), verbose_(verbose), preconditioner_(preconditioner),
      backend_(backend), native_opts_(native_opts)
  {
    if (nswp_ < 1 || eps_ < 0 || max_rank_ < 1 || max_full < 0 ||
        kickrank < 0 || kick2 < 0 || local_iterations_ < 1 || resets_ < 1) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`nswp`, `max_rank`, `local_iterations`, and `resets` must be greater\n"
        "than or equal to 1 and `eps`, `max_full`, `kickrank`, and `kick2`\n"
        "must be greater than or equal to 0");
    }

    if (preconditioner == linalg::AMEnPreconditioner::RANK1 &&
        backend != linalg::AMEnBackend::NATIVE) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`AMEnPreconditioner::RANK1` is only supported with "
        "`AMEnBackend::NATIVE` -- the torchTT backend has no rank-1 "
        "preconditioner");
    }

    if (native_opts.rank_freeze_eps > 0 &&
        backend != linalg::AMEnBackend::NATIVE) {
      throw utils::runtime_error("ttnte::solvers::AMEnSolver::AMEnSolver",
        "`rank_freeze_eps` is only supported with `AMEnBackend::NATIVE` -- "
        "the torchTT backend has no zero-enrichment code path");
    }

    eps_floor_ = eps;
    eps_forcing_ = eps_forcing;
  }

public:
  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new AMEn solver instance.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new AMEnSolver(std::forward<Args>(args)...));
  }

  /// @brief Solve the local linear system.
  /// @param local_system The local linear system to be solved.
  void solve(const linalg::LinearSystem::Ptr& local_system) override final;

  /// @brief Update min_error_ (via LocalSolver), then force eps_ toward
  /// eps_floor_ as min_error_ improves: eps_ = max(eps_floor_, eps_forcing_ *
  /// min_error_). Once eps_ drops to or below
  /// `native_opts_.rank_freeze_eps` (if enabled), permanently disables
  /// enrichment for every subsequent solve() call -- see `rank_freeze_eps`'s
  /// doc comment. Since this runs strictly after the solve() call whose
  /// error triggered it, that solve already completed as a normal,
  /// enrichment-active AMEn solve with its usual post-solve round; freezing
  /// only affects solve() calls from this point on.
  void update_convergence_criteria(double error) override
  {
    LocalSolver::update_convergence_criteria(error);
    eps_ = std::max(eps_floor_, eps_forcing_ * min_error_);

    if (!rank_frozen_ && native_opts_.rank_freeze_eps > 0 &&
        eps_ <= native_opts_.rank_freeze_eps) {
      rank_frozen_ = true;
      kickrank_ = 0;
      kick2_ = 0;
      native_opts_.als_residual_rank = 0;
    }
  }

  // =================================================================
  // Public getters / setters
  /// @return The current truncation tolerance of the solver.
  double get_eps() const override final { return eps_; }
  /// @return The maximum rank.
  int64_t get_max_rank() const final override
  {
    return static_cast<int64_t>(max_rank_);
  }
  /// @return Whether rank freezing (`native_opts_.rank_freeze_eps`) has
  /// triggered.
  bool is_rank_frozen() const noexcept { return rank_frozen_; }

  /// @return Always FormatType::TENSOR_TRAIN.
  linalg::FormatType get_state_format() override final
  {
    return linalg::FormatType::TENSOR_TRAIN;
  }
};

} // namespace ttnte::solvers
