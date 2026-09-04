#include "ttnte/solvers/source_iteration_solver.hpp"
#include "ttnte/linalg/ops.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
void SourceIterationSolver::solve(const linalg::LinearSystem::Ptr& local_system)
{
  auto [A, b_fixed, x0] = presolve(local_system);

  last_si_sweeps_ = 0;
  min_error_si_ = 1.0;
  last_si_residual_ = std::numeric_limits<double>::max();

  if (!b_fixed.defined()) {
    // No source/boundary forcing at all this iteration -- nothing to
    // source-iterate against. A*psi=0 -> psi=0, matching AMEnSolver::solve()'s
    // own convention for this degenerate case.
    linalg::State psi = linalg::State::zeros(linalg::FormatType::TENSOR_TRAIN,
      A.as_tt().get_n_modes(), A.get_device(), A.get_dtype());
    postsolve(local_system, std::move(psi));
    return;
  }

  const linalg::Operator& scatter_op = local_system->get_scatter_op();
  const linalg::Operator& moment_projector =
    local_system->get_moment_projector();

  // Checks self-consistency BEFORE solving, not after: A*psi = fixed +
  // scatter_op*psi requires evaluating the scattering source at the SAME
  // psi used on the left-hand side, so this reuses the just-computed
  // q_scatter/rhs as that sweep's AMEn RHS if not yet converged, rather than
  // paying for a second mv(scatter_op, ...) at the new psi after solving.
  // Runs one extra time past max_si_sweeps_ solves so the reported
  // diagnostics (last_si_sweeps_/last_si_residual_) always reflect the
  // actual final psi, never a stale pre-update value.
  linalg::State psi = x0;
  for (int m = 0; m <= max_si_sweeps_; m++) {
    linalg::State q_scatter = linalg::mv(scatter_op, psi);
    linalg::State rhs = round_conserved(b_fixed + q_scatter, moment_projector);

    // True SI residual -- not successive-iterate agreement, which has the
    // same blind spot as the DD-level boundary-only Schwarz metric (see the
    // class documentation).
    linalg::State residual_state = linalg::mv(A, psi) - rhs;
    const double rhs_norm = rhs.norm();
    const double si_residual =
      rhs_norm > 0.0 ? residual_state.norm() / rhs_norm : residual_state.norm();

    last_si_sweeps_ = m;
    last_si_residual_ = si_residual;
    if (si_residual < eps_ || m == max_si_sweeps_) {
      break;
    }
    if (si_residual < min_error_si_ && si_residual > 0) {
      min_error_si_ = si_residual;
    }

    // Inner, SI-level forcing: never tighter than the outer DD-level target
    // eps_, but starts loose and tightens as SI's own residual improves --
    // see the class documentation.
    const double eps_si = std::max(eps_, eps_forcing_si_ * min_error_si_);

    // psi warm-starts every sweep, unlike defect correction's cold-started
    // correction (see the class documentation).
    psi = linalg::amen_solve(A, rhs, psi, nswp_, eps_si, max_rank_, max_full_,
      kickrank_, kick2_, local_iterations_, resets_, verbose_, preconditioner_,
      backend_, native_opts_);
    psi = round_conserved(std::move(psi), moment_projector);
  }

  postsolve(local_system, std::move(psi));
}

} // namespace ttnte::solvers
