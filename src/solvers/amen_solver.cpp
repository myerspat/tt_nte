#include "ttnte/solvers/amen_solver.hpp"

namespace ttnte::solvers {

// =================================================================
// Public methods
void AMEnSolver::solve(const linalg::LinearSystem::Ptr& local_system)
{
  auto [A, b, x0] = presolve(local_system);

  linalg::State x;
  if (b.defined()) {
    // Run AMEn solver
    x = linalg::amen_solve(A, b, x0, nswp_, eps_, max_rank_, max_full_,
      kickrank_, kick2_, local_iterations_, resets_, verbose_, preconditioner_,
      backend_, native_opts_);
    // Once rank-frozen, the per-core truncation loop inside amen_sweep
    // already keeps every bond's rank minimal throughout a pure-ALS sweep
    // (bounded above by the warm start, never past it -- see
    // rank_freeze_eps's doc comment), so an extra global round here would
    // be redundant.
    if (!rank_frozen_) {
      x.round_(eps_, max_rank_);
    }
  } else {
    x = linalg::State::zeros(linalg::FormatType::TENSOR_TRAIN,
      A.as_tt().get_n_modes(), A.get_device(), A.get_dtype());
  }

  postsolve(local_system, std::move(x));
}

} // namespace ttnte::solvers
