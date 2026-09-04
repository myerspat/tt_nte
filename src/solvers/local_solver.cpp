#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/linalg/ops.hpp"
#include "ttnte/utils/exception.hpp"
#include <vector>

namespace ttnte::solvers {

// =================================================================
// Public methods
std::tuple<linalg::Operator, linalg::State, linalg::State>
LocalSolver::presolve(const linalg::LinearSystem::Ptr& sys) const
{
  // Guard against a mismatch between how this LinearSystem was assembled
  // (source_iterate_scattering) and which solver is being used to solve it
  // -- silent, not a crash, if unchecked (scattering vanishes or gets
  // double-counted). See LocalSolver::handles_scatter_source().
  if (sys->get_scatter_op().defined() != handles_scatter_source()) {
    throw utils::runtime_error("ttnte::solvers::LocalSolver::presolve",
      sys->get_scatter_op().defined()
        ? "This LinearSystem was assembled with source_iterate_scattering "
          "(scatter_op is kept separate from interior_op), but this solver "
          "does not handle an explicit scattering source -- solving it "
          "would silently drop scattering from the physics. Use a solver "
          "with handles_scatter_source() == true (e.g. "
          "SourceIterationSolver), or assemble without "
          "source_iterate_scattering."
        : "This solver expects a scattering source it iterates explicitly "
          "(handles_scatter_source() == true), but this LinearSystem's "
          "scatter_op is undefined -- it was assembled with scattering "
          "already folded into interior_op, so iterating it separately "
          "would double-count it. Assemble with source_iterate_scattering "
          "= true, or use a solver with handles_scatter_source() == "
          "false.");
  }

  // Get the operators for the linear system
  linalg::Operator A = sys->get_interior_op();
  linalg::State x0 = sys->get_state();

  // Gather boundary contributions and sum them in one batched pass (see
  // linalg::direct_sum) rather than folding them together one at a time --
  // that would reallocate and copy the full, ever-growing set of cores at
  // every step. The result is a fresh State (not aliased with any
  // coupling.recv_buffer or EigenSource::state_), so it's safe to combine
  // with the source below via the binary operator+.
  std::vector<linalg::State> boundary_terms;
  for (auto& coupling : sys->get_couplings()) {
    if (coupling.recv_buffer.defined()) {
      boundary_terms.push_back(std::move(coupling.recv_buffer));
      coupling.recv_buffer = linalg::State();
    }
  }

  const bool has_boundary = !boundary_terms.empty();
  linalg::State boundary_sum =
    has_boundary ? linalg::direct_sum(boundary_terms) : linalg::State();

  // Build the RHS: fission/fixed source + boundary.
  // When boundary is present, use binary operator+ so the result owns fresh
  // StateData — this prevents aliasing through the shallow-copy State handle
  // from corrupting EigenSource::state_ via operator+=.
  linalg::State b;
  const auto* src = sys->get_source().get();
  if (src && src->get_state().defined()) {
    if (has_boundary) {
      b = src->get_state() + boundary_sum;
      b = round_conserved(std::move(b), sys->get_moment_projector());
    } else {
      b = src->get_state();
    }
  } else if (has_boundary) {
    b = std::move(boundary_sum);
    b = round_conserved(std::move(b), sys->get_moment_projector());
  }

  return std::make_tuple(std::move(A), std::move(b), std::move(x0));
}

void LocalSolver::postsolve(
  const linalg::LinearSystem::Ptr& sys, const linalg::State& x) const
{
  const auto& x0 = sys->get_state();

  // Compute per-coupling Schwarz convergence error: compare the OUTGOING
  // partial current (the boundary-narrowed angular flux reduced via
  // coupling.current_op -- the (Omega . n)_+ upwind mask times the angular
  // quadrature weights) at the face of x with the face of x0 along each
  // internal boundary dimension. Falls back to the raw angular-flux face
  // if current_op isn't defined (not yet built for every format). A rough
  // rounding is applied to the diff to keep its rank manageable -- only a
  // convergence indicator is needed, not a precise residual.
  for (auto& coupling : sys->get_couplings()) {
    const size_t bdim = static_cast<size_t>(x.ndimension()) -
                        coupling.connection.mapping.flip.size() - 2 +
                        coupling.dim;
    linalg::State face_new = x.narrow(bdim, coupling.is_upper ? -1 : 0, 1);
    linalg::State face_old = x0.narrow(bdim, coupling.is_upper ? -1 : 0, 1);

    if (coupling.current_op.defined()) {
      face_new = linalg::mv(coupling.current_op, face_new);
      face_old = linalg::mv(coupling.current_op, face_old);
    }

    linalg::State diff = face_new - face_old;
    diff.round_(get_eps() * 0.01, get_max_rank());
    const double n_diff = diff.norm();
    const double n_old = face_old.norm();
    coupling.sq_diff = n_diff * n_diff / 2.0;
    coupling.sq_prev = n_old * n_old / 2.0;
  }

  // Update the linear system
  sys->set_state(std::move(x));
}

void LocalSolver::init(const Systems& local_systems)
{
  local_systems_ = local_systems;
}

void LocalSolver::step()
{
  for (auto& sys : local_systems_) {
    solve(sys);
  }
}

} // namespace ttnte::solvers
