import pytest
import torch
import torchtt as tntt

from ttnte.linalg import State, Operator, LinearSystem, TTEngine, Source
from ttnte.solvers import AMEnSolver, SourceIterationSolver


def _split_diag_dominant_system(d, n, rank, dtype, device, seed):
    """A diagonally-dominant streaming+removal operator H and a separate,

    smaller-scale scattering operator S such that H - S stays invertible and
    well-conditioned -- lets a direct solve of (H - S)x = b (today's
    AMEnSolver convention) be compared against a source-iterated solve of
    Hx = b + S*x (SourceIterationSolver), which should converge to the same
    unique solution. Mirrors test_amen_solver.py's _diag_dominant_system
    construction, split into its two pieces.
    """
    torch.manual_seed(seed)
    op_ranks = [1] + [rank] * (d - 1) + [1]
    pert = tntt.random([(n, n)] * d, op_ranks, dtype=dtype).to(device)
    scales = torch.logspace(0, 0.5, d)
    H = pert * 0.01
    for i in range(d):
        H = H + scales[i] * tntt.eye([n] * d, dtype=dtype).to(device)

    # Scattering: small enough relative to H's diagonal dominance that H - S
    # stays invertible and well-conditioned, same spirit as pert's 0.01 scale
    # above -- this is a synthetic stand-in for a real scattering operator,
    # not physically motivated.
    S = 0.05 * tntt.random([(n, n)] * d, op_ranks, dtype=dtype).to(device)

    H = H.round(1e-13)
    S = S.round(1e-13)
    x_true = tntt.random([n] * d, [1, rank, rank, 1], dtype=dtype).to(device)
    b = ((H - S) @ x_true).round(1e-13)
    return H, S, b


def test_source_iteration_solver_matches_direct_combined_solve():
    """SourceIterationSolver source-iterating scattering (Hx = b + Sx) must.

    converge to the same solution a direct AMEnSolver solve of the combined
    system ((H - S)x = b) reaches -- source iteration only changes how the
    local solve is decomposed, not the linear system being solved.
    """
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    H, S, b = _split_diag_dominant_system(d, n, r, dtype, device, seed=10)

    H_eng = Operator(TTEngine(H.cores))
    S_eng = Operator(TTEngine(S.cores))
    b_eng = State(TTEngine(b.cores))
    combined_eng = Operator(TTEngine((H - S).round(1e-13).cores))

    torch.manual_seed(11)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)
    x0_eng = State(TTEngine(x0.cores))

    ls_direct = LinearSystem(combined_eng, source=Source(b_eng))
    ls_direct.state = x0_eng
    AMEnSolver(eps=1e-8, nswp=15).solve(ls_direct)

    ls_si = LinearSystem(H_eng, source=Source(b_eng), scatter_op=S_eng)
    ls_si.state = x0_eng
    si_solver = SourceIterationSolver(
        eps=1e-8, nswp=15, max_si_sweeps=40, eps_forcing_si=0.1
    )
    si_solver.solve(ls_si)

    assert si_solver.last_si_residual < 1e-8
    assert si_solver.last_si_sweeps < 40  # actually converged, not exhausted budget

    x_direct = tntt.TT([c.squeeze(2) for c in ls_direct.state.as_tt().cores])
    x_si = tntt.TT([c.squeeze(2) for c in ls_si.state.as_tt().cores])
    assert ((x_si - x_direct).norm() / x_direct.norm()).item() < 1e-4


def test_source_iteration_solver_no_forcing_zeros_state():
    """With no source and no boundary couplings at all, solve() must zero the state --
    same degenerate-case convention as AMEnSolver's direct solve (A*x = 0 -> x = 0)."""
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    H, S, _ = _split_diag_dominant_system(d, n, r, dtype, device, seed=12)
    H_eng = Operator(TTEngine(H.cores))
    S_eng = Operator(TTEngine(S.cores))

    torch.manual_seed(13)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    ls = LinearSystem(H_eng, scatter_op=S_eng)
    ls.state = State(TTEngine(x0.cores))

    solver = SourceIterationSolver()
    solver.solve(ls)

    x_result = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
    assert x_result.norm().item() < 1e-10
    assert solver.last_si_sweeps == 0


def test_source_iteration_solver_rejects_missing_scatter_op():
    """SourceIterationSolver on a LinearSystem with no scatter_op (assembled
    without source_iterate_scattering) must be rejected -- solving it would
    double-count scattering if get_scatter_op() were ever defined, or
    silently proceed with no scattering source at all here since it's
    undefined; either way it's the mismatch LocalSolver::presolve() guards
    against."""
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    H, _, b = _split_diag_dominant_system(d, n, r, dtype, device, seed=14)
    H_eng = Operator(TTEngine(H.cores))
    b_eng = State(TTEngine(b.cores))

    ls = LinearSystem(H_eng, source=Source(b_eng))  # no scatter_op
    with pytest.raises(RuntimeError):
        SourceIterationSolver().solve(ls)


def test_amen_solver_rejects_present_scatter_op():
    """The mismatch in the other direction: a plain AMEnSolver on a
    LinearSystem that DOES carry a separate scatter_op (assembled with
    source_iterate_scattering=True) must also be rejected -- solving it with
    AMEnSolver would silently drop scattering from the physics, since
    AMEnSolver never reads get_scatter_op()."""
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    H, S, b = _split_diag_dominant_system(d, n, r, dtype, device, seed=15)
    H_eng = Operator(TTEngine(H.cores))
    S_eng = Operator(TTEngine(S.cores))
    b_eng = State(TTEngine(b.cores))

    ls = LinearSystem(H_eng, source=Source(b_eng), scatter_op=S_eng)
    with pytest.raises(RuntimeError):
        AMEnSolver().solve(ls)
