import pytest
import torch
import torchtt as tntt

from ttnte.linalg import (
    State,
    Operator,
    LinearSystem,
    TTEngine,
    Source,
    AMEnBackend,
    AMEnNativeOptions,
    AMEnPreconditioner,
)
from ttnte.solvers import AMEnSolver

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

backends = [AMEnBackend.TORCHTT, AMEnBackend.NATIVE]


@pytest.mark.parametrize("device, dtype", test_params)
@pytest.mark.parametrize("backend", backends)
def test_amen_solver(device, dtype, backend):
    torch.manual_seed(42)

    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create random matrix and solution
    A = tntt.random(
        [(4, 4), (5, 5), (6, 6), (3, 3)], [1, 2, 3, 2, 1], dtype=dtype, device=device
    )
    b = A @ tntt.random([4, 5, 6, 3], [1, 3, 2, 2, 1], dtype=dtype, device=device)
    x0 = tntt.ones([4, 5, 6, 3], dtype=dtype, device=device)

    # Run torchTT AMEn solve
    xe = tntt.solvers.amen_solve(A, b, x0=x0, use_cpp=True)

    A = Operator(TTEngine(A.cores))
    x0 = State(TTEngine(x0.cores))
    b = State(TTEngine(b.cores))
    ls = LinearSystem(A, source=Source(b))
    ls.state = x0

    # Run AMEnSolver
    solver = AMEnSolver(backend=backend)
    solver.solve(ls)

    # Get the solution vector
    xa = tntt.TT([core.squeeze(2) for core in ls.state.as_tt().cores])
    assert (
        xa - xe
    ).norm() / xe.norm() < 6e-4  # They won't be the exact same because of random enrichment


def _diag_dominant_system(d, n, rank, dtype, device, seed):
    """Well-conditioned enough that both backends actually converge under a
    modest sweep budget -- a purely random TT operator doesn't (see
    benchmarks/amen_scaling_benchmark.py's `random_system` for the same
    construction and why it's used instead of plain `tntt.random`)."""
    torch.manual_seed(seed)
    op_ranks = [1] + [rank] * (d - 1) + [1]
    pert = tntt.random([(n, n)] * d, op_ranks, dtype=dtype).to(device)
    scales = torch.logspace(0, 0.5, d)
    A = pert * 0.01
    for i in range(d):
        A = A + scales[i] * tntt.eye([n] * d, dtype=dtype).to(device)
    A = A.round(1e-13)
    x_true = tntt.random([n] * d, op_ranks, dtype=dtype).to(device)
    b = (A @ x_true).round(1e-13)
    return A, b


@pytest.mark.parametrize("device, dtype", test_params)
@pytest.mark.parametrize(
    "preconditioner", [AMEnPreconditioner.LOCAL_C_PREC, AMEnPreconditioner.LOCAL_R_PREC]
)
def test_amen_solver_local_preconditioner_matches_torchtt(device, dtype, preconditioner):
    """NATIVE's local (per-core) preconditioner (LOCAL_C_PREC, LOCAL_R_PREC)
    should match TORCHTT's own local preconditioner on the same well-conditioned
    problem -- a direct, already-available correctness oracle instead of a
    from-scratch derivation."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    A, b = _diag_dominant_system(3, 8, 2, dtype, device, seed=0)

    results = {}
    for backend in backends:
        A_eng = Operator(TTEngine(A.cores))
        x0_eng = State(TTEngine(tntt.ones(A.N, dtype=dtype, device=device).cores))
        b_eng = State(TTEngine(b.cores))
        ls = LinearSystem(A_eng, source=Source(b_eng))
        ls.state = x0_eng

        solver = AMEnSolver(backend=backend, prec=preconditioner, nswp=15)
        solver.solve(ls)
        x_result = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
        results[backend] = ((A @ x_result - b).norm() / b.norm()).item()

    tol = 1e-4 if dtype == torch.float32 else 1e-8
    for backend, res in results.items():
        assert res < tol, f"{backend} with preconditioner={preconditioner} residual {res:.3e}"


def test_amen_solver_rank_freeze_rejects_torchtt_backend():
    """rank_freeze_eps relies on amen_sweep.cpp's `enrichment_disabled`
    zero-rank code path, which only exists for AMEnBackend.NATIVE -- the
    vendored torchTT backend crashes (ambiguous 0-element reshape) if
    enrichment is ever forced to zero rank. Must be rejected up front."""
    with pytest.raises(RuntimeError):
        AMEnSolver(
            backend=AMEnBackend.TORCHTT,
            native_opts=AMEnNativeOptions(rank_freeze_eps=1e-3),
        )


def test_amen_solver_rank_freeze():
    """AMEnNativeOptions.rank_freeze_eps: once the solver's own
    adaptively-tightening eps (see AMEnSolver.update_convergence_criteria)
    drops to or below this value, it should permanently switch to pure ALS
    (zero enrichment) for every later solve() call on this instance, and
    per-bond rank should never grow again from that point on -- see memory
    project_c5g7_amen_options_sweep's "grow-then-freeze" question."""
    torch.manual_seed(0)
    device, dtype = "cpu", torch.float64
    d, n, r = 3, 8, 2

    A, b = _diag_dominant_system(d, n, r, dtype, device, seed=1)
    x0 = tntt.random([n] * d, [1, r, r, 1], dtype=dtype).to(device)

    A_eng = Operator(TTEngine(A.cores))
    b_eng = State(TTEngine(b.cores))
    x0_eng = State(TTEngine(x0.cores))
    ls = LinearSystem(A_eng, source=Source(b_eng))
    ls.state = x0_eng

    freeze_eps = 1e-3
    solver = AMEnSolver(
        eps=1e-8,
        eps_forcing=1.0,  # eps_ = max(eps_floor_, min_error_ so far)
        nswp=5,
        native_opts=AMEnNativeOptions(rank_freeze_eps=freeze_eps),
    )
    assert not solver.is_rank_frozen()

    errors = [1.0, 0.5, 0.1, 1e-2, 1e-4, 1e-5, 1e-6, 1e-7]
    frozen_before_solve = []
    ranks_over_time = []
    residuals_over_time = []
    for err in errors:
        frozen_before_solve.append(solver.is_rank_frozen())
        solver.solve(ls)
        ranks_over_time.append(ls.state.as_tt().ranks)
        x_now = tntt.TT([c.squeeze(2) for c in ls.state.as_tt().cores])
        residuals_over_time.append(((A @ x_now - b).norm() / b.norm()).item())
        solver.update_convergence_criteria(err)

    assert not any(frozen_before_solve[:4])
    assert any(frozen_before_solve)
    first_frozen = frozen_before_solve.index(True)

    # Once frozen, every bond's rank must be non-increasing sweep over
    # sweep -- the SVD truncation loop inside amen_sweep can only shrink or
    # hold rank when enrichment is disabled, never grow it, regardless of
    # how the RHS evolves (see the "Correction" note in the rank-freeze
    # plan).
    for prev, curr in zip(
        ranks_over_time[first_frozen:], ranks_over_time[first_frozen + 1 :]
    ):
        assert all(rc <= rp for rc, rp in zip(curr, prev))

    # Pure ALS should still be doing useful work: residual keeps improving
    # after freezing, not stalled.
    assert residuals_over_time[-1] < residuals_over_time[first_frozen]
    assert residuals_over_time[-1] < 1e-4
