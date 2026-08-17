import pytest
import torch
from ttnte.linalg import (
    AMEnPreconditioner,
    FoldedLocalOperator,
    LocalPreconditioner,
    gmres_solve,
    gmres_solve_cpu,
    gmres_solve_gpu,
)

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

torch.manual_seed(123)


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _dense_as_local_operator(A):
    """Embeds a dense [dim, dim] matrix A as a FoldedLocalOperator with all.

    bond ranks trivial (l=r=L=R=s=S=1) and the physical mode carrying `dim`,
    so `op.apply(y)` with `y` shaped [1, dim, 1] computes `A @ y`.
    """
    device, dtype = A.device, A.dtype
    dim = A.shape[0]
    phi_left = torch.ones(1, 1, 1, device=device, dtype=dtype)
    phi_right = torch.ones(1, 1, 1, device=device, dtype=dtype)
    a_core = A.reshape(1, dim, dim, 1)
    return FoldedLocalOperator.build(phi_left, a_core, phi_right)


def _random_square_system(dim, device, dtype, diag_boost=0.0):
    A = torch.randn(dim, dim, device=device, dtype=dtype)
    if diag_boost > 0.0:
        A = A + diag_boost * torch.eye(dim, device=device, dtype=dtype)
    return A, _dense_as_local_operator(A)


@pytest.mark.parametrize("device,dtype", test_params)
def test_gmres_matches_torch_linalg_solve_well_conditioned(device, dtype):
    _skip_if_unavailable(device)
    tol = 2e-3 if dtype == torch.float32 else 1e-8
    dim = 12

    A, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)

    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    solvers = [gmres_solve, gmres_solve_cpu if device == "cpu" else gmres_solve_gpu]
    for solver in solvers:
        x = solver(op, rhs, x0, 30, 3, 1e-10)
        err_vs_torch_solve = (x - x_ref).norm() / x_ref.norm()
        err_vs_true = (x - x_true).norm() / x_true.norm()
        assert err_vs_torch_solve.item() < tol, solver.__name__
        assert err_vs_true.item() < tol, solver.__name__


@pytest.mark.parametrize("device,dtype", test_params)
def test_gmres_matches_torch_linalg_solve_ill_conditioned(device, dtype):
    """Explicitly exercise an ill-conditioned dense system, comparing against
    torch.linalg.solve as ground truth."""
    _skip_if_unavailable(device)
    if dtype == torch.float32:
        pytest.skip("float32 cannot resolve this condition number")
    tol = 1e-5
    dim = 10

    U, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, dtype=dtype))
    V, _ = torch.linalg.qr(torch.randn(dim, dim, device=device, dtype=dtype))
    s = torch.logspace(0, -6, dim, device=device, dtype=dtype)
    A = U @ torch.diag(s) @ V.transpose(0, 1)
    op = _dense_as_local_operator(A)

    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    x = gmres_solve(op, rhs, x0, 60, 5, 1e-12)

    err = (x - x_ref).norm() / x_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", test_params)
def test_gmres_warm_start_converges(device, dtype):
    """A good initial guess should still converge to the correct answer."""
    _skip_if_unavailable(device)
    tol = 2e-3 if dtype == torch.float32 else 1e-8
    dim = 8

    A, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = x_true + 0.01 * torch.randn_like(x_true)
    x = gmres_solve(op, rhs, x0, 30, 3, 1e-10)
    err = (x - x_ref).norm() / x_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gmres_restart_actually_restarts(device):
    """`max_iterations` deliberately smaller than the problem dimension, so a single
    (non-restarted) Krylov cycle cannot span the full space and convergence requires the
    restart loop body (residual recompute, Arnoldi restart from the updated `x`) to
    actually run more than once -- every other GMRES test in this file picks
    `max_iterations >= dim`, which converges in the first cycle and never exercises that
    code path.

    float64-only (locally reseeded for determinism): float32 precision is too
    coarse to reliably distinguish "one cycle under-converges" from "several
    restarts converge" at a tolerance loose enough for fp32 in the first place.
    """
    _skip_if_unavailable(device)
    dtype = torch.float64
    torch.manual_seed(999)
    tol = 1e-8
    dim = 20
    max_iterations = 5
    restarts = 20

    # Mildly (not strongly) diagonally dominant: strong enough that repeated
    # restarts reliably converge, weak enough that a single 5-vector Krylov
    # cycle can't resolve the full 20-dimensional system to `tol`.
    A = torch.randn(dim, dim, device=device, dtype=dtype) + 1.2 * dim * torch.eye(
        dim, device=device, dtype=dtype
    )
    op = _dense_as_local_operator(A)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    solvers = [gmres_solve, gmres_solve_cpu if device == "cpu" else gmres_solve_gpu]
    for solver in solvers:
        x = solver(op, rhs, x0, max_iterations, restarts, 1e-10)
        err = (x - x_ref).norm() / x_ref.norm()
        assert err.item() < tol, solver.__name__

        # A single cycle alone (restarts=1) should NOT be enough to reach the
        # same tolerance -- otherwise this test wouldn't actually be
        # exercising more than one restart.
        x_one_cycle = solver(op, rhs, x0, max_iterations, 1, 1e-10)
        err_one_cycle = (x_one_cycle - x_ref).norm() / x_ref.norm()
        assert err_one_cycle.item() > 100 * tol, solver.__name__


@pytest.mark.parametrize("device,dtype", test_params)
def test_gmres_solve_with_local_preconditioner(device, dtype):
    """Exercises `gmres_solve`'s `prec` argument end-to-end -- previously
    completely untested at this level (only indirectly covered, for the
    NATIVE backend as a whole, by
    test_amen_solver.py::test_amen_solver_local_preconditioner_matches_torchtt).
    Uses the same trivial-rank (l=r=L=R=s=S=1) embedding as
    `_dense_as_local_operator`, so `LocalPreconditioner`'s block equals the
    dense operator itself -- the point is verifying the `prec` plumbing
    (Python binding -> gmres_solve_cpu/gpu's `local_apply` ->
    apply_forward/apply_inverse) is wired correctly end to end, not
    stressing the preconditioner numerically (see test_local_preconditioner.py
    for that)."""
    _skip_if_unavailable(device)
    tol = 2e-3 if dtype == torch.float32 else 1e-8
    dim = 10

    A, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    phi_left = torch.ones(1, 1, 1, device=device, dtype=dtype)
    phi_right = torch.ones(1, 1, 1, device=device, dtype=dtype)
    a_core = A.reshape(1, dim, dim, 1)
    prec = LocalPreconditioner.build(
        phi_left, a_core, phi_right, AMEnPreconditioner.LOCAL_C_PREC
    )

    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    y0 = prec.apply_forward(x0)
    solve_fn = gmres_solve_cpu if device == "cpu" else gmres_solve_gpu
    y = solve_fn(op, rhs, y0, 30, 3, 1e-10, prec)
    x = prec.apply_inverse(y)

    err = (x - x_ref).norm() / x_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gmres_mixed_precision_matches_torch_linalg_solve(device):
    """`gmres_mixed_precision=True` runs the inner Arnoldi/Gram-Schmidt/Givens
    build in float32 while keeping x/rhs/the true per-restart residual in
    float64 (see `AMEnNativeOptions::gmres_mixed_precision`) -- on both CPU
    and CUDA tensors. float64-only (mixing "up" from an already-float32
    input is a no-op by construction, not a meaningful case).

    Checks two things: the returned solution is close to the float64 ground
    truth (`torch.linalg.solve`), and -- the specific failure mode Jang,
    Jolivet & Mary (NLAA 2026) show for a structurally similar
    cheap-incremental-update pattern in GMRES-DR -- an INDEPENDENTLY,
    freshly computed true residual (via the original float64 operator, not
    reusing any of GMRES's own internal incremental bookkeeping) is actually
    small. If the incremental Givens residual estimate this codebase tracks
    (`beta[k+1]`) were silently drifting from the true residual under
    reduced precision, GMRES could declare false convergence while this
    independent check would catch it.
    """
    _skip_if_unavailable(device)
    dtype = torch.float64
    tol = 2e-3
    dim = 12

    A, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    rel_tol = 1e-10
    x = gmres_solve(
        op,
        rhs,
        x0,
        30,
        3,
        rel_tol,
        prefer_incremental=True,
        gmres_mixed_precision=True,
    )

    err_vs_torch_solve = (x - x_ref).norm() / x_ref.norm()
    err_vs_true = (x - x_true).norm() / x_true.norm()
    assert err_vs_torch_solve.item() < tol
    assert err_vs_true.item() < tol

    # Independent true-residual check (the Jang/Jolivet/Mary-motivated part):
    # recompute b - A @ x from scratch via the original float64 op, with no
    # reuse of GMRES's own internal state.
    true_residual = (rhs - op.apply(x)).norm() / rhs.norm()
    assert true_residual.item() < 10 * rel_tol


def test_gmres_mixed_precision_gpu_batched_strategy_matches_torch_linalg_solve():
    """Same check as `test_gmres_mixed_precision_matches_torch_linalg_solve`, but
    through `gmres_solve_gpu`'s fixed-budget Arnoldi strategy instead of the
    incremental-Givens one -- `gmres_mixed_precision` applies to both GMRES strategies,
    not just the incremental one."""
    _skip_if_unavailable("cuda")
    device, dtype = "cuda", torch.float64
    tol = 2e-3
    dim = 12

    A, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x_ref = torch.linalg.solve(A, rhs.reshape(-1)).reshape(x_true.shape)

    x0 = torch.zeros_like(x_true)
    rel_tol = 1e-10
    x = gmres_solve_gpu(op, rhs, x0, 30, 3, rel_tol, gmres_mixed_precision=True)

    err_vs_torch_solve = (x - x_ref).norm() / x_ref.norm()
    err_vs_true = (x - x_true).norm() / x_true.norm()
    assert err_vs_torch_solve.item() < tol
    assert err_vs_true.item() < tol

    true_residual = (rhs - op.apply(x)).norm() / rhs.norm()
    assert true_residual.item() < 10 * rel_tol


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_gmres_mixed_precision_default_off_is_unchanged(device):
    """`gmres_mixed_precision` defaults to False -- omitting it entirely must produce
    bit-for-bit the same result as passing it explicitly False, and the same code path
    as before this option existed.

    Checked against both
    GMRES strategies (gmres_solve_cpu on CPU, gmres_solve_gpu on CUDA, since
    that's what `gmres_solve` itself would dispatch to by default).
    """
    _skip_if_unavailable(device)
    dtype = torch.float64
    dim = 10

    _, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x0 = torch.zeros_like(x_true)

    solve_fn = gmres_solve_cpu if device == "cpu" else gmres_solve_gpu
    x_default = solve_fn(op, rhs, x0, 30, 3, 1e-10)
    x_explicit_off = solve_fn(op, rhs, x0, 30, 3, 1e-10, gmres_mixed_precision=False)

    assert torch.equal(x_default, x_explicit_off)


@pytest.mark.parametrize(
    "device,dtype", [("cuda", torch.float32), ("cuda", torch.float64)]
)
def test_gmres_prefer_incremental_forces_cpu_strategy_on_gpu_tensors(device, dtype):
    """`gmres_solve`'s `prefer_incremental` flag should route CUDA tensors through
    `gmres_solve_cpu`'s incremental-Givens strategy instead of `gmres_solve_gpu`'s
    fixed-budget strategy -- previously untested (every other test in this file either
    omits the flag or only exercises the per-device dispatch's default routing)."""
    _skip_if_unavailable(device)
    tol = 2e-3 if dtype == torch.float32 else 1e-8
    dim = 12

    _, op = _random_square_system(dim, device, dtype, diag_boost=5.0 * dim)
    x_true = torch.randn(1, dim, 1, device=device, dtype=dtype)
    rhs = op.apply(x_true)
    x0 = torch.zeros_like(x_true)

    x_dispatched = gmres_solve(op, rhs, x0, 30, 3, 1e-10, prefer_incremental=True)
    x_direct_cpu_strategy = gmres_solve_cpu(op, rhs, x0, 30, 3, 1e-10)

    err = (x_dispatched - x_direct_cpu_strategy).norm() / x_direct_cpu_strategy.norm()
    assert err.item() < tol
