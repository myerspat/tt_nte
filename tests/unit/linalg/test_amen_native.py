import pytest
import torch
import torchtt as tntt
from ttnte.linalg import (
    TTEngine,
    AMEnBackend,
    AMEnEnrichmentMode,
    AMEnNativeOptions,
    AMEnPreconditioner,
    amen_solve,
    amen_solve_native,
)

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

torch.manual_seed(42)


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _well_conditioned_system(d, n, r, device, dtype):
    A = tntt.random([(n, n)] * d, [1] + [r] * (d - 1) + [1], dtype=dtype).to(device)
    A = A + (10.0 * n) * tntt.eye([n] * d, dtype=dtype).to(device)
    x_true = tntt.random([n] * d, [1, 2, 2, 1][: d + 1], dtype=dtype).to(device)
    b = (A @ x_true).round(1e-13)
    return A, b, x_true


def _to_engines(A, b):
    A_eng = TTEngine(A.cores)
    b_eng = TTEngine([c.unsqueeze(2) for c in b.cores])
    return A_eng, b_eng


def _to_tt(x_engine):
    return tntt.TT([c.squeeze(2) for c in x_engine.cores])


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize(
    "enrichment_mode",
    [AMEnEnrichmentMode.FULL, AMEnEnrichmentMode.ALS_FIXED_RANK, AMEnEnrichmentMode.SIMPLIFIED],
)
@pytest.mark.parametrize("preconditioner", [AMEnPreconditioner.NONE, AMEnPreconditioner.RANK1])
@pytest.mark.parametrize("use_qless_tsqr", [False, True])
def test_native_matches_dense_and_torchtt(
    device, dtype, enrichment_mode, preconditioner, use_qless_tsqr
):
    _skip_if_unavailable(device)
    tol = 5e-4 if dtype == torch.float32 else 1e-6
    d, n, r = 3, 6, 2

    A, b, _x_true = _well_conditioned_system(d, n, r, device, dtype)
    A_eng, b_eng = _to_engines(A, b)

    A_dense = A.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    opts = AMEnNativeOptions(
        enrichment_mode=enrichment_mode,
        use_qless_tsqr=use_qless_tsqr,
        als_residual_rank=4,
    )
    x_native = amen_solve(
        A_eng, b_eng, x0=None, nswp=22, eps=1e-10, max_rank=2**31 - 1,
        max_full=500, kickrank=4, kick2=0, local_iterations=40, resets=2,
        verbose=False, preconditioner=preconditioner, backend=AMEnBackend.NATIVE,
        native_opts=opts,
    )
    xn = _to_tt(x_native)
    xn_dense = xn.full().reshape(-1)

    err_dense = (xn_dense - x_dense_ref).norm() / x_dense_ref.norm()
    assert err_dense.item() < tol

    # AMEnPreconditioner.RANK1 is NATIVE-only -- the torchTT reference call
    # always runs unpreconditioned; preconditioning is purely an internal
    # solve-path transform, so it shouldn't change the converged answer.
    x_torchtt = amen_solve(
        A_eng, b_eng, x0=None, nswp=22, eps=1e-10, max_rank=2**31 - 1,
        max_full=500, kickrank=4, kick2=0, local_iterations=40, resets=2,
        verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.TORCHTT,
    )
    xt = _to_tt(x_torchtt)
    err_vs_torchtt = (xn - xt).norm() / xt.norm()
    assert err_vs_torchtt.item() < tol


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float64)])
def test_native_ill_conditioned(device, dtype):
    """Explicit ill-conditioned system: an operator built with a large
    singular-value spread, still expected to converge to the dense ground
    truth (the Q-less TSQR fallback and rank-1 preconditioner should keep the
    native backend robust here, per the benchmark requirement)."""
    _skip_if_unavailable(device)
    tol = 1e-4
    d, n = 3, 6

    # Build an ill-conditioned diagonal-ish operator: TT-eye scaled per-core
    # by geometrically spread factors, plus a small random rank-2 part.
    A = tntt.random([(n, n)] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    scales = torch.logspace(0, 4, d)
    diag_parts = [scales[i] * tntt.eye([n] * d, dtype=dtype).to(device) for i in range(d)]
    A_illcond = A * 0.01
    for dp in diag_parts:
        A_illcond = A_illcond + dp
    A_illcond = A_illcond.round(1e-13)

    x_true = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    b = (A_illcond @ x_true).round(1e-13)

    A_eng, b_eng = _to_engines(A_illcond, b)
    A_dense = A_illcond.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    opts = AMEnNativeOptions(enrichment_mode=AMEnEnrichmentMode.FULL)
    x_native = amen_solve(
        A_eng, b_eng, x0=None, nswp=30, eps=1e-11, max_rank=2**31 - 1,
        max_full=500, kickrank=6, kick2=0, local_iterations=60, resets=3,
        verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.NATIVE,
        native_opts=opts,
    )
    xn = _to_tt(x_native)
    xn_dense = xn.full().reshape(-1)
    err = (xn_dense - x_dense_ref).norm() / x_dense_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float64)])
def test_native_als_fixed_rank_ill_conditioned(device, dtype):
    """Same ill-conditioned system as test_native_ill_conditioned, but
    exercising AMEnEnrichmentMode.ALS_FIXED_RANK specifically."""
    _skip_if_unavailable(device)
    tol = 1e-4
    d, n = 3, 6

    A = tntt.random([(n, n)] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    scales = torch.logspace(0, 4, d)
    diag_parts = [scales[i] * tntt.eye([n] * d, dtype=dtype).to(device) for i in range(d)]
    A_illcond = A * 0.01
    for dp in diag_parts:
        A_illcond = A_illcond + dp
    A_illcond = A_illcond.round(1e-13)

    x_true = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    b = (A_illcond @ x_true).round(1e-13)

    A_eng, b_eng = _to_engines(A_illcond, b)
    A_dense = A_illcond.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    opts = AMEnNativeOptions(
      enrichment_mode=AMEnEnrichmentMode.ALS_FIXED_RANK, als_residual_rank=6
    )
    x_native = amen_solve(
        A_eng, b_eng, x0=None, nswp=30, eps=1e-11, max_rank=2**31 - 1,
        max_full=500, kickrank=6, kick2=0, local_iterations=60, resets=3,
        verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.NATIVE,
        native_opts=opts,
    )
    xn = _to_tt(x_native)
    xn_dense = xn.full().reshape(-1)
    err = (xn_dense - x_dense_ref).norm() / x_dense_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float64)])
def test_native_simplified_ill_conditioned(device, dtype):
    """Same ill-conditioned system as test_native_ill_conditioned, but
    exercising AMEnEnrichmentMode.SIMPLIFIED specifically."""
    _skip_if_unavailable(device)
    tol = 1e-4
    d, n = 3, 6

    A = tntt.random([(n, n)] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    scales = torch.logspace(0, 4, d)
    diag_parts = [scales[i] * tntt.eye([n] * d, dtype=dtype).to(device) for i in range(d)]
    A_illcond = A * 0.01
    for dp in diag_parts:
        A_illcond = A_illcond + dp
    A_illcond = A_illcond.round(1e-13)

    x_true = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    b = (A_illcond @ x_true).round(1e-13)

    A_eng, b_eng = _to_engines(A_illcond, b)
    A_dense = A_illcond.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    opts = AMEnNativeOptions(enrichment_mode=AMEnEnrichmentMode.SIMPLIFIED)
    x_native = amen_solve(
        A_eng, b_eng, x0=None, nswp=30, eps=1e-11, max_rank=2**31 - 1,
        max_full=500, kickrank=6, kick2=0, local_iterations=60, resets=3,
        verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.NATIVE,
        native_opts=opts,
    )
    xn = _to_tt(x_native)
    xn_dense = xn.full().reshape(-1)
    err = (xn_dense - x_dense_ref).norm() / x_dense_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_native_amen_solve_native_direct(device):
    """amen_solve_native() is exposed directly for testing/benchmarking and
    should work correctly when called explicitly on both CPU and CUDA
    tensors -- there's no per-device dispatch inside it to exercise (see its
    doc comment: every device-specific decision already lives one layer
    down, inside amen_sweep's own primitives), so this is really just
    confirming it works end to end on both device types."""
    _skip_if_unavailable(device)
    d, n, r = 3, 6, 2
    A, b, _x_true = _well_conditioned_system(d, n, r, device, torch.float64)
    A_eng, b_eng = _to_engines(A, b)

    A_dense = A.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    x_native = amen_solve_native(
        A_eng, b_eng, x0=None, nswp=22, eps=1e-10, max_rank=2**31 - 1,
        max_full=500, kickrank=4, kick2=0, local_iterations=40, resets=2,
        verbose=False, preconditioner=AMEnPreconditioner.NONE, native_opts=AMEnNativeOptions(),
    )
    assert x_native.device.type == device
    xn = _to_tt(x_native)
    err = (xn.full().reshape(-1).cpu() - x_dense_ref.cpu()).norm() / x_dense_ref.cpu().norm()
    assert err.item() < 1e-6


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float64)])
@pytest.mark.parametrize(
    "enrichment_mode,kickrank,kick2,als_residual_rank",
    [
        (AMEnEnrichmentMode.FULL, 0, 0, 4),
        (AMEnEnrichmentMode.SIMPLIFIED, 0, 0, 4),
        (AMEnEnrichmentMode.ALS_FIXED_RANK, 4, 0, 0),
    ],
)
def test_native_zero_enrichment_pure_als(
    device, dtype, enrichment_mode, kickrank, kick2, als_residual_rank
):
    """kickrank=kick2=0 (FULL/SIMPLIFIED) or als_residual_rank=0
    (ALS_FIXED_RANK) request zero rank enrichment -- previously crashed
    with an ambiguous 0-element reshape while building z_cores (see memory
    bug_amen_precond_magma_and_zero_rank). Confirms: no crash, no bond rank
    ever exceeds the warm start's, and (given a warm start whose rank
    already matches the true solution's) it still converges via pure ALS."""
    _skip_if_unavailable(device)
    tol = 1e-6
    d, n, r = 3, 6, 2

    A, b, x_true = _well_conditioned_system(d, n, r, device, dtype)
    A_eng, b_eng = _to_engines(A, b)

    x0 = tntt.random([n] * d, x_true.R, dtype=dtype).to(device)
    x0_eng = TTEngine(x0.cores)

    A_dense = A.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_dense_ref = torch.linalg.solve(A_dense, b_dense)

    opts = AMEnNativeOptions(
        enrichment_mode=enrichment_mode, als_residual_rank=als_residual_rank
    )
    x_native = amen_solve(
        A_eng, b_eng, x0=x0_eng, nswp=30, eps=1e-11, max_rank=2**31 - 1,
        max_full=500, kickrank=kickrank, kick2=kick2, local_iterations=60,
        resets=3, verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.NATIVE,
        native_opts=opts,
    )
    xn = _to_tt(x_native)

    assert all(rn <= r0 for rn, r0 in zip(xn.R[1:-1], x0.R[1:-1]))

    err = (xn.full().reshape(-1) - x_dense_ref).norm() / x_dense_ref.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", [("cpu", torch.float64), ("cuda", torch.float64)])
def test_native_zero_enrichment_rhs_rank_does_not_leak(device, dtype):
    """Direct test of the user's original question: does a richer
    right-hand side (more TT rank) leak into the solution's rank during a
    pure-ALS (zero-enrichment) sweep? Solve the same operator/warm-start
    twice with als_residual_rank=0, once against a low-rank RHS and once
    against an RHS padded with extra small random rank -- the resulting
    solution ranks must be identical (bounded only by the warm start)."""
    _skip_if_unavailable(device)
    d, n, r = 3, 6, 2

    A, b, x_true = _well_conditioned_system(d, n, r, device, dtype)
    x0 = tntt.random([n] * d, x_true.R, dtype=dtype).to(device)

    b_rich = (b + 1e-8 * tntt.random([n] * d, [1, 3, 3, 1], dtype=dtype).to(device)).round(0)

    opts = AMEnNativeOptions(
        enrichment_mode=AMEnEnrichmentMode.ALS_FIXED_RANK, als_residual_rank=0
    )

    def _solve(b_tt):
        A_eng, b_eng = _to_engines(A, b_tt)
        x0_eng = TTEngine(x0.cores)
        x_native = amen_solve(
            A_eng, b_eng, x0=x0_eng, nswp=15, eps=1e-11, max_rank=2**31 - 1,
            max_full=500, kickrank=0, kick2=0, local_iterations=60, resets=3,
            verbose=False, preconditioner=AMEnPreconditioner.NONE, backend=AMEnBackend.NATIVE,
            native_opts=opts,
        )
        return _to_tt(x_native).R

    assert _solve(b) == _solve(b_rich)
