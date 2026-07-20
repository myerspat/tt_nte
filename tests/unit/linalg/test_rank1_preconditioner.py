import pytest
import torch
import torchtt as tntt
from ttnte.linalg import TTEngine, Rank1Preconditioner

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

torch.manual_seed(7)


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _random_spd_tt(d, n, r, device, dtype):
    A = tntt.random([(n, n)] * d, [1] + [r] * (d - 1) + [1], dtype=dtype).to(device)
    # Make it diagonally dominant (well-conditioned) so a plain solve is a
    # meaningful ground truth to compare against.
    A = A + (5.0 * n) * tntt.eye([n] * d, dtype=dtype).to(device)
    return A


@pytest.mark.parametrize("device,dtype", test_params)
def test_preconditioner_preserves_b_rank(device, dtype):
    """Applying P_left to b must not increase its TT rank (the key property
    motivating the rank-1 preconditioner choice)."""
    _skip_if_unavailable(device)
    d, n, r = 3, 6, 3

    A = _random_spd_tt(d, n, r, device, dtype)
    b = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)

    prec = Rank1Preconditioner.build(TTEngine(A.cores))
    b_cores = list(b.cores)  # plain 3-D vector cores: [r_l, N, r_r]
    pb_cores = prec.apply_left(b_cores)

    assert len(pb_cores) == len(b_cores)
    for orig, transformed in zip(b_cores, pb_cores):
        # Bond ranks (dims 0 and 2) must be unchanged.
        assert transformed.shape[0] == orig.shape[0]
        assert transformed.shape[2] == orig.shape[2]


@pytest.mark.parametrize("device,dtype", test_params)
def test_apply_right_inverse_round_trips_with_apply_right(device, dtype):
    """apply_right_inverse maps x-space -> canonical space; apply_right maps
    it back. Used to warm-start a preconditioned solve from a user-supplied
    x0 living in the operator's input space."""
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    d, n, r = 3, 6, 3

    A = _random_spd_tt(d, n, r, device, dtype)
    x0 = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)

    prec = Rank1Preconditioner.build(TTEngine(A.cores))
    x0_cores = list(x0.cores)
    y0_cores = prec.apply_right_inverse(x0_cores)
    x0_roundtrip_cores = prec.apply_right(y0_cores)

    x0_dense = tntt.TT(x0_cores).full().reshape(-1)
    roundtrip_dense = tntt.TT(x0_roundtrip_cores).full().reshape(-1)
    err = (roundtrip_dense - x0_dense).norm() / x0_dense.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", test_params)
def test_preconditioned_solve_matches_unpreconditioned(device, dtype):
    """Solving the preconditioned dense system and mapping back with
    apply_right should reproduce the same solution as solving the original
    dense system directly."""
    _skip_if_unavailable(device)
    tol = 1e-3 if dtype == torch.float32 else 1e-8
    d, n, r = 3, 5, 2

    A = _random_spd_tt(d, n, r, device, dtype)
    x_true = tntt.random([n] * d, [1, 2, 2, 1], dtype=dtype).to(device)
    b = (A @ x_true).round(1e-13)

    A_dense = A.full().reshape(n**d, n**d)
    b_dense = b.full().reshape(-1)
    x_ref = torch.linalg.solve(A_dense, b_dense)

    prec = Rank1Preconditioner.build(TTEngine(A.cores))
    Ap_cores = prec.sandwich_operator(list(A.cores))
    Pb_cores = prec.apply_left(list(b.cores))

    m_total = 1
    n_total = 1
    for c in Ap_cores:
        m_total *= c.shape[1]
        n_total *= c.shape[2]

    Ap_dense = tntt.TT(Ap_cores).full().reshape(m_total, n_total)
    Pb_dense = tntt.TT(Pb_cores).full().reshape(-1)

    y_dense = torch.linalg.solve(Ap_dense, Pb_dense)

    # Exactly TT-decompose y (canonical/preconditioned space) and map it back
    # to x-space with apply_right; x should match the reference dense solve
    # of the *original*, unpreconditioned system.
    y_tt_cores_3d = [
      c.squeeze(2)
      for c in TTEngine.from_dense(
        y_dense.reshape([c.shape[1] for c in Ap_cores]), eps=1e-12
      ).cores
    ]
    x_cores = prec.apply_right(y_tt_cores_3d)
    x_dense = tntt.TT(x_cores).full().reshape(-1)

    err = (x_dense - x_ref).norm() / x_ref.norm()
    assert err.item() < tol
