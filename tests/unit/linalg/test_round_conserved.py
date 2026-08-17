import torch
import pytest
import torchtt as tntt

from ttnte.linalg import State, Operator, TTEngine, mv, round_conserved

test_dtypes = [torch.float32, torch.float64]


def _rank1_projector(mode_sizes, dtype, seed=0):
    """P = |v><v| (x) I on the remaining modes -- a genuine orthogonal projector onto a
    single rank-1 direction of the first mode, independent of any physics.

    Sufficient to exercise round_conserved()'s contract.
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.rand(mode_sizes[0], dtype=dtype, generator=g)
    v = v / v.norm()
    cores = [torch.outer(v, v).reshape(1, mode_sizes[0], mode_sizes[0], 1)]
    for m in mode_sizes[1:]:
        cores.append(torch.eye(m, dtype=dtype).reshape(1, m, m, 1))
    return Operator(TTEngine(cores))


@pytest.mark.parametrize("dtype", test_dtypes)
def test_round_conserved_preserves_moment_under_aggressive_rounding(dtype):
    device = "cpu"
    n = [5, 6, 4]
    P = _rank1_projector(n, dtype)

    torch.manual_seed(0)
    x = State(TTEngine(tntt.random(n, [1, 3, 3, 1], dtype=dtype).to(device).cores))
    moment_before = mv(P, x)

    tight_eps = 1e-12
    loose_eps = 0.5  # deliberately aggressive -- should visibly perturb x
    x_rounded = round_conserved(x, P, loose_eps, 1000, tight_eps, 1000)

    # The aggressive pass actually did something.
    assert (x_rounded - x).as_tt().norm() > 1e-6

    # But the projected moment survived it far better than the loose
    # tolerance alone would explain -- rounding x_remainder can still leak a
    # little energy back into P's range (SVD truncation targets low Frobenius
    # error, not orthogonality to a fixed subspace), so this isn't bounded by
    # moment_eps itself, just far below loose_eps. float32's coarser working
    # precision leaks noticeably more of that energy back than float64 does.
    tol = 1e-3 if dtype == torch.float32 else 1e-6
    moment_after = mv(P, x_rounded)
    rel = (moment_after - moment_before).as_tt().norm() / moment_before.as_tt().norm()
    assert rel < tol


def test_round_conserved_undefined_projector_falls_back_to_plain_round():
    dtype = torch.float64
    n = [4, 4]
    torch.manual_seed(0)
    x = State(TTEngine(tntt.random(n, [1, 2, 1], dtype=dtype).cores))

    x_rounded = round_conserved(x, Operator(), 1e-6, 1000, 1e-12, 1000)
    x_plain = x.round(1e-6, 1000)
    # Relative, not absolute -- the two round_() calls go through the same
    # eps/max_rank but aren't guaranteed bit-identical (SVD in the backing
    # BLAS/LAPACK isn't guaranteed deterministic across separate calls).
    rel = (x_rounded - x_plain).as_tt().norm() / x_plain.as_tt().norm()
    assert rel < 1e-6
