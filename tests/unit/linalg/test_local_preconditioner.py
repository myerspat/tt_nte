import pytest
import torch
from ttnte.linalg import AMEnPreconditioner, LocalPreconditioner

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

torch.manual_seed(13)


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _reference_block(phi_left, a_core, phi_right, mode):
    """Direct port of torchTT's AMENsolveMV::setter (matvecs.h) -- the same
    formula this module's build() implements, kept independent here so the
    test isn't just checking the implementation against itself."""
    Jl = torch.tensordot(torch.diagonal(phi_left, 0, 0, 2), a_core, dims=([0], [0]))
    if mode == AMEnPreconditioner.LOCAL_C_PREC:
        Jr = torch.diagonal(phi_right, 0, 0, 2)
        Jt = torch.tensordot(Jl, Jr, dims=([3], [0]))
        return Jt.permute(0, 3, 1, 2).contiguous()  # [d, d2, m, n]
    Jt = torch.tensordot(Jl, phi_right, dims=([3], [1]))
    return Jt.permute(0, 1, 3, 2, 4).contiguous()  # [d, m, L, n, R]


def _reference_forward(Jt, x, mode, d, n, R):
    if mode == AMEnPreconditioner.LOCAL_C_PREC:
        x_p = x.permute(0, 2, 1).unsqueeze(-1)
        return torch.matmul(Jt, x_p).squeeze(-1).permute(0, 2, 1)
    Jt2 = Jt.reshape(d, n * R, n * R)
    return torch.matmul(Jt2, x.reshape(d, n * R, 1)).reshape(d, n, R)


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize("mode", [AMEnPreconditioner.LOCAL_C_PREC, AMEnPreconditioner.LOCAL_R_PREC])
@pytest.mark.parametrize("d,d2,s,n,S", [(3, 4, 2, 5, 3), (1, 1, 1, 3, 1)])
def test_local_preconditioner_matches_reference(device, dtype, mode, d, d2, s, n, S):
    """Cross-checks build()/apply_forward() against an independent
    reimplementation of the exact baseline formula (torchTT's
    AMENsolveMV::setter/apply_prec, matvecs.h), not just internal
    self-consistency."""
    _skip_if_unavailable(device)
    tol = 1e-3 if dtype == torch.float32 else 1e-9

    phi_left = torch.randn(d, s, d, device=device, dtype=dtype)
    a_core = torch.randn(s, n, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(d2, S, d2, device=device, dtype=dtype)

    prec = LocalPreconditioner.build(phi_left, a_core, phi_right, mode)
    x = torch.randn(d, n, d2, device=device, dtype=dtype)
    out = prec.apply_forward(x)

    Jt_ref = _reference_block(phi_left, a_core, phi_right, mode)
    ref = _reference_forward(Jt_ref, x, mode, d, n, d2)

    assert torch.allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize("mode", [AMEnPreconditioner.LOCAL_C_PREC, AMEnPreconditioner.LOCAL_R_PREC])
def test_local_preconditioner_round_trip(device, dtype, mode):
    """apply_inverse(apply_forward(x)) ~= x -- the property the sweep relies
    on to recover the true solution from the preconditioned GMRES output."""
    _skip_if_unavailable(device)
    tol = 1e-3 if dtype == torch.float32 else 1e-7
    d, d2, s, n, S = 3, 4, 2, 5, 3

    phi_left = torch.randn(d, s, d, device=device, dtype=dtype)
    a_core = torch.randn(s, n, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(d2, S, d2, device=device, dtype=dtype)

    prec = LocalPreconditioner.build(phi_left, a_core, phi_right, mode)
    x = torch.randn(d, n, d2, device=device, dtype=dtype)

    y = prec.apply_forward(x)
    x_recovered = prec.apply_inverse(y)
    assert torch.allclose(x, x_recovered, atol=tol, rtol=tol)

    # And the other direction, since gmres_local warm-starts with
    # y0 = apply_forward(x0) and needs apply_inverse(apply_forward(.)) == id
    # in both directions for that round trip to be meaningful.
    y2 = torch.randn(d, n, d2, device=device, dtype=dtype)
    x2 = prec.apply_inverse(y2)
    y2_recovered = prec.apply_forward(x2)
    assert torch.allclose(y2, y2_recovered, atol=tol, rtol=tol)


def test_local_preconditioner_rejects_bad_mode():
    """NONE and RANK1 are valid AMEnPreconditioner values overall, but
    invalid for LocalPreconditioner::build specifically (NONE means "don't
    precondition at all"; RANK1 is the unrelated global preconditioner)."""
    d, s, n, S = 2, 2, 3, 2
    phi_left = torch.randn(d, s, d, dtype=torch.float64)
    a_core = torch.randn(s, n, n, S, dtype=torch.float64)
    phi_right = torch.randn(d, S, d, dtype=torch.float64)
    with pytest.raises(RuntimeError):
        LocalPreconditioner.build(phi_left, a_core, phi_right, AMEnPreconditioner.NONE)
    with pytest.raises(RuntimeError):
        LocalPreconditioner.build(phi_left, a_core, phi_right, AMEnPreconditioner.RANK1)
