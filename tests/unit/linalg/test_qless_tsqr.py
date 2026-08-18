import pytest
import torch
from ttnte.linalg import tsqr_r, qless_orthogonalize

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


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize("rows,cols,block_size", [(200, 5, 32), (37, 4, 16), (64, 8, 64)])
def test_tsqr_r_matches_linalg_qr(device, dtype, rows, cols, block_size):
    """R should match at::linalg_qr's R up to sign, and reconstruct M."""
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9

    M = torch.randn(rows, cols, device=device, dtype=dtype)

    R = tsqr_r(M, block_size)
    assert R.shape == (cols, cols)

    _, R_ref = torch.linalg.qr(M, mode="reduced")

    # Compare up to a per-column sign flip (QR is only unique up to signs).
    sign = torch.sign(torch.diagonal(R)) * torch.sign(torch.diagonal(R_ref))
    R_aligned = R * sign.unsqueeze(1)
    assert torch.allclose(R_aligned, R_ref, atol=tol, rtol=tol)

    # R^T R should reconstruct the Gram matrix of M regardless of sign.
    gram = M.transpose(0, 1) @ M
    assert torch.allclose(R.transpose(0, 1) @ R, gram, atol=tol * gram.norm(), rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize("rows,cols,block_size", [(200, 5, 32), (37, 4, 16)])
def test_qless_orthogonalize_reconstructs_and_is_orthonormal(
    device, dtype, rows, cols, block_size
):
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9

    M = torch.randn(rows, cols, device=device, dtype=dtype)

    Q, R = qless_orthogonalize(M, block_size)
    assert Q.shape == M.shape
    assert R.shape == (cols, cols)

    # Reconstruction: Q @ R ~= M
    recon_err = (Q @ R - M).norm() / M.norm()
    assert recon_err.item() < tol

    # Orthonormality: Q^T Q ~= I
    gram = Q.transpose(0, 1) @ Q
    eye = torch.eye(cols, device=device, dtype=dtype)
    assert torch.allclose(gram, eye, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_tsqr_r_ill_conditioned_falls_back_gracefully(device, dtype):
    """A near rank-deficient input should not crash and should still
    reconstruct M reasonably well (via the at::linalg_qr fallback)."""
    _skip_if_unavailable(device)
    tol = 1e-2 if dtype == torch.float32 else 1e-6

    rows, cols = 128, 6
    U, _ = torch.linalg.qr(torch.randn(rows, cols, device=device, dtype=dtype))
    # Impose a huge singular value spread -> near-singular Gram matrix.
    s = torch.logspace(0, -10, cols, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(torch.randn(cols, cols, device=device, dtype=dtype))
    M = U @ torch.diag(s) @ V.transpose(0, 1)

    R = tsqr_r(M, 32)
    gram = M.transpose(0, 1) @ M
    err = (R.transpose(0, 1) @ R - gram).norm() / gram.norm()
    assert err.item() < tol


@pytest.mark.parametrize("device,dtype", test_params)
def test_qless_orthogonalize_ill_conditioned_takes_qr_fallback_path(device, dtype):
    """Confirms *which* fallback path `qless_orthogonalize` actually takes on
    an ill-conditioned input -- `test_tsqr_r_ill_conditioned_falls_back_gracefully`
    only checks reconstruction error, which would pass whether or not the
    diagonal-ratio QR-fallback branch (vs. the normal "recover Q via R^-1"
    path) actually triggered. A singular-value spread of 1e-10 (as used
    there) turns out to still leave `tsqr_r`'s aggregated diagonal ratio just
    above the 1e-10 fallback threshold (empirically ~1e-8), so this uses a
    much more extreme spread (1e-16) to reliably push the ratio below it. When
    the fallback triggers, the implementation returns
    `torch.linalg_qr(M, "reduced")` directly, so its output should be
    numerically identical to calling that ourselves; the normal TSQR path
    uses a different algorithm (Cholesky-QR tree reduction + triangular
    solve) and should NOT match it as closely."""
    _skip_if_unavailable(device)
    if dtype == torch.float32:
        pytest.skip(
            "float32 can't represent a 1e-16 singular-value spread meaningfully"
        )
    tol = 1e-9

    rows, cols = 128, 6
    U, _ = torch.linalg.qr(torch.randn(rows, cols, device=device, dtype=dtype))
    s = torch.logspace(0, -16, cols, device=device, dtype=dtype)
    V, _ = torch.linalg.qr(torch.randn(cols, cols, device=device, dtype=dtype))
    M_ill = U @ torch.diag(s) @ V.transpose(0, 1)

    Q_ill, R_ill = qless_orthogonalize(M_ill, 32)
    Q_ref, R_ref = torch.linalg.qr(M_ill, mode="reduced")
    assert torch.allclose(R_ill, R_ref, atol=tol, rtol=tol)
    assert torch.allclose(Q_ill, Q_ref, atol=tol, rtol=tol)

    # A well-conditioned input of the same shape should NOT take the same
    # fallback -- its Q should differ from a fresh, independent
    # torch.linalg.qr call (different algorithm, not just different signs).
    M_well = torch.randn(rows, cols, device=device, dtype=dtype)
    Q_well, _ = qless_orthogonalize(M_well, 32)
    Q_well_ref, _ = torch.linalg.qr(M_well, mode="reduced")
    assert not torch.allclose(Q_well, Q_well_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_tsqr_r_cols_zero_returns_empty(device, dtype):
    """`cols == 0` is an explicit early-return branch in tsqr_r, never
    exercised by any rows>>cols-shaped test above."""
    _skip_if_unavailable(device)
    M = torch.randn(50, 0, device=device, dtype=dtype)
    R = tsqr_r(M, 16)
    assert R.shape == (0, 0)


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize("rows,cols", [(6, 6), (6, 10)])
def test_tsqr_r_not_tall_skinny_matches_linalg_qr(device, dtype, rows, cols):
    """`rows <= cols` (square or wide) takes tsqr_r's "not tall-skinny"
    single-Cholesky-QR-block branch, skipping the tree-reduction loop
    entirely -- untested by the rows>>cols-only parametrizations above."""
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9

    M = torch.randn(rows, cols, device=device, dtype=dtype)
    R = tsqr_r(M, 16)
    assert R.shape == (min(rows, cols), cols)

    _, R_ref = torch.linalg.qr(M, mode="reduced")
    sign = torch.sign(torch.diagonal(R)) * torch.sign(torch.diagonal(R_ref))
    R_aligned = R * sign.unsqueeze(1)
    assert torch.allclose(R_aligned, R_ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_qless_orthogonalize_wide_input_matches_linalg_qr(device, dtype):
    """`M.size(0) < M.size(1)` (wide input) takes qless_orthogonalize's
    direct `torch::linalg_qr` fallback (no square R to invert) -- untested by
    the tall-skinny-only parametrizations above."""
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9

    rows, cols = 5, 9
    M = torch.randn(rows, cols, device=device, dtype=dtype)
    Q, R = qless_orthogonalize(M, 16)
    assert Q.shape == (rows, rows)
    assert R.shape == (rows, cols)

    Q_ref, R_ref = torch.linalg.qr(M, mode="reduced")
    assert torch.allclose(Q, Q_ref, atol=tol, rtol=tol)
    assert torch.allclose(R, R_ref, atol=tol, rtol=tol)
