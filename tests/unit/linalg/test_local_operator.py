import pytest
import torch
from ttnte.linalg import FoldedLocalOperator

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    ("cuda", torch.float32),
    ("cuda", torch.float64),
]

torch.manual_seed(11)


def _skip_if_unavailable(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _reference_apply(phi_left, a_core, phi_right, y):
    """Matches torchTT's `local_product` einsum: lsr,smnS,LSR,rnR->lmL."""
    return torch.einsum("lsr,smnS,LSR,rnR->lmL", phi_left, a_core, phi_right, y)


@pytest.mark.parametrize("device,dtype", test_params)
@pytest.mark.parametrize(
    "l,s,r,m,n,S,R,L", [(3, 2, 4, 5, 6, 3, 4, 3), (1, 1, 1, 3, 3, 1, 1, 1)]
)
def test_folded_apply_matches_naive_einsum(device, dtype, l, s, r, m, n, S, R, L):
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-10

    phi_left = torch.randn(l, s, r, device=device, dtype=dtype)
    a_core = torch.randn(s, m, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(L, S, R, device=device, dtype=dtype)
    y = torch.randn(r, n, R, device=device, dtype=dtype)

    op = FoldedLocalOperator.build(phi_left, a_core, phi_right)
    out = op.apply(y)
    ref = _reference_apply(phi_left, a_core, phi_right, y)

    assert out.shape == ref.shape
    assert torch.allclose(out, ref, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_folded_to_dense_matches_apply(device, dtype):
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-9
    l, s, r, m, n, S, R, L = 2, 3, 3, 4, 4, 2, 3, 2

    phi_left = torch.randn(l, s, r, device=device, dtype=dtype)
    a_core = torch.randn(s, m, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(L, S, R, device=device, dtype=dtype)

    op = FoldedLocalOperator.build(phi_left, a_core, phi_right)
    B = op.to_dense()
    assert B.shape == (l * m * L, r * n * R)

    y = torch.randn(r, n, R, device=device, dtype=dtype)
    out_apply = op.apply(y).reshape(-1)
    out_dense = B @ y.reshape(-1)

    assert torch.allclose(out_apply, out_dense, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_folded_operator_regularization(device, dtype):
    """`regularization` shifts a square operator by `regularization * I`:

    `apply()` adds `regularization * y`, `to_dense()` adds `regularization * I`.
    Zero (the default) must reproduce the unregularized operator exactly.
    """
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    # Square: l == r, m == n, L == R.
    l, s, r, m, n, S, R, L = 3, 2, 3, 4, 4, 2, 3, 3

    phi_left = torch.randn(l, s, r, device=device, dtype=dtype)
    a_core = torch.randn(s, m, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(L, S, R, device=device, dtype=dtype)
    y = torch.randn(r, n, R, device=device, dtype=dtype)

    op = FoldedLocalOperator.build(phi_left, a_core, phi_right)
    op_zero_reg = FoldedLocalOperator.build(
        phi_left, a_core, phi_right, regularization=0.0
    )
    lam = 0.37
    op_reg = FoldedLocalOperator.build(phi_left, a_core, phi_right, regularization=lam)

    # Default/explicit-zero regularization reproduces the plain operator.
    assert torch.allclose(op.apply(y), op_zero_reg.apply(y), atol=tol, rtol=tol)
    assert torch.allclose(op.to_dense(), op_zero_reg.to_dense(), atol=tol, rtol=tol)

    # Nonzero regularization shifts apply()/to_dense() by lam * y / lam * I.
    expected_apply = op.apply(y) + lam * y
    assert torch.allclose(op_reg.apply(y), expected_apply, atol=tol, rtol=tol)

    dense = op.to_dense()
    eye = torch.eye(dense.shape[0], device=device, dtype=dtype)
    expected_dense = dense + lam * eye
    assert torch.allclose(op_reg.to_dense(), expected_dense, atol=tol, rtol=tol)

    # apply() and to_dense() must stay consistent under regularization too.
    out_apply = op_reg.apply(y).reshape(-1)
    out_dense = op_reg.to_dense() @ y.reshape(-1)
    assert torch.allclose(out_apply, out_dense, atol=tol, rtol=tol)


@pytest.mark.parametrize("device,dtype", test_params)
def test_folded_operator_reused_across_multiple_ys(device, dtype):
    """The whole point of caching build() is correctness under reuse across many
    different candidate vectors (as GMRES would do)."""
    _skip_if_unavailable(device)
    tol = 1e-4 if dtype == torch.float32 else 1e-10
    l, s, r, m, n, S, R, L = 2, 2, 3, 3, 3, 2, 3, 2

    phi_left = torch.randn(l, s, r, device=device, dtype=dtype)
    a_core = torch.randn(s, m, n, S, device=device, dtype=dtype)
    phi_right = torch.randn(L, S, R, device=device, dtype=dtype)
    op = FoldedLocalOperator.build(phi_left, a_core, phi_right)

    for _ in range(5):
        y = torch.randn(r, n, R, device=device, dtype=dtype)
        out = op.apply(y)
        ref = _reference_apply(phi_left, a_core, phi_right, y)
        assert torch.allclose(out, ref, atol=tol, rtol=tol)
