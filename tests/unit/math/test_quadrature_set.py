import pytest
import torch

from ttnte.math import ProductQuadrature, QuadratureSet1D
from ttnte.linalg import State, TTEngine

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_integrate_1d(device, dtype):
    """`QuadratureSet1D.integrate()` should reduce the leading (angular) core of a State
    by a weighted sum, matching a dense tensordot."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    n, ns, ng = 4, 3, 2

    weights = torch.rand(n, device=device, dtype=dtype)
    weights /= weights.sum()
    points = torch.linspace(-1, 1, n, device=device, dtype=dtype)
    qset = QuadratureSet1D(points, weights, weighting_factor=2.0)

    tensor = torch.rand(n, ns, ng, device=device, dtype=dtype)
    state = State(TTEngine.from_dense(tensor))

    result = qset.integrate(state)
    actual = result.as_tt().to_dense().squeeze()

    expected = torch.einsum("i,ijk->jk", weights, tensor)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device, dtype", test_params)
def test_integrate_product_quadrature(device, dtype):
    """`ProductQuadrature.integrate()` should reduce both leading (polar, azimuthal)
    cores of a State by their factored weights."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    n_polar, n_azimuthal, ns = 3, 5, 4

    qset = ProductQuadrature.gauss_legendre_chebyshev(n_polar, n_azimuthal, ndim=3)
    qset.to_(device, dtype)
    w0, w1 = qset.factored_weights

    tensor = torch.rand(w0.shape[0], w1.shape[0], ns, device=device, dtype=dtype)
    state = State(TTEngine.from_dense(tensor))

    result = qset.integrate(state)
    actual = result.as_tt().to_dense().squeeze()

    expected = torch.einsum("i,j,ijk->k", w0, w1, tensor)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device, dtype", test_params)
def test_integrate_unsupported_format_raises(device, dtype):
    """A State without enough cores for the quadrature's angular dimensions should
    raise, rather than silently misinterpreting the layout."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    device = torch.device(device)
    n_polar, n_azimuthal = 3, 5

    qset = ProductQuadrature.gauss_legendre_chebyshev(n_polar, n_azimuthal, ndim=3)
    qset.to_(device, dtype)
    w0, _ = qset.factored_weights

    # Only one core -- not enough for a 2-angular-core ProductQuadrature.
    tensor = torch.rand(w0.shape[0], device=device, dtype=dtype)
    state = State(TTEngine.from_dense(tensor))

    with pytest.raises(RuntimeError):
        qset.integrate(state)
