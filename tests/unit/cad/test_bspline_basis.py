import pytest
import torch
import numpy as np
from scipy.interpolate import BSpline

from ttnte.cad import BSplineBasis

test_params = [
    ("cpu", torch.float32),
    ("cpu", torch.float64),
    # ("cuda", torch.float32),
    # ("cuda", torch.float64),
]


@pytest.mark.parametrize("device, dtype", test_params)
def test_initialize(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize degree 2 B-Spline basis
    knotvector = torch.tensor([0, 0, 0, 0.5, 1, 1, 1], device=device, dtype=dtype)
    degree = 2
    label = "u"

    basis = BSplineBasis(knotvector, degree, label)

    assert torch.equal(basis.knotvector, knotvector)
    assert basis.degree == degree
    assert basis.label.to_string() == label


@pytest.mark.parametrize("device, dtype", test_params)
def test_initialize_normalizes_non_zero_start_knotvector(device, dtype):
    # A knot vector sliced from an interior sub-domain (e.g. igakit's
    # `NURBS.slice(0, 1/3, 2/3)`, which preserves absolute parameter values
    # rather than renormalizing) does not start at 0. normalize_knotvector()
    # must still map it onto [0, 1] -- dividing by the raw last-knot value
    # instead of the true span width silently mis-scales this case.
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    knotvector = torch.tensor(
        [1 / 3, 1 / 3, 1 / 3, 0.5, 2 / 3, 2 / 3, 2 / 3], device=device, dtype=dtype
    )
    degree = 2

    basis = BSplineBasis(knotvector, degree)

    expected = torch.tensor([0, 0, 0, 0.5, 1, 1, 1], device=device, dtype=dtype)
    assert torch.allclose(basis.knotvector, expected, atol=1e-6)


@pytest.mark.parametrize("device, dtype", test_params)
def test_find_spans(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Initialize degree 2 B-Spline basis
    knotvector = torch.tensor(
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
        device=device,
        dtype=dtype,
    )
    degree = 2

    basis = BSplineBasis(knotvector, degree)

    # Batch of points to test
    u = torch.tensor(
        [0, 0.15, 0.3, 0.56, 0.75, 0.80, 1],
        device=device,
        dtype=dtype,
    )

    spans = basis.find_spans(u)
    assert torch.equal(spans, torch.tensor([2, 2, 3, 4, 6, 6, 6], device=device))


@pytest.mark.parametrize("device, dtype", test_params)
def test_evaluate_all(device, dtype):
    # Skip if GPU is requested but not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # ==========================================
    # Evaluate a vector of points
    # Degree 2 B-Spline
    knotvector = torch.tensor(
        [0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 0.75, 1.0, 1.0, 1.0],
        device=device,
        dtype=dtype,
    )
    degree = 2

    # Batch of points to test
    u = torch.tensor(
        [0.0, 0.15, 0.3, 0.56, 0.75, 0.80, 1.0], device=device, dtype=dtype
    )

    # Run scipy for comparison
    num_basis_functions = knotvector.size(0) - degree - 1
    s = BSpline(knotvector, np.eye(num_basis_functions), degree)
    values_expected = torch.tensor(s(u.numpy()), device=device, dtype=dtype)

    # Run ttnte implementation
    basis = BSplineBasis(knotvector, degree)
    values = basis.evaluate_all(u)

    # Compare
    torch.testing.assert_close(values, values_expected)

    # Test derivatives implementation
    basis = BSplineBasis(knotvector, degree)
    values = basis.evaluate_all(u, derivative_order=1)

    # Compare zeroth derivative
    torch.testing.assert_close(values[:, 0, :], values_expected)

    derivs = []
    for j in range(num_basis_functions):
        c = np.zeros(num_basis_functions)
        c[j] = 1.0
        basis_func = BSpline(knotvector, c, degree)
        derivs.append(basis_func.derivative())

    values_expected = torch.tensor(
        np.array([derivs[i](u.numpy()) for i in range(num_basis_functions)]),
        device=device,
        dtype=dtype,
    ).T

    # Compare first derivative
    torch.testing.assert_close(values[:, 1, :], values_expected)
