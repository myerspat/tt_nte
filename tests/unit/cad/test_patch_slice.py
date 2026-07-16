import numpy as np
import pytest
import torch

from ttnte.cad._patch import patch_slice

from .test_patch import (
    _bspline_1d_patch,
    _bspline_2d_patch,
    _rational_2d_patch,
    _tols,
    test_params,
)


def _direct_field_grid(patch, resolution, field, device, dtype):
    """Evaluate `field` at the same tensor-product parametric grid `patch_slice()` uses,
    independently of patch_slice() itself, then flatten with the SAME 'F' order
    patch_slice() uses for point_data (see the plan/comments in _patch.py -- verified
    empirically to match pv.StructuredGrid's own internal point ordering)."""
    tensor_product_pts = patch.ndim * [
        torch.linspace(0, 1, resolution, device=device, dtype=dtype)
    ]
    mesh_coords = torch.meshgrid(tensor_product_pts, indexing="ij")
    flat_points = torch.stack([c.flatten() for c in mesh_coords], dim=-1)
    values = patch.evaluate_field(field, flat_points)[..., 0]
    values = values.reshape(patch.ndim * [resolution])
    return values.cpu().numpy().flatten(order="F")


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_slice_field_matches_evaluate_field_rational_2d(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    patch = _rational_2d_patch(device, dtype)
    atol, rtol, _ = _tols(dtype)
    resolution = 6

    # Use the x-coordinate channel of the patch's own control points as a
    # synthetic scalar field -- any per-control-point array works, this one
    # just gives non-trivial, independently-known values.
    field = patch.ctrlpts[..., 0:1].contiguous()

    grid = patch_slice(
        patch, resolution=resolution, normal=None, origin=None, field=field
    )
    assert "field" in grid.point_data
    field_values = grid.point_data["field"]
    assert field_values.shape == (resolution**patch.ndim,)

    direct = _direct_field_grid(patch, resolution, field, device, dtype)
    np.testing.assert_allclose(field_values, direct, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_slice_field_matches_evaluate_field_bspline_1d(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    patch = _bspline_1d_patch(device, dtype)
    atol, rtol, _ = _tols(dtype)
    resolution = 10

    field = patch.ctrlpts[..., 1:2].contiguous()

    grid = patch_slice(
        patch, resolution=resolution, normal=None, origin=None, field=field
    )
    field_values = grid.point_data["field"]
    assert field_values.shape == (resolution**patch.ndim,)

    direct = _direct_field_grid(patch, resolution, field, device, dtype)
    np.testing.assert_allclose(field_values, direct, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_slice_field_matches_evaluate_field_bspline_2d(device, dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    patch = _bspline_2d_patch(device, dtype)
    atol, rtol, _ = _tols(dtype)
    resolution = 6

    field = patch.ctrlpts[..., 2:3].contiguous()

    grid = patch_slice(
        patch, resolution=resolution, normal=None, origin=None, field=field
    )
    field_values = grid.point_data["field"]
    assert field_values.shape == (resolution**patch.ndim,)

    direct = _direct_field_grid(patch, resolution, field, device, dtype)
    np.testing.assert_allclose(field_values, direct, atol=atol, rtol=rtol)


@pytest.mark.parametrize("device, dtype", test_params)
def test_patch_slice_no_field_omits_point_data(device, dtype):
    """With no `field` given, patch_slice() should not add a "field" point_data array --
    regression check for the default (no-field) behavior."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    patch = _bspline_1d_patch(device, dtype)
    grid = patch_slice(patch, resolution=8, normal=None, origin=None)
    assert "field" not in grid.point_data
