import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
import torch
from igakit.cad import trilinear

from ttnte.cad import Patch
from ttnte.visualization.style import MplPatchStyle, PvPatchStyle

from .test_patch import _bspline_1d_patch, _bspline_2d_patch, _rational_2d_patch


def _trilinear_3d_patch(device, dtype):
    pts = np.zeros((2, 2, 2, 3))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                pts[i, j, k] = [i, j, k]
    return Patch.from_igakit(trilinear(pts), device=torch.device(device), dtype=dtype)


def _synthetic_field(patch, device, dtype, seed=0):
    ctrlpts_shape = [patch.get_ctrlpts_size(d) for d in range(patch.ndim)]
    torch.manual_seed(seed)
    return torch.rand(*ctrlpts_shape, 1, device=device, dtype=dtype)


device, dtype = "cpu", torch.float64


@pytest.mark.parametrize(
    "make_patch", [_rational_2d_patch, _bspline_1d_patch, _bspline_2d_patch]
)
def test_plot_field_matplotlib_smoke(make_patch):
    """Patch.plot(field=..., backend="matplotlib") must not raise, for both a
    1-D-parametric (rational and B-spline) and a 2-D-parametric patch, and must actually
    use the field (colorby="field" wins by default whenever a field is given -- see
    plot()'s effective_colorby)."""
    patch = make_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = MplPatchStyle()
    style.mesh.cmap = "viridis"
    ax = patch.plot(resolution=8, style=style, field=field, backend="matplotlib")
    assert ax is not None


@pytest.mark.parametrize(
    "make_patch", [_rational_2d_patch, _bspline_1d_patch, _bspline_2d_patch]
)
def test_plot_field_pyvista_smoke(make_patch):
    """Same as above, PyVista backend."""
    patch = make_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = PvPatchStyle()
    style.mesh.cmap = "viridis"
    plotter, actors = patch.plot(
        resolution=8, style=style, field=field, backend="pyvista"
    )
    assert plotter is not None
    assert actors.patch is not None


def test_plot_field_matplotlib_3d_full_smoke():
    """3-D (no slice) uses a different matplotlib code path (plot_surface with per-face
    facecolors) than the 1-D/2-D cases above -- exercise it separately."""
    patch = _trilinear_3d_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = MplPatchStyle()
    style.mesh.cmap = "viridis"
    ax = patch.plot(resolution=6, style=style, field=field, backend="matplotlib")
    assert ax is not None


def test_plot_field_matplotlib_3d_sliced_smoke():
    """3-D WITH a slicing plane uses yet another matplotlib code path (the sliced-to-2-D
    PolyCollection) -- exercise it separately."""
    patch = _trilinear_3d_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = MplPatchStyle()
    style.mesh.cmap = "viridis"
    style.normal = [0.0, 0.0, 1.0]
    style.origin = (0.5, 0.5, 0.5)
    ax = patch.plot(resolution=6, style=style, field=field, backend="matplotlib")
    assert ax is not None


def test_plot_field_pyvista_3d_smoke():
    patch = _trilinear_3d_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = PvPatchStyle()
    style.mesh.cmap = "viridis"
    plotter, actors = patch.plot(
        resolution=6, style=style, field=field, backend="pyvista"
    )
    assert plotter is not None
    assert actors.patch is not None


def test_plot_field_wins_over_colorby_default():
    """Field always wins for coloring, even with style.colorby left at its default
    ("material") -- see plot()'s effective_colorby."""
    patch = _bspline_1d_patch(device, dtype)
    field = _synthetic_field(patch, device, dtype)

    style = MplPatchStyle()
    assert style.colorby == "material"
    ax = patch.plot(resolution=8, style=style, field=field, backend="matplotlib")

    # The legend should say "Field", not the material name, confirming
    # effective_colorby picked field mode.
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert "Field" in legend_labels
