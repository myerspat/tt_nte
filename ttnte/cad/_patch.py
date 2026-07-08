from typing import List, Literal, Optional, Tuple, Union
from dataclasses import dataclass
from itertools import product

import torch
import numpy as np
import pyvista as pv
from igakit.cad import NURBS
from matplotlib.collections import PolyCollection, QuadMesh
from matplotlib.patches import Patch as MplPatch
import matplotlib.pyplot as plt

from ttnte import mpi_context
from ttnte.cpp.ttnte_python.xs import MaterialLabel
from ttnte.cpp.ttnte_python.cad import BSplineBasis, Patch
from ttnte.visualization.style import MplPatchStyle, PvPatchStyle
from ttnte.visualization.window import (
    export_pv_plot,
    finalize_mpl_figure,
    init_plotter,
    label_mpl_axes,
    pad_ctrlpts_limits,
    setup_pv_camera,
)


@dataclass
class PatchActors:
    """Container for PyVista actors associated with plotting a patch.

    Attributes
    ----------
    patch: pyvista.Actor or None
        The main actor for the patch fill.
    control_points: pyvista.Actor or None
        The actor for the control points.
    control_net: pyvista.Actor or None
        The actor for the control net.
    boundary: pyvista.Actor or None
        The patch boundary actor.
    """

    patch: pv.Actor | None = None
    control_points: pv.Actor | None = None
    control_net: pv.Actor | None = None
    boundary: pv.Actor | None = None


def patch_slice(
    self: Patch,
    resolution: int,
    normal: Optional[Union[List[float], Tuple[float]]],
    origin: Optional[Union[List[float], Tuple[float]]],
) -> Union[pv.PolyData, pv.StructuredGrid]:
    """Slice a patch based on the plane defined by a normal vector and a point on the
    plane. If `normal` and `origin` are `None` then the evaluated patch data is returned
    without slicing.

    Parameters
    ----------
    resolution: int
        The number of parametric points to evaluate along a given parametric
        dimension.
    normal: list of float or tuple of float or None
        The normal vector of the plane.
    origin: list of float or tuple of float or None
        The point that lives on the plane.

    Returns
    -------
    grid: pyvista.PolyData or pyvista.StructuredGrid
        The evaluated points of the patch.
    """
    # Create parametric grid
    tensor_product_pts = self.ndim * [
        torch.linspace(0, 1, resolution, device=self.device, dtype=self.dtype)
    ]
    pts = self(tensor_product_pts)
    pts = [pts[..., i].cpu().numpy() for i in range(pts.shape[-1])]
    pts += ((3 - self.ndim) * [np.zeros_like(pts[0])]) if self.ndim <= 2 else []
    grid = pv.StructuredGrid(*pts)

    # Add scalar attributes
    grid.cell_data["Material ID"] = np.full(grid.n_cells, self.fill.to_int())
    grid.cell_data["Patch ID"] = np.full(grid.n_cells, self.label.to_int())

    # Only slice if we have a 3-D patch
    if self.ndim == 3 and normal is not None and origin is not None:
        grid = grid.slice(normal=normal, origin=origin)
        # Check if the slice failed
        if grid.n_points == 0:
            raise ValueError(
                f"The slice plane (`normal={normal}`, `origin={origin}`) "
                f"missed the patch with bounding box \n{self.get_bbox().cpu().numpy()}"
            )
    assert isinstance(grid, pv.PolyData) or isinstance(grid, pv.StructuredGrid)

    return grid


def _iter_vtk_lines(lines: np.ndarray, points: np.ndarray):
    """Yields coordinate segments from a flat VTK lines array."""
    i = 0
    while i < len(lines):
        n_points = lines[i]
        yield points[lines[i + 1 : i + 1 + n_points]]
        i += 1 + n_points


def _plot_matplotlib(
    self: Patch,
    resolution: int,
    show_ctrlpts: bool,
    show_ctrlnet: bool,
    show_boundary: bool,
    ax: Optional[plt.Axes],
    filename: Optional[str],
    label: str,
    style: MplPatchStyle,
) -> plt.Axes | None:
    """Plot a patch using matplotlib.

    Parameters
    ----------
    resolution: int
        The number of parametric points to evaluate along a parametric dimension.
    show_ctrlpts: bool
        Whether to plot the control points.
    show_ctrlnet: bool
        Whether to plot the control net.
    show_boundary: bool
        Whether to plot the patch boundary (not allowed in 3-D).
    ax: matplotlib.pyplot.Axes or None
        The existing axes to add to.
    filename:
        The name of the file to save to.
    label: str
        The fill label of the patch.
    style: ttnte.visualization.style.MplPatchStyle
        The settings for plotting.

    Returns
    -------
    ax: matplotlib.pyplot.Axes or None
        The resulting axes object with the added plots for this patch. If `filename != None`
        and `ax == None` then the plot is saved and the axes and figure closed. Therefore,
        `None` is returned.
    """
    # Run slice algorithm
    grid = patch_slice(
        self, resolution=resolution, normal=style.normal, origin=style.origin
    )
    is_3d = self.ndim == 3 and style.normal is None
    points = grid.points
    if style.normal is None or self.ndim <= 2:
        points = points.reshape(self.ndim * [resolution] + [3])
    handles = []
    labels = []

    # Create axes if not given
    standalone = ax is None
    fig = None
    if standalone:
        fig = plt.figure(**style.figure.to_dict())
        style.subplot.projection = "3d" if is_3d else None
        ax = fig.add_subplot(111, **style.subplot.to_dict())
    assert ax is not None

    if self.ndim == 1:
        # Plot 1-D geometry
        X, Y = points[..., 0], points[..., 1]
        handles.append(ax.plot(X, Y, **style.mesh.to_plot()))
        labels.append(label)

    elif self.ndim == 2:
        # Plot 2-D surface
        X, Y, Z = points[..., 0], points[..., 1], points[:-1, :-1, 2]
        style.mesh.shading = "flat"
        handles.append(ax.pcolormesh(X, Y, Z, **style.mesh.to_pcolormesh()))
        labels.append(label)

    elif is_3d:
        # Plot 3-D geometry
        settings = style.mesh.to_plot_surface()
        # Iterate over the 3 parametric axes (U, V, W) and the 2 boundaries (start, end)
        handle = None
        for axis in (0, 1, 2):
            for idx in (0, -1):
                face_x = np.take(points[..., 0], idx, axis=axis)
                face_y = np.take(points[..., 1], idx, axis=axis)
                face_z = np.take(points[..., 2], idx, axis=axis)
                handle = ax.plot_surface(face_x, face_y, face_z, **settings)
        handles.append(handle)
        labels.append(label)

    else:
        n = np.array(style.normal, dtype=float)
        n /= np.linalg.norm(n)

        # 1. Determine the dominant CAD axes to anchor our 2D plot coordinates.
        # This guarantees the 2D plot respects physical X/Y/Z directions and prevents mirroring.
        if abs(n[2]) > 0.5:  # Z-normal dominant (slice is mostly X-Y)
            ref_u, ref_v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
        elif abs(n[0]) > 0.5:  # X-normal dominant (slice is mostly Y-Z)
            ref_u, ref_v = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
        else:  # Y-normal dominant (slice is mostly X-Z)
            ref_u, ref_v = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])

        # 2. Project the primary reference axis onto the plane to act as horizontal (U)
        u = ref_u - np.dot(ref_u, n) * n
        u /= np.linalg.norm(u)

        # 3. Project the secondary reference axis and orthogonalize against U to act as vertical (V)
        v = ref_v - np.dot(ref_v, n) * n
        v = v - np.dot(v, u) * u
        v /= np.linalg.norm(v)

        # --- PLOT THE GEOMETRY ---
        surf = grid.extract_surface(algorithm="dataset_surface").triangulate()
        if surf.n_cells > 0:
            faces = surf.faces.reshape((-1, 4))[:, 1:4]

            # Project the points onto our physically-anchored plane
            u_coords = np.dot(surf.points, u)
            v_coords = np.dot(surf.points, v)
            points_2d = np.stack([u_coords, v_coords], axis=-1)

            poly = PolyCollection(points_2d[faces], **style.mesh.to_poly())
            ax.add_collection(poly)
            ax.autoscale(tight=True)
            handles.append(poly)
            labels.append(label)

    # Plot control points
    if show_ctrlpts:
        labels.append("Control Points")
        c_pts = self.ctrlpts.cpu().numpy()
        cX, cY = c_pts[..., 0], c_pts[..., 1]
        if is_3d:
            cZ = c_pts[..., 2]
            # Plot only the boundaries
            handle = None
            for idx, axis in product([0, -1], [0, 1, 2]):
                handle = ax.scatter(
                    np.take(cX, idx, axis).flatten(),
                    np.take(cY, idx, axis).flatten(),
                    np.take(cZ, idx, axis).flatten(),
                    **style.control_points.to_dict(),
                )
            handles.append(handle)
        else:
            handles.append(
                ax.scatter(cX.flatten(), cY.flatten(), **style.control_points.to_dict())
            )

    # Plot control net
    if show_ctrlnet:
        labels.append("Control Net")
        c_pts = self.ctrlpts.cpu().numpy()
        handle = None

        for i in range(self.ndim):
            # Move our active line-drawing axis to the second-to-last position
            net_lines = np.moveaxis(c_pts, i, -2)

            # This captures the shape of all the OTHER dimensions fixing the line
            other_shape = net_lines.shape[:-2]

            # Loop over every unique line coordinate in the tensor grid
            for idx in np.ndindex(other_shape):
                # A line lives on the boundary if ANY of its fixed coordinates
                # are at the first (0) or last (max - 1) slice of their axis
                is_boundary = any(
                    coord == 0 or coord == (max_val - 1)
                    for coord, max_val in zip(idx, other_shape)
                )

                if is_boundary or not is_3d:
                    # Extract the clean 1D array of coordinates for this boundary line
                    line = net_lines[idx]

                    if is_3d:
                        handle = ax.plot(
                            line[:, 0],
                            line[:, 1],
                            line[:, 2],
                            **style.control_net.to_dict(),
                        )
                    else:
                        handle = ax.plot(
                            line[:, 0],
                            line[:, 1],
                            **style.control_net.to_dict(),
                        )

        handles.append(handle)

    # Plot the boundary
    if show_boundary:
        labels.append("Boundary")
        # Extract boundary and control points
        boundary = grid.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
            feature_edges=False,
        )

        handle = None
        if boundary.n_cells > 0:
            for segment in _iter_vtk_lines(boundary.lines, boundary.points):
                if is_3d:
                    handle = ax.plot(
                        segment[:, 0],
                        segment[:, 1],
                        segment[:, 2],
                        **style.boundary.to_dict(),
                    )
                else:
                    handle = ax.plot(
                        segment[:, 0], segment[:, 1], **style.boundary.to_dict()
                    )
        handles.append(handle)

    if standalone:
        label_mpl_axes(ax, style, is_3d)

        # Clean legend handles
        clean_handles = []
        for h in handles:
            h_obj = h[0] if isinstance(h, list) else h

            if isinstance(h_obj, QuadMesh):
                # Extract the color from the QuadMesh and make a standard Patch square
                colors = h_obj.get_facecolor()
                c = colors[0] if len(colors) > 0 else "gray"
                clean_handles.append(MplPatch(facecolor=c, edgecolor="black"))
            else:
                clean_handles.append(h_obj)

        ax.legend(
            handles=clean_handles,
            labels=labels,
            **style.legend.to_dict(),
        )
        ax.set_aspect(style.aspect)

        # Adjust window to not cut the control points
        if show_ctrlpts:
            pad_ctrlpts_limits(ax)

        if filename is not None:
            # Save either a png or html
            assert fig is not None
            finalize_mpl_figure(fig, filename, style)
            return None

    return ax


def _plot_pyvista(
    self: Patch,
    resolution: int,
    show_ctrlpts: bool,
    show_ctrlnet: bool,
    show_boundary: bool,
    plotter: Optional[pv.Plotter],
    filename: Optional[str],
    label: str,
    style: PvPatchStyle,
) -> Tuple[pv.Plotter, PatchActors] | None:
    """Plot a patch with PyVista.

    Parameters
    ----------
    resolution: int
        The number of parametric points to evaluate along a parametric dimension.
    show_ctrlpts: bool
        Whether to plot the control points.
    show_ctrlnet: bool
        Whether to plot the control net.
    show_boundary: bool
        Whether to plot the patch boundary (not allowed in 3-D).
    plotter: pyvista.Plotter or None
        The existing plotter.
    filename:
        The name of the file to save to.
    label: str
        The fill label of the patch.
    style: ttnte.visualization.style.PvPatchStyle
        The settings for plotting.

    Returns
    -------
    plotter: pyvista.Plotter or None
        The resulting plotter with the new actors. If `plotter == None` and `filename != None`
        then the plot is saved and `None` is returned.
    actors: ttnte._patch.PatchActors or None
        The resulting actors. If `plotter == None` and `filename != None` then this is `None`.
    """
    # Create plotter if we don't already have one
    standalone = plotter is None
    if standalone:
        plotter = init_plotter(
            style.window,
            "trame" if filename and filename.endswith("html") else "static",
        )
    assert isinstance(plotter, pv.Plotter)

    # Create physical space grid
    grid = patch_slice(
        self, resolution=resolution, normal=style.normal, origin=style.origin
    )

    # Create labels list
    labels = [[label, style.mesh.color, "r"]]

    # Add to plotter
    patch_actor = plotter.add_mesh(grid, **style.mesh.to_dict())

    ctrlpts_actor = None
    ctrlnet_actor = None
    if show_ctrlpts or show_ctrlnet:
        # Create control points
        ctrlpts = np.zeros((*self.ctrlpts.shape[:-1], 3))
        ctrlpts[..., : self.ndim] = self.ctrlpts.cpu().numpy()

        # Add control points to plot
        if show_ctrlpts:
            ctrlpts_actor = plotter.add_mesh(
                pv.PolyData(ctrlpts.reshape((-1, 3))), **style.control_points.to_dict()
            )
            labels.append(["Control Points", style.control_points.color, "o"])

        # Add control net to plot
        if show_ctrlnet:
            ctrlnet_actor = plotter.add_mesh(
                pv.StructuredGrid(*(ctrlpts[..., i] for i in range(3))),
                **style.control_net.to_dict(),
            )
            labels.append(["Control Net", style.control_net.color, "-"])

    boundary_actor = None
    if show_boundary:
        # Extract boundary and control points
        boundary = grid.extract_feature_edges(
            boundary_edges=True,
            non_manifold_edges=False,
            manifold_edges=False,
            feature_edges=False,
        )

        boundary_actor = plotter.add_mesh(boundary, **style.boundary.to_dict())
        labels.append(["Boundary", style.boundary.color, "-"])

    if standalone:
        setup_pv_camera(plotter, style, self.ndim)

        # Plot the legend
        style.legend.labels = (
            labels if style.legend.labels is None else style.legend.labels
        )
        plotter.theme.font.color = "black"
        plotter.add_legend(**style.legend.to_dict())

        if filename is not None:
            # Save either a png or html
            export_pv_plot(plotter, filename, style)
            return None

    return plotter, PatchActors(
        patch_actor, ctrlpts_actor, ctrlnet_actor, boundary_actor
    )


@staticmethod
def from_igakit(
    kitpatch: NURBS,
    label: Optional[str] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    fill: Optional[MaterialLabel] = None,
    drop_invariant_dims: bool = True,
    tol: float = 1e-10,
):
    """Build a ``ttnte.cad.Patch`` from an ``igakit.cad.NURBS``.

    Parameters
    ----------
    kitpatch: igakit.cad.NURBS
        Input geometric patch.
    label: str, optional
        Name of the patch.
    device: torch.device, default=torch.get_default_device()
        Device to initialize the patch on.
    dtype: torch.dtype, default=torch.get_default_dtype()
        Data type to initialize the control points and knot vectors as.
    fill: torch.xs.MaterialLabel, optional
        Fill of the patch.
    drop_invariant_dims: bool, default=True
        Drop spatial dimensions that do not vary by control point.
    tol: float, default=1e-10
        The tolerance for checking if this is a B-spline or NURBS and
        which dimensions do not very.

    Return
    ------
    patch: ttnte.cad.Patch
        Valid patch.
    """
    # Evaluate dynamic defaults
    device = device if device is not None else torch.get_default_device()
    dtype = dtype if dtype is not None else torch.get_default_dtype()

    patch = Patch(label)

    # Create the basis
    basis = [
        BSplineBasis(
            torch.tensor(kitpatch.knots[i], device=device, dtype=dtype),
            kitpatch.degree[i],
        )
        for i in range(kitpatch.dim)
    ]
    patch.set_basis(basis)

    # Convert the control points to a pytorch tensor
    ctrlptsw = torch.tensor(kitpatch.control, device=device, dtype=dtype)

    if drop_invariant_dims:
        # Separate and un-weight control points from their weights
        weights = ctrlptsw[..., -1:]
        ctrlpts = ctrlptsw[..., :-1] / weights

        # Flatten and check variation
        flat_ctrlpts = ctrlpts.reshape((-1, ctrlpts.shape[-1]))
        variation = flat_ctrlpts.max(dim=0).values - flat_ctrlpts.min(dim=0).values
        active_dims = variation > tol

        # Check this is not a single point patch
        if not active_dims.any():
            active_dims[0] = True

        # Append True to keep the weight dimension at the end
        keep_indices = torch.cat([active_dims, torch.tensor([True], device=device)])

        # Filter the control tensor
        ctrlptsw = ctrlptsw[..., keep_indices]

    # Set control points for B-spline or NURBS
    if (torch.abs(ctrlptsw[..., -1] - 1.0) < tol).all():
        # B-spline
        patch.set_ctrlpts(ctrlptsw[..., :-1])
    else:
        # NURBS
        patch.set_ctrlptsw(ctrlptsw)

    # Add fill
    if fill is not None:
        patch.fill = fill

    # Validate the patch
    patch.finalize()

    return patch


def plot(
    self: Patch,
    resolution: int = 25,
    show_ctrlpts: bool = False,
    show_ctrlnet: bool = False,
    show_boundary: bool = False,
    filename: Optional[str] = None,
    style: Optional[Union[MplPatchStyle, PvPatchStyle]] = None,
    backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
    **kwargs,
) -> Tuple[pv.Plotter, PatchActors] | plt.Axes | None:
    """Plot this patch using matplotlib or PyVista. If `style.normal != None and
    self.ndim == 3` then `show_ctrlpts` and `show_ctrlnet` are forced to be `False` and
    a 2-D slice of the 3-D model is plotted based on a plane with a normal vector
    `style.normal` and a point existing on the plane `style.origin`.

    Parameters
    ----------
    resolution: int
        The number of parametric points to evaluate along a parametric dimension.
    show_ctrlpts: bool
        Whether to plot the control points.
    show_ctrlnet: bool
        Whether to plot the control net.
    show_boundary: bool
        Whether to plot the patch boundary (not allowed in 3-D).
    filename:
        The name of the file to save to.
    style: ttnte.visualization.style.PvPatchStyle, ttnte.visualization.style.MplPatchStyle, or None, default=None
        The settings for plotting. If this is `None` then the default visualization settings are
        used for the chosen backend.
    backend: "matplotlib", "pyvista", or "auto"
        The plotting backend to use. Note that the backend must match the `style` passed. If
        `backend == "matplotlib"` then `isinstance(style, ttnte.visualization.style.MplPatchStyle)`
        must be `True` and the opposite is true for `backend == "pyvista"`. If `backend == "auto"`
        then PyVista is used for non-sliced 3-D plotting while matplotlib is used for the
        rest.
    **kwargs: dict of any
        This includes the `plotter` for `backend == "pyvista"` and the `ax` for
        `backend == "matplotlib"`.

    Returns
    -------
    result: tuple of pyvista.Plotter and ttnte._patch.PatchActors, matplotlib.pyplot.Axes, or None
        If `backend == "matplotlib" and filename == None` then the matplotlib axes is
        returned. If `backend == "pyvista" and filename  == None` then the PyVista plotter
        and patch actors are returned. Otherwise `None` is returned or when `ttnte.mpi_context.rank != 0`.
    """
    # Return early for other MPI ranks
    if mpi_context.rank != 0:
        return None

    # Figure out auto backend and turn off control point/net
    # plotting if we're slicing a 3-D object
    if backend == "auto":
        if style is None:
            style = MplPatchStyle() if self.ndim < 3 else PvPatchStyle()
        backend = "matplotlib" if isinstance(style, MplPatchStyle) else "pyvista"

    # Generate the default style if it hasn't been given
    style = (
        style
        if style is not None
        else (MplPatchStyle() if backend == "matplotlib" else PvPatchStyle())
    )

    # Get the label for this patch
    label = (
        self.fill.to_string() if style.colorby == "material" else self.label.to_string()
    )

    # Check if we are plotting a 3-D slice and disable control points and control net
    if style.normal is not None:
        show_ctrlpts = False
        show_ctrlnet = False
    elif self.ndim == 3:
        show_boundary = False

    if backend == "matplotlib":
        if isinstance(style, PvPatchStyle):
            raise ValueError(
                "The `backend='matplotlib'` but the style given was `PvPatchStyle` "
                "instead of `MplPatchStyle`"
            )

        ax = kwargs.get("ax", None)
        return _plot_matplotlib(
            self,
            resolution=resolution,
            show_ctrlpts=show_ctrlpts,
            show_ctrlnet=show_ctrlnet,
            show_boundary=show_boundary,
            ax=ax,
            filename=filename,
            label=label,
            style=style,
        )
    else:
        if isinstance(style, MplPatchStyle):
            raise ValueError(
                "The `backend='pyvista'` but the style given was `MplPatchStyle` "
                "instead of `PvPatchStyle`"
            )

        plotter = kwargs.get("plotter", None)
        return _plot_pyvista(
            self,
            resolution,
            show_ctrlpts=show_ctrlpts,
            show_ctrlnet=show_ctrlnet,
            show_boundary=show_boundary,
            plotter=plotter,
            filename=filename,
            label=label,
            style=style,
        )


# Add methods to the patch class
Patch.from_igakit = from_igakit
Patch.plot = plot
