from typing import List, Optional, Tuple, Union, Literal
from itertools import product

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch as MplPatch
from matplotlib.lines import Line2D
import pyvista as pv

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.cad._patch import PatchActors
from ttnte.cpp.ttnte_python.mesh import IGAMesh
from ttnte.mesh._gather import GatheredPatch, gather_mesh_patches
from ttnte.visualization.style import MplPatchStyle, PvPatchStyle
from ttnte.visualization.window import (
    export_pv_plot,
    finalize_mpl_figure,
    init_plotter,
    label_mpl_axes,
    pad_ctrlpts_limits,
    setup_pv_camera,
)


def plane_intersects_bbox(
    patch: Patch,
    normal: Union[Tuple[float], List[float]],
    origin: Union[Tuple[float], List[float]],
):
    """Check if a plane intersects the bounding box of a patch.

    Parameters
    ----------
    patch: Patch
        The patch.
    normal: tuple or list of float
        The normal of the intersecting plane.
    origin: tuple or list of float
        A point on the intersecting plane.

    Returns
    -------
    intersects: bool
        Whether the plane intersects the bounding box of the patch.
    """
    bbox = patch.get_bbox().cpu().numpy()
    pmin = bbox[0]
    pmax = bbox[1]

    # Get the corners of the bounding box
    corners = np.array(list(product(*zip(pmin, pmax))))

    # Calculate the distance to the plane
    distances = np.dot(corners - np.array(origin), np.array(normal))

    return np.max(distances) >= 0 and np.min(distances) <= 0


def plot(
    self: IGAMesh,
    resolution: int = 25,
    show_ctrlpts: bool = False,
    show_ctrlnet: bool = False,
    show_boundary: bool = False,
    filename: Optional[str] = None,
    style: Optional[Union[MplPatchStyle, PvPatchStyle]] = None,
    backend: Literal["matplotlib", "pyvista", "auto"] = "auto",
    solution=None,
    gather: bool = False,
    field_label: Optional[str] = None,
    **kwargs,
) -> Tuple[pv.Plotter, List[PatchActors]] | plt.Axes | None:
    """Plot the multi-patch mesh using matplotlib or PyVista. If `style.normal != None
    and the mesh's patches are 3-D` then `show_ctrlpts` and `show_ctrlnet` are forced to
    be `False` and a 2-D slice of the 3-D model is plotted based on a plane with a
    normal vector `style.normal` and a point existing on the plane `style.origin`.

    With `gather=False` (the default), only this rank's own local patches (`self.blocks`)
    are plotted -- no MPI. Passing `gather=True` additionally gathers every OTHER rank's
    patches (using `self.gid2rank`, always populated -- trivially if the mesh was never
    distributed) so every rank's patch appears in one composited plot; this makes the call
    collective (every rank must call it) -- see `ttnte.mesh._gather.gather_mesh_patches()`.

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
    solution: ttnte.driver.TransportSolution, optional
        If given (together with `gather=True`), every patch is colored by its field via
        `solution.get_local_field(gid)` -- must be spatial-only (`compute_scalar_flux()`'s
        result). Ignored if `gather` is not also `True`.
    gather: bool, default=False
        If `True`, gathers every rank's patches for a whole-mesh composited plot (always
        collective); if `False`, only this rank's own local patches are plotted (no MPI).
    field_label: str, optional
        Legend/colorbar label to use when coloring by `solution` (e.g. `"Scalar Flux"`).
        Defaults to `"Field"` if not given. Ignored if `solution` is not given.
    **kwargs: dict of any
        This includes the `plotter` for `backend == "pyvista"` and the `ax` for
        `backend == "matplotlib"`.

    Returns
    -------
    result: tuple of pyvista.Plotter and a list of ttnte._patch.PatchActors, matplotlib.pyplot.Axes, or None
        If `backend == "matplotlib" and filename == None` then the matplotlib axes is
        returned. If `backend == "pyvista" and filename  == None` then the PyVista plotter
        and patch actors are returned. Otherwise `None` is returned or when `ttnte.mpi_context.rank != 0`.
    """
    # Gather every rank's patches first if requested -- collective, so this
    # must run before any early return below (every rank must participate).
    if gather:
        blocks = gather_mesh_patches(self, resolution, solution=solution, root=0)
    else:
        blocks = self.blocks

    # Return early for other MPI ranks
    if mpi_context.rank != 0 or len(blocks) == 0:
        return None

    # Figure out auto backend and turn off control point/net
    # plotting if we're slicing a 3-D object
    if backend == "auto":
        if style is None:
            style = MplPatchStyle() if blocks[0].ndim < 3 else PvPatchStyle()
        backend = "matplotlib" if isinstance(style, MplPatchStyle) else "pyvista"

    # Generate the default style if it hasn't been given
    style = (
        style
        if style is not None
        else (MplPatchStyle() if backend == "matplotlib" else PvPatchStyle())
    )
    is_3d = blocks[0].ndim == 3 and style.normal is None

    # A given `solution` always wins, coloring every patch by its field
    # regardless of `style.colorby` -- matches Patch.plot()'s own
    # field-always-wins rule (see plot()'s effective_colorby there).
    field_mode = solution is not None

    # Compute each patch's label once (reused for the color mapping and the plot loop)
    effective_field_label = field_label if field_label is not None else "Field"
    patch_labels = [
        (
            effective_field_label
            if field_mode
            else (
                patch.fill.to_string()
                if style.colorby == "material"
                else patch.label.to_string()
            )
        )
        for patch in blocks
    ]
    unique_labels = list(set(patch_labels))

    # Build the explicit color mapping dictionary
    original_color = style.mesh.color
    original_facecolor = getattr(style.mesh, "facecolor", None)
    original_cmap = style.mesh.cmap
    # MplPatchStyle.mesh (SurfaceStyle) uses vmin/vmax; PvPatchStyle.mesh
    # (AddMeshStyle) uses a single clim=(min, max) tuple instead.
    has_vmin_vmax = hasattr(style.mesh, "vmin")
    original_vmin = style.mesh.vmin if has_vmin_vmax else None
    original_vmax = style.mesh.vmax if has_vmin_vmax else None
    original_clim = getattr(style.mesh, "clim", None)
    color_mapping = {}
    used_mapping = {}

    # When coloring by field, use ONE color scale across every patch (not
    # each patch auto-scaling to its own local min/max), so the same color
    # means the same value everywhere in the composited plot.
    color_scale_already_set = (
        (style.mesh.vmin is not None or style.mesh.vmax is not None)
        if has_vmin_vmax
        else original_clim is not None
    )
    if solution is not None and not color_scale_already_set:
        field_values = [
            (
                patch.field_values
                if isinstance(patch, GatheredPatch)
                else solution.get_local_field(patch.gid).to_dense()
            )
            for patch in blocks
            if not isinstance(patch, GatheredPatch) or patch.has_field
        ]
        if field_values:
            flat = torch.cat([v.flatten().to(torch.float64) for v in field_values])
            vmin, vmax = flat.min().item(), flat.max().item()
            if has_vmin_vmax:
                style.mesh.vmin = vmin
                style.mesh.vmax = vmax
            else:
                style.mesh.clim = (vmin, vmax)

    if isinstance(style.mesh.cmap, dict):
        # User provided an explicit dictionary mapping
        color_mapping = style.mesh.cmap

    else:
        # User provided a colormap name (or we use the default 'tab10')
        cmap_name = style.mesh.cmap if isinstance(style.mesh.cmap, str) else None

        # Generate color map based on the number of unique labels
        if cmap_name is None:
            if len(unique_labels) <= 10:
                cmap_name = "tab10"
            elif len(unique_labels) <= 20:
                cmap_name = "tab20"
            else:
                cmap_name = "turbo"

        cmap_obj = plt.get_cmap(cmap_name)

        # Extract the actual colors
        if hasattr(cmap_obj, "colors") and len(unique_labels) <= len(cmap_obj.colors):
            # It's a discrete map (like tab10) and we have enough colors
            raw_colors = cmap_obj.colors[: len(unique_labels)]
        else:
            # It's a continuous map (or we exceeded a discrete map's limit).
            # We sample it evenly from 0.0 to 1.0 to get exactly N unique colors.
            raw_colors = cmap_obj(np.linspace(0, 1, len(unique_labels)))

        # Map the unique labels to their generated hex colors
        for i, label_str in enumerate(unique_labels):
            color_mapping[label_str] = mcolors.to_hex(raw_colors[i])

    # Check if we are plotting a 3-D slice and disable control points and control net
    if style.normal is not None:
        show_ctrlpts = False
        show_ctrlnet = False
    elif blocks[0].ndim == 3:
        show_boundary = False

    # Create base figure or plotter
    standalone = True
    if backend == "matplotlib" and "ax" not in kwargs:
        kwargs["fig"] = plt.figure(**style.figure.to_dict())
        style.subplot.projection = "3d" if is_3d else None
        kwargs["ax"] = kwargs["fig"].add_subplot(111, **style.subplot.to_dict())
    elif backend == "pyvista" and "plotter" not in kwargs:
        kwargs["plotter"] = init_plotter(
            style.window,
            "trame" if filename and filename.endswith("html") else "static",
        )
    else:
        standalone = False

    # Loop through all the patches and plot them
    actors = []
    for patch, label_str in zip(blocks, patch_labels):
        if (
            patch.ndim == 3
            and style.normal is not None
            and not plane_intersects_bbox(patch, style.normal, style.origin)
        ):
            continue

        # Set the color for this region -- skipped in field_mode, which
        # keeps the caller's own (continuous) style.mesh.cmap intact for
        # every patch instead of flattening it to one solid color per
        # material/patch label.
        if not field_mode:
            assigned_color = color_mapping.get(label_str, "#808080")

            if label_str not in used_mapping:
                used_mapping[label_str] = assigned_color

            style.mesh.color = assigned_color
            if backend == "matplotlib":
                style.mesh.facecolor = assigned_color
                style.mesh.cmap = mcolors.ListedColormap([assigned_color])

        # A live patch's field comes straight from `solution`; a gathered
        # patch's field was already evaluated on the owning rank and just
        # needs its placeholder passed through (see GatheredPatch).
        if isinstance(patch, GatheredPatch):
            field = patch.field_placeholder if patch.has_field else None
        else:
            field = (
                solution.get_local_field(patch.gid).to_dense()
                if solution is not None
                else None
            )

        result = patch.plot(
            resolution=resolution,
            show_ctrlpts=show_ctrlpts,
            show_ctrlnet=show_ctrlnet,
            show_boundary=show_boundary,
            style=style,
            field=field,
            field_label=effective_field_label,
            **kwargs,
        )

        if backend == "pyvista":
            actors.append(result[1])

    # Restore the caller's style object (color/facecolor/cmap/vmin/vmax/clim
    # were mutated above)
    style.mesh.color = original_color
    style.mesh.cmap = original_cmap
    if has_vmin_vmax:
        style.mesh.vmin = original_vmin
        style.mesh.vmax = original_vmax
    else:
        style.mesh.clim = original_clim
    if backend == "matplotlib":
        style.mesh.facecolor = original_facecolor

    if standalone and backend == "matplotlib":
        ax = kwargs["ax"]
        assert isinstance(style, MplPatchStyle)
        label_mpl_axes(ax, style, is_3d)

        # Legend handles
        handles = []
        labels = []
        for label_str, color in used_mapping.items():
            handles.append(MplPatch(facecolor=color, edgecolor="black"))
            labels.append(label_str)

        if show_ctrlpts:
            handle = ax.scatter([0], [0], **style.control_points.to_dict())
            handle.remove()
            handles.append(handle)
            labels.append("Control Points")

        if show_ctrlnet:
            handles.append(Line2D([0], [0], **style.control_net.to_dict()))
            labels.append("Control Net")

        if show_boundary:
            handles.append(Line2D([0], [0], **style.boundary.to_dict()))
            labels.append("Patch Boundaries")

        if handles:
            ax.legend(
                handles=handles,
                labels=labels,
                **style.legend.to_dict(),
            )
        ax.set_aspect(style.aspect)

        # Adjust window to not cut the control points
        if show_ctrlpts:
            pad_ctrlpts_limits(ax)

        if filename is not None:
            # Save either a png or html
            fig = kwargs["fig"]
            assert fig is not None
            finalize_mpl_figure(fig, filename, style)
            return None

    elif standalone and backend == "pyvista":
        plotter = kwargs["plotter"]
        assert plotter is not None and isinstance(style, PvPatchStyle)
        setup_pv_camera(plotter, style, blocks[0].ndim)

        # Plot the legend
        labels = []
        for label_str, color in color_mapping.items():
            labels.append([label_str, color, "r"])

        if show_ctrlpts:
            labels.append(["Control Points", style.control_points.color, "o"])
        if show_ctrlnet:
            labels.append(["Control Net", style.control_net.color, "-"])
        if show_boundary:
            labels.append(["Patch Boundaries", style.boundary.color, "-"])

        style.legend.labels = labels
        plotter.add_legend(**style.legend.to_dict())

        if filename is not None:
            # Save either a png or html
            export_pv_plot(plotter, filename, style)
            return None

    if backend == "matplotlib":
        return kwargs["ax"]

    return kwargs["plotter"], actors


IGAMesh.plot = plot
