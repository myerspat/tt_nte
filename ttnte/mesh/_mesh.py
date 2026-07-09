from typing import List, Optional, Tuple, Union, Literal
from itertools import product

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch as MplPatch
from matplotlib.lines import Line2D
import pyvista as pv

from ttnte import mpi_context
from ttnte.cad import Patch
from ttnte.cad._patch import PatchActors
from ttnte.cpp.ttnte_python.mesh import IGAMesh
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
    **kwargs,
) -> Tuple[pv.Plotter, List[PatchActors]] | plt.Axes | None:
    """Plot the multi-patch mesh using matplotlib or PyVista. If `style.normal != None
    and self.ndim == 3` then `show_ctrlpts` and `show_ctrlnet` are forced to be `False`
    and a 2-D slice of the 3-D model is plotted based on a plane with a normal vector
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
    result: tuple of pyvista.Plotter and a list of ttnte._patch.PatchActors, matplotlib.pyplot.Axes, or None
        If `backend == "matplotlib" and filename == None` then the matplotlib axes is
        returned. If `backend == "pyvista" and filename  == None` then the PyVista plotter
        and patch actors are returned. Otherwise `None` is returned or when `ttnte.mpi_context.rank != 0`.
    """
    # Return early for other MPI ranks
    if mpi_context.rank != 0 or len(self.blocks) == 0:
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
    is_3d = self.blocks[0].ndim == 3 and style.normal is None

    # Compute each patch's label once (reused for the color mapping and the plot loop)
    patch_labels = [
        (
            patch.fill.to_string()
            if style.colorby == "material"
            else patch.label.to_string()
        )
        for patch in self.blocks
    ]
    unique_labels = list(set(patch_labels))

    # Build the explicit color mapping dictionary
    original_color = style.mesh.color
    original_facecolor = getattr(style.mesh, "facecolor", None)
    original_cmap = style.mesh.cmap
    color_mapping = {}
    used_mapping = {}

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
    elif self.blocks[0].ndim == 3:
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
    for patch, label_str in zip(self.blocks, patch_labels):
        if (
            patch.ndim == 3
            and style.normal is not None
            and not plane_intersects_bbox(patch, style.normal, style.origin)
        ):
            continue

        # Set the color for this region
        assigned_color = color_mapping.get(label_str, "#808080")

        if label_str not in used_mapping:
            used_mapping[label_str] = assigned_color

        style.mesh.color = assigned_color
        if backend == "matplotlib":
            style.mesh.facecolor = assigned_color
            style.mesh.cmap = mcolors.ListedColormap([assigned_color])

        result = patch.plot(
            resolution=resolution,
            show_ctrlpts=show_ctrlpts,
            show_ctrlnet=show_ctrlnet,
            show_boundary=show_boundary,
            style=style,
            **kwargs,
        )

        if backend == "pyvista":
            actors.append(result[1])

    # Restore the caller's style object (color/facecolor/cmap were mutated per patch above)
    style.mesh.color = original_color
    style.mesh.cmap = original_cmap
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
        setup_pv_camera(plotter, style, self.blocks[0].ndim)

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
