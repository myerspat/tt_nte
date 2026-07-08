import os

import vtk
from PIL import Image
import numpy as np
import pyvista as pv
from ttnte.visualization.style import ScreenshotStyle, PlotterStyle


def init_plotter(style: PlotterStyle, jupyter_backend="static"):
    """"""
    pv.set_jupyter_backend(jupyter_backend)
    os.environ["EGL_LOG_LEVEL"] = "fatal"
    vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_OFF)
    vtk.vtkObject.GlobalWarningDisplayOff()
    return pv.Plotter(**style.to_dict())


def align_camera_to_slice(plotter: pv.Plotter, normal: list | tuple | None):
    """"""
    if normal is None:
        return

    # 1. Normalize the slice normal vector
    n = np.array(normal, dtype=float)
    n /= np.linalg.norm(n)

    # 2. Find the center of the geometry to use as the focal point
    focal_point = np.array(plotter.center)

    # 3. Step away from the center along the normal vector to set camera position
    # (Distance doesn't matter much for parallel/orthographic projections)
    distance = np.linalg.norm(plotter.renderer.get_pick_position() or 10.0)
    camera_position = focal_point + (n * distance)

    # 4. Compute a robust "View Up" vector
    # We pick a global reference vector (usually Z-axis for XY cuts, or Y-axis if slicing along Z)
    if np.abs(n[2]) > 0.99:
        global_up = np.array([0.0, 1.0, 0.0])  # Look along Z? Up is Y
    else:
        global_up = np.array([0.0, 0.0, 1.0])  # Otherwise? Up is Z

    # Project global_up onto the slice plane to make it perfectly orthogonal to the normal
    view_up = global_up - np.dot(global_up, n) * n
    view_up /= np.linalg.norm(view_up)

    # 5. Apply directly to the underlying VTK camera object
    plotter.camera.focal_point = focal_point
    plotter.camera.position = camera_position
    plotter.camera.up = view_up


def export_cropped_screenshot(
    plotter: pv.Plotter, filename: str, pad: int, style: ScreenshotStyle
):
    """"""
    style.filename = None
    style.return_img = True

    if not plotter._rendered:
        plotter.render()

    # Take raw snapshot using unpacked clean dictionary options
    img = plotter.screenshot(**style.to_dict())

    # Extract bounding box from non-white pixels
    mask = np.any(img[..., :3] != 255, axis=-1)
    coords = np.argwhere(mask)

    if coords.size > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        # Apply localized cushion boundaries using our style configuration variable
        y_min = max(0, y_min - pad)
        y_max = min(img.shape[0], y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(img.shape[1], x_max + pad)

        img = img[y_min:y_max, x_min:x_max]

    # Save output down to file asset
    Image.fromarray(img).save(filename)


def label_mpl_axes(ax, style, is_3d: bool):
    """Label a standalone matplotlib Axes, or clean up its 2-D spines."""
    ax.set_xlabel(style.xlabel)
    ax.set_ylabel(style.ylabel)
    if is_3d:
        ax.set_zlabel(style.zlabel)
    else:
        ax.spines[["right", "top"]].set_visible(False)


def pad_ctrlpts_limits(ax):
    """Expand the x/y limits so control points near the edge aren't clipped."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    diffx = (abs(xlim[0]) + abs(xlim[1])) * 0.05 / 2
    diffy = (abs(ylim[0]) + abs(ylim[1])) * 0.05 / 2
    ax.set_xlim((xlim[0] - diffx, xlim[1] + diffx))
    ax.set_ylim((ylim[0] - diffy, ylim[1] + diffy))


def finalize_mpl_figure(fig, filename: str, style):
    """Save a standalone matplotlib figure to `filename` and clear it."""
    fig.savefig(filename, **style.savefig.to_dict())
    fig.clear()


def setup_pv_camera(plotter: pv.Plotter, style, ndim: int):
    """Add the axes widget and align the camera for a standalone PyVista plot."""
    plotter.add_axes(**style.axes.to_dict())
    if ndim == 3:
        align_camera_to_slice(plotter, style.normal)
    else:
        plotter.view_xy()


def export_pv_plot(plotter: pv.Plotter, filename: str, style):
    """Export a standalone PyVista plot to `filename` (png or html) and close it."""
    if filename.endswith(".html"):
        plotter.export_html(filename)
    else:
        export_cropped_screenshot(
            plotter, filename, style.crop_padding, style.screenshot
        )
    plotter.close()
