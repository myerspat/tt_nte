from dataclasses import dataclass, field, asdict
from typing import Union, Any, Sequence, Optional, Literal, Tuple, Dict, List

from matplotlib.colors import ListedColormap


class SerializableStyle:
    """A mixin/base class to add smart dictionary serialization to dataclasses."""

    def to_dict(self, filter_none: bool = True) -> dict[str, Any]:
        """Convert the dataclass fields to a standard Python dictionary.

        Parameters
        ----------

        filter_none: bool, default=True
            If True, skips fields that are explicitly None so PyVista fallback
            defaults aren't overridden.
        """
        data = asdict(self)
        if filter_none:
            return {k: v for k, v in data.items() if v is not None}
        return data


# =========================================================
# matplotlib settings


@dataclass
class PlotStyle(SerializableStyle):
    # --- Basic Appearance ---
    color: Any | None = None
    alpha: float | None = None
    label: str | None = None
    visible: bool = True
    zorder: float | None = None

    # --- Line Styling ---
    linestyle: str | None = "-"
    linewidth: float | None = None
    antialiased: bool | None = None
    drawstyle: Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"] = (
        "default"
    )

    # --- Marker Basics ---
    marker: Any | None = None
    markersize: float | None = None
    markevery: Any | None = None

    # --- Marker Edge & Face Styling ---
    markerfacecolor: Any | None = None
    markeredgecolor: Any | None = None
    markeredgewidth: float | None = None
    markerfacecoloralt: Any | None = None
    fillstyle: Literal["full", "left", "right", "bottom", "top", "none"] = "full"

    # --- Advanced Customization & Caps ---
    solid_capstyle: Literal["butt", "projecting", "round"] | None = None
    solid_joinstyle: Literal["miter", "round", "bevel"] | None = None
    dash_capstyle: Literal["butt", "projecting", "round"] | None = None
    dash_joinstyle: Literal["miter", "round", "bevel"] | None = None
    dashes: Sequence[float] | None = None
    gapcolor: Any | None = None


@dataclass
class SubplotStyle(SerializableStyle):
    # --- Projection & Subplot Type ---
    projection: str | Any | None = None
    polar: bool | None = None
    axes_class: Any | None = None

    # --- Coordinate & Axis Sharing ---
    sharex: Any | None = None
    sharey: Any | None = None

    # --- Basic Axes Appearance (Passed through to Axes constructor) ---
    facecolor: Any | None = None
    frameon: bool | None = None
    label: str | None = None

    # --- AspectRatios & Scaling ---
    aspect: str | float | None = None
    box_aspect: Sequence[float] | None = None


@dataclass
class FigureStyle(SerializableStyle):
    # --- Window & Canvas Identification ---
    num: int | str | None = None
    clear: bool = False

    # --- Canvas Dimensions & Resolution ---
    figsize: Tuple[float, float] | None = None
    dpi: float | None = None

    # --- Canvas Edge & Background Styling ---
    facecolor: Any | None = None
    edgecolor: Any | None = None
    frameon: bool | None = None
    linewidth: float | None = None

    # --- Modern Layout Managers ---
    layout: Literal["constrained", "compressed", "tight"] | Any | None = None


@dataclass
class PcolormeshStyle(SerializableStyle):
    """Configuration dataclass for matplotlib.pyplot.pcolormesh() and QuadMesh
    properties."""

    # --- Grid Layout & Interpolation ---
    shading: Literal["flat", "nearest", "gouraud", "auto"] = "auto"

    # --- Colormaps & Scalar Normalization ---
    cmap: Any | None = None
    vmin: float | None = None
    vmax: float | None = None
    norm: Any | None = None

    # --- Edge & Boundary Styling (QuadMesh Collection properties) ---
    edgecolors: Any | Literal["face", "none"] | None = "face"

    linewidths: float | None = 0.0
    antialiased: bool | None = None

    # --- Visibility & Performance Performance ---
    alpha: float | None = None
    rasterized: bool | None = True
    zorder: float | None = None
    visible: bool = True


@dataclass
class PlotSurfaceStyle(SerializableStyle):
    # --- Basic Appearance & Opacity ---
    color: Any | None = None
    alpha: float | None = None
    zorder: float | None = None

    # --- Colormaps & Scalar Normalization ---
    cmap: Any | None = None
    vmin: float | None = None
    vmax: float | None = None
    norm: Any | None = None

    # --- Edge & Grid Styling ---
    edgecolor: Any | Literal["face", "none"] | None = "none"
    linewidth: float | None = 0.0
    antialiased: bool | None = True

    # --- Downsampling & Performance ---
    rcount: int | None = 50
    ccount: int | None = 50
    rstride: int | None = None
    cstride: int | None = None

    # --- 3D Shading & Shading Models ---
    shade: bool | None = True
    lightsource: Any | None = None


@dataclass
class Poly3DCollectionStyle(SerializableStyle):
    # --- Face Styling (Polygons Interior) ---
    facecolors: Any | None = None
    alpha: float | None = None

    # --- Edge / Boundary Styling ---
    edgecolors: Any | Literal["face", "none"] | None = "none"
    linewidths: float | Sequence[float] | None = 0.0
    linestyles: str | Sequence[str] | None = "-"

    # --- Colormaps & Scalar Mapping ---
    cmap: Any | None = None
    norm: Any | None = None
    clim: Sequence[float] | None = None

    # --- 3D Shading & Sorting ---
    shade: bool = False
    zorder: float | None = None
    visible: bool = True

    # --- Advanced Customization ---
    antialiaseds: bool | None = True
    hatch: str | None = None


@dataclass
class ScatterStyle(SerializableStyle):
    # --- Marker Size & Styling ---
    s: float | Sequence[float] | None = 20.0
    marker: Any | None = "o"

    # --- Marker Interior Fill Coloring ---
    c: Any | None = None
    alpha: float | None = None

    # --- Marker Edge / Border Styling ---
    edgecolors: Any | Literal["face", "none"] | None = None
    linewidths: float | Sequence[float] | None = None

    # --- Colormaps & Scalar Mapping (used if 'c' is an array of floats) ---
    cmap: Any | None = None
    norm: Any | None = None
    vmin: float | None = None
    vmax: float | None = None

    # --- Layout & Performance Performance ---
    plotnonfinite: bool = False
    rasterized: bool | None = None
    zorder: float | None = None
    visible: bool = True


@dataclass
class SurfaceStyle(SerializableStyle):
    # plot settings
    color: Any | None = None
    alpha: float | None = None
    label: str | None = None
    visible: bool = True
    zorder: float | None = None
    linestyle: str | None = "-"
    linewidth: float | None = None
    antialiased: bool | None = None
    drawstyle: Literal["default", "steps", "steps-pre", "steps-mid", "steps-post"] = (
        "default"
    )
    marker: Any | None = None
    markersize: float | None = None
    markevery: Any | None = None
    markerfacecolor: Any | None = None
    markeredgecolor: Any | None = None
    markeredgewidth: float | None = None
    markerfacecoloralt: Any | None = None
    fillstyle: Literal["full", "left", "right", "bottom", "top", "none"] = "full"
    solid_capstyle: Literal["butt", "projecting", "round"] | None = None
    solid_joinstyle: Literal["miter", "round", "bevel"] | None = None
    dash_capstyle: Literal["butt", "projecting", "round"] | None = None
    dash_joinstyle: Literal["miter", "round", "bevel"] | None = None
    dashes: Sequence[float] | None = None
    gapcolor: Any | None = None

    # pcolormesh settings
    shading: Literal["flat", "nearest", "gouraud", "auto"] = "auto"
    cmap: Any | None = None
    vmin: float | None = None
    vmax: float | None = None
    norm: Any | None = None
    edgecolor: Any | Literal["face", "none"] | None = "none"
    rasterized: bool | None = True

    # plot_surface settings
    rcount: int | None = 50
    ccount: int | None = 50
    rstride: int | None = None
    cstride: int | None = None
    shade: bool | None = True
    lightsource: Any | None = None

    # PolyCollection
    sizes: Optional[Sequence[float]] = None
    closed: bool = True
    antialiaseds: Optional[Union[bool, Sequence[bool]]] = None
    facecolor: Optional[Union[str, Tuple, Sequence]] = None
    array: Optional[Any] = None
    clim: Optional[Tuple[float, float]] = None
    hatch: Optional[str] = None

    def to_plot(self) -> dict[str, Any]:
        kwargs = {
            "color": self.color,
            "alpha": self.alpha,
            "label": self.label,
            "visible": self.visible,
            "zorder": self.zorder,
            "linestyle": self.linestyle,
            "linewidth": self.linewidth,
            "antialiased": self.antialiased,
            "drawstyle": self.drawstyle,
            "marker": self.marker,
            "markersize": self.markersize,
            "markevery": self.markevery,
            "markerfacecolor": self.markerfacecolor,
            "markeredgecolor": self.markeredgecolor,
            "markeredgewidth": self.markeredgewidth,
            "markerfacecoloralt": self.markerfacecoloralt,
            "fillstyle": self.fillstyle,
            "solid_capstyle": self.solid_capstyle,
            "solid_joinstyle": self.solid_joinstyle,
            "dash_capstyle": self.dash_capstyle,
            "dash_joinstyle": self.dash_joinstyle,
            "dashes": self.dashes,
            "gapcolor": self.gapcolor,
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    def to_pcolormesh(self) -> dict[str, Any]:
        kwargs = {
            "shading": self.shading,
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "norm": self.norm,
            "edgecolors": self.edgecolor,
            "linewidths": self.linewidth,
            "antialiased": self.antialiased,
            "alpha": self.alpha,
            "rasterized": self.rasterized,
            "zorder": self.zorder,
            "visible": self.visible,
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    def to_plot_surface(self) -> dict[str, Any]:
        """Translates intent-driven semantic fields directly to ax.plot_surface()
        kwargs."""
        kwargs = {
            "color": self.color,
            "alpha": self.alpha,
            "zorder": self.zorder,
            "cmap": self.cmap,
            "vmin": self.vmin,
            "vmax": self.vmax,
            "norm": self.norm,
            "edgecolor": self.edgecolor,
            "linewidth": self.linewidth,
            "antialiased": self.antialiased,
            "rcount": self.rcount,
            "ccount": self.ccount,
            "rstride": self.rstride,
            "cstride": self.cstride,
            "shade": self.shade,
            "lightsource": self.lightsource,
        }
        return {k: v for k, v in kwargs.items() if v is not None}

    def to_poly(self) -> dict[str, Any]:
        """Translates intent-driven semantic fields directly to Poly3DCollection
        kwargs."""
        kwargs = {
            "facecolors": self.facecolor,
            "alpha": self.alpha,
            "edgecolors": self.edgecolor,
            "linewidths": self.linewidth,
            "linestyles": self.linestyle,
            "cmap": self.cmap,
            "norm": self.norm,
            "clim": self.clim,
            "zorder": self.zorder,
            "visible": self.visible,
            "antialiaseds": self.antialiased,
            "hatch": self.hatch,
        }
        return {k: v for k, v in kwargs.items() if v is not None}


@dataclass
class SavefigStyle(SerializableStyle):
    format: str | None = None
    transparent: bool | None = False
    dpi: float | Literal["figure"] | None = "figure"
    bbox_inches: Any | Literal["tight"] | None = "tight"
    pad_inches: float | None = 0.1
    facecolor: Any | Literal["auto"] | None = "auto"
    edgecolor: Any | Literal["auto"] | None = "auto"
    orientation: Literal["landscape", "portrait"] | None = "portrait"
    backend: str | None = None
    metadata: dict[str, str] | None = None
    pil_kwargs: dict[str, Any] | None = None


@dataclass
class MplLegendStyle(SerializableStyle):
    # Positioning
    loc: Optional[Union[str, int, Tuple[float, float]]] = None
    bbox_to_anchor: Optional[
        Union[Tuple[float, float], Tuple[float, float, float, float]]
    ] = None
    bbox_transform: Optional[Any] = None

    # Layout & Alignment
    ncols: Optional[int] = None  # Replaced 'ncol' in newer matplotlib versions
    reverse: Optional[bool] = None
    mode: Optional[str] = None
    alignment: Optional[str] = None

    # Text & Styling
    title: Optional[str] = None
    title_fontsize: Optional[Union[int, str]] = None
    title_fontproperties: Optional[Any] = None
    prop: Optional[Dict[str, Any]] = None  # Font dictionary
    fontsize: Optional[Union[int, str]] = None
    labelcolor: Optional[Union[str, List[str]]] = None

    # Markers & Lines
    numpoints: Optional[int] = None
    scatterpoints: Optional[int] = None
    scatteryoffsets: Optional[List[float]] = None
    markerscale: Optional[float] = None
    markerfirst: Optional[bool] = None

    # Frame/Box Styling
    frameon: Optional[bool] = None
    fancybox: Optional[bool] = None
    shadow: Optional[bool] = None
    framealpha: Optional[float] = None
    facecolor: Optional[str] = None
    edgecolor: Optional[str] = None

    # Spacing & Padding (Floats are fractions of fontsize)
    borderpad: Optional[float] = None
    labelspacing: Optional[float] = None
    handlelength: Optional[float] = None
    handleheight: Optional[float] = None
    handletextpad: Optional[float] = None
    borderaxespad: Optional[float] = None
    columnspacing: Optional[float] = None

    # Advanced
    handler_map: Optional[Dict[Any, Any]] = None
    draggable: Optional[bool] = None


# =========================================================
# PyVista settings


@dataclass
class PlotterStyle(SerializableStyle):
    """Configuration dataclass for initializing a standard pv.Plotter instance."""

    # --- Window Dimensions & Display ---
    window_size: Sequence[int] = (1024, 768)
    off_screen: bool = True
    notebook: bool | None = None

    # --- Layout & Borders ---
    shape: Union[Sequence[int], str] = (1, 1)
    border: bool | None = None
    border_color: Any = "k"  # ColorLike

    # --- Anti-Aliasing & Smoothing ---
    line_smoothing: bool = False
    polygon_smoothing: bool = False

    # --- Lighting & Effects ---
    lighting: str = "light kit"  # 'light kit', 'three lights', or 'none'

    # --- Global Controls ---
    theme: Any | None = None  # pyvista.plotting.themes.Theme
    image_scale: int = 1


@dataclass
class AddMeshStyle(SerializableStyle):
    """Configuration dataclass for pv.Plotter.add_mesh() arguments."""

    # --- Basic Appearance ---
    color: Any | None = None  # ColorLike
    style: Any | None = None  # StyleOptions
    opacity: float | Sequence[float] | Any | None = None  # OpacityOptions
    point_size: float | None = None
    line_width: float | None = None
    show_edges: bool | None = None
    edge_color: Any | None = None  # ColorLike
    edge_opacity: float | None = None
    show_vertices: bool | None = None

    # --- Scalars & Colormaps ---
    scalars: str | Any | None = None  # NumpyArray[float]
    clim: Sequence[float] | None = None
    cmap: Any | None = None  # ColormapOptions | LookupTable
    n_colors: int = 256
    nan_color: Any | None = None  # ColorLike
    nan_opacity: float = 1.0
    flip_scalars: bool = False
    interpolate_before_map: bool | None = None
    rgb: bool | None = None
    categories: bool = False
    log_scale: bool = False
    multi_colors: bool = False
    below_color: Any | None = None  # ColorLike
    above_color: Any | None = None  # ColorLike
    annotations: dict[float, str] | None = None

    # --- Scalar Bar ---
    show_scalar_bar: bool | None = None
    scalar_bar_args: Any | None = None  # ScalarBarArgs

    # --- Labels & Identification ---
    name: str | None = None
    label: str | None = None

    # --- Rendering & Shading ---
    render_points_as_spheres: bool | None = None
    point_shape: Any | str | None = None  # PointSpriteShape
    render_lines_as_tubes: bool | None = None
    smooth_shading: bool | None = None
    split_sharp_edges: bool | None = None
    use_transparency: bool = False
    culling: Any | bool | None = None  # CullingOptions
    silhouette: Any | bool | None = None  # SilhouetteArgs
    texture: Any | None = None  # Texture | NumpyArray[float]
    backface_params: Any | None = None  # BackfaceArgs | Property

    # --- Lighting & PBR ---
    lighting: bool | None = None
    ambient: float | None = None
    diffuse: float | None = None
    specular: float | None = None
    specular_power: float | None = None
    pbr: bool | None = None
    metallic: float | None = None
    roughness: float | None = None
    emissive: bool | None = None

    # --- Control & Environment ---
    pickable: bool = True
    preference: str = "point"  # PointLiteral | CellLiteral
    render: bool = True
    reset_camera: bool | None = None
    user_matrix: Any | None = None  # TransformLike
    component: int | None = None
    copy_mesh: bool = False
    remove_existing_actor: bool | None = None


@dataclass
class AxesStyle(SerializableStyle):
    """Configuration dataclass for pv.Plotter.add_axes() arguments."""

    # --- Structural Layout ---
    line_width: float = 2.0
    cone_radius: float = 0.4
    shaft_length: float = 0.8
    tip_length: float = 0.2

    # --- Label and Typography Settings ---
    ambient: float = 0.5
    label_size: Sequence[float] = (0.25, 0.1)
    label_color: Any | None = None  # ColorLike
    labels_off: bool = False

    # --- Interactive Controls ---
    box: bool = False
    box_args: dict | None = None
    viewport: Sequence[float] = (0.0, 0.0, 0.2, 0.2)


@dataclass
class ScreenshotStyle(SerializableStyle):
    """Configuration dataclass for pv.Plotter.screenshot() arguments."""

    # --- Output File Properties ---
    filename: Any | None = None  # str, pathlib.Path, or file-like object
    transparent_background: bool = False
    return_img: bool = True  # Default to True so our cropping machinery can process it

    # --- Canvas Sampling & Sizing ---
    window_size: Sequence[int] | None = None
    scale: int = 1


@dataclass
class PvLegendStyle(SerializableStyle):
    """Configuration dataclass for pv.Plotter.add_legend() arguments."""

    # --- Structural Entries ---
    labels: Sequence[Any] | None = None  # Can be passed dynamically during runtime
    bcolor: Any | None = None  # ColorLike (Background color)
    border: bool = False
    size: Sequence[float] = (0.2, 0.2)
    name: str | None = None

    # --- Text & Typography ---
    font_family: str | None = None
    face: str | None = None  # 'rectangle', 'line', 'circle', or None

    # --- Location Positioning ---
    # Can be a string like 'upper right', 'lower left', or relative coordinate positions
    loc: str | Sequence[float] | None = "upper right"


# =========================================================
# ttnte style settings


@dataclass
class MplPatchStyle:
    colorby: Literal["material", "patch"] = "material"
    crop_padding: int = 10
    normal: Optional[list | tuple] = None
    origin: tuple = (0, 0, 0)
    xlabel: str = "$x$"
    ylabel: str = "$y$"
    zlabel: str = "$z$"
    aspect: Literal["auto", "equal"] = "equal"
    control_offset3d: float = 0.00005

    # Window settings
    figure: FigureStyle = field(default_factory=lambda: FigureStyle(layout="tight"))
    subplot: SubplotStyle = field(default_factory=lambda: SubplotStyle(aspect="equal"))
    savefig: SavefigStyle = field(default_factory=lambda: SavefigStyle(dpi=300))
    legend: MplLegendStyle = field(
        default_factory=lambda: MplLegendStyle(
            loc="upper right", bbox_to_anchor=(1.05, 1)
        )
    )

    # Mesh settings
    mesh: SurfaceStyle = field(
        default_factory=lambda: SurfaceStyle(
            color="#007ACC",
            facecolor="#007ACC",
            cmap=ListedColormap(["#007ACC"]),
            shading="flat",
            shade=True,
            antialiased=False,
            linewidth=0.15,
            edgecolor="none",
            zorder=2,
        )
    )
    control_points: ScatterStyle = field(
        default_factory=lambda: ScatterStyle(c="black", edgecolors="none", zorder=3)
    )
    control_net: PlotStyle = field(
        default_factory=lambda: PlotStyle(
            color="black", linewidth=2, alpha=0.6, zorder=3
        )
    )
    boundary: PlotStyle = field(
        default_factory=lambda: PlotStyle(color="black", linewidth=3, zorder=3)
    )


@dataclass
class PvPatchStyle:
    colorby: Literal["material", "patch"] = "material"
    crop_padding: int = 10
    normal: Optional[list | tuple] = None
    origin: tuple = (0, 0, 0)
    n_values: int = 10

    # Window settings
    window: PlotterStyle = field(default_factory=lambda: PlotterStyle(off_screen=True))
    axes: AxesStyle = field(default_factory=AxesStyle)
    screenshot: ScreenshotStyle = field(default_factory=ScreenshotStyle)
    legend: PvLegendStyle = field(default_factory=PvLegendStyle)

    # Add mesh settings
    mesh: AddMeshStyle = field(
        default_factory=lambda: AddMeshStyle(
            color="#007ACC", point_size=10, opacity=1.0, render_points_as_spheres=True
        )
    )
    control_points: AddMeshStyle = field(
        default_factory=lambda: AddMeshStyle(
            color="black", point_size=10, opacity=1.0, render_points_as_spheres=True
        )
    )
    control_net: AddMeshStyle = field(
        default_factory=lambda: AddMeshStyle(
            color="black", style="wireframe", line_width=2, opacity=0.6
        )
    )
    boundary: AddMeshStyle = field(
        default_factory=lambda: AddMeshStyle(color="black", line_width=3, opacity=1.0)
    )


def get_patch_style(backend: Literal["matplotlib", "pyvista"]):
    """Get the plotting style settings for a given backend.

    Parameters
    ----------
    backend: "matplotlib" or "pyvista"
        The plotting backend.

    Returns
    -------
    style: ttnte.visualization.style.MplPatchStyle or ttnte.visualization.style.PvPatchStyle
        The plotter settings for that backend.
    """
    return MplPatchStyle() if backend == "matplotlib" else PvPatchStyle()
