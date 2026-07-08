import multiprocessing

import torch
from igakit.cad import circle, line, ruled, refine, coons, extrude

from ttnte import mpi_context
from ttnte.visualization.style import get_patch_style
from ttnte.xs.benchmarks import c5g7
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
)

if __name__ == "__main__":
    # =========================================================================
    # Code setup
    dtype = torch.float64
    cpu = torch.device("cpu")

    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(multiprocessing.cpu_count() // mpi_context.world_size, 4)
    torch.set_num_threads(num_threads_per_rank)

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # =========================================================================
    # Macroscopic cross section information
    fills, xs_server = c5g7(device=cpu, dtype=dtype)

    # =========================================================================
    # Build multi-patch NURBS geometry

    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    z = 1  # cm

    c0 = circle(radius, angle=torch.pi / 4)
    c1 = circle(radius, angle=-torch.pi / 4).rotate(torch.pi / 2)
    l0 = line((0, 0), (radius, 0))
    l1 = line((0, 0), (0, radius))

    s0 = coons([[l1, c0], [l0, c1]])
    c0 = s0.boundary(0, 1)
    c1 = s0.boundary(1, 1)

    # Create water patch
    l2 = line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l3 = line(p0=(0, pitch / 2), p1=(pitch / 2, pitch / 2))

    # Create NURBS surfaces
    fuel = [extrude(s0, z, 2)]
    water = [extrude(ruled(c0, l2), z, 2), extrude(ruled(c1, l3), z, 2)]

    # Create the mesh
    mesh = IGAMesh(mpi_context)

    # Add all the patches to the mesh
    for fill, surface in zip([fills[0], fills[-2], fills[-2]], fuel + water):
        mesh.add_block(
            Patch.from_igakit(
                refine(surface, 10, 2), device=cpu, dtype=dtype, fill=fill
            )
        )

    # Connect patches
    mesh.connect()

    # Set the boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, x_max=True, y_max=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )
    mesh.set_axis_aligned_conditions(
        BCPlane(z_min=True, z_max=True),
        BoundaryType.PERIODIC,
        tol=1e-6,
    )
    mesh.finalize()

    # =========================================================================
    # Plotting with PyVista

    cmap = {
        fills[0].to_string(): "maroon",
        fills[-2].to_string(): "cornflowerblue",
    }

    # Plot the 3-D geometry
    backend = "pyvista"
    style = get_patch_style(backend)
    style.legend.size = (0.15, 0.15)
    style.screenshot.scale = 1.5
    style.mesh.cmap = cmap
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        backend=backend,
        filename="figs/c5g7_pincell_pv3d.png",
        style=style,
    )

    # Plot a 2-D slice defined by the intersection of a plane with the 3-D model
    style.legend.size = (0.2, 0.2)
    style.normal = (0, 0, 1)
    style.origin = (0, 0, 0.5)
    mesh.plot(
        resolution=25,
        show_boundary=True,
        backend=backend,
        filename="figs/c5g7_pincell_pv2d.png",
        style=style,
    )

    # =========================================================================
    # Plotting with matplotlib

    # Plot the 3-D geometry
    backend = "matplotlib"
    style = get_patch_style(backend)
    style.mesh.edgecolor = "black"
    style.mesh.antialiased = True
    style.mesh.cmap = cmap
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        backend=backend,
        filename="figs/c5g7_pincell_mpl3d.png",
        style=style,
    )

    # Plot a 2-D slice defined by the intersection of a plane with the 3-D model
    style.mesh.edgecolor = "none"
    style.mesh.antialiased = False
    style.normal = (0, 0, 1)
    style.origin = (0, 0, 0.5)
    mesh.plot(
        resolution=25,
        show_boundary=True,
        backend=backend,
        filename="figs/c5g7_pincell_mpl2d.png",
        style=style,
    )

    # =========================================================================
    # Check the boundary conditions for each patch

    fuel = mesh.blocks[0]
    assert fuel.get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert fuel.get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert fuel.get_boundary_info(1, False).type == BoundaryType.REFLECTIVE
    assert fuel.get_boundary_info(1, True).type == BoundaryType.INTERNAL
    assert fuel.get_boundary_info(2, False).type == BoundaryType.PERIODIC
    assert fuel.get_boundary_info(2, True).type == BoundaryType.PERIODIC

    water = mesh.blocks[1:]
    assert water[0].get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert water[0].get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert water[0].get_boundary_info(1, False).type == BoundaryType.INTERNAL
    assert water[0].get_boundary_info(1, True).type == BoundaryType.REFLECTIVE
    assert water[0].get_boundary_info(2, False).type == BoundaryType.PERIODIC
    assert water[0].get_boundary_info(2, True).type == BoundaryType.PERIODIC
    assert water[1].get_boundary_info(0, False).type == BoundaryType.REFLECTIVE
    assert water[1].get_boundary_info(0, True).type == BoundaryType.INTERNAL
    assert water[1].get_boundary_info(1, False).type == BoundaryType.INTERNAL
    assert water[1].get_boundary_info(1, True).type == BoundaryType.REFLECTIVE
    assert water[1].get_boundary_info(2, False).type == BoundaryType.PERIODIC
    assert water[1].get_boundary_info(2, True).type == BoundaryType.PERIODIC
