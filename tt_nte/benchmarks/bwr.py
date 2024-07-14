from pathlib import Path

import gmsh
import numpy as np
import pandas as pd

from tt_nte.geometry import Geometry
from tt_nte.xs import Server


def bwr(xy_num_nodes, control_rod=True, reflected=False, return_model=False):
    pitch = 1.3 * 1.2

    # Get number of nodes per cell
    num_elements_per_cell = np.ones(23, dtype=int) * int((xy_num_nodes / 23)) + 1
    num_elements_per_cell[: int(xy_num_nodes % 23 - 1)] += 1
    num_elements_per_cell = num_elements_per_cell[::-1]

    # Read XSs
    data = pd.read_csv(
        (
            Path(__file__).parent
            / "supporting/bwr/"
            / ("xs_control.dat" if control_rod else "xs.dat")
        ),
        delim_whitespace=True,
        header=None,
    ).values
    ordinates = pd.read_csv(
        (Path(__file__).parent / "./supporting/bwr/quadrature.dat"),
        delim_whitespace=True,
    ).values[:, [4, 1, 2]]

    xs = {
        "chi": np.array(
            [
                1.0588870e-11,
                2.4979600e-11,
                1.9664890e-10,
                4.6093200e-10,
                8.2317503e-7,
                3.9163700e-4,
                0.13135319,
                0.86825800,
            ]
        )
    }
    num_elements = int(np.max(data[:, 0]))
    num_groups = int(np.max(data[:, 1]))
    num_moments = int((data.shape[1] - 5) / num_groups)

    for i in range(num_elements):
        idx = np.argwhere(data[:, 0] == (i + 1)).flatten()
        scatter_gtg = np.zeros((num_moments, num_groups, num_groups))

        for l in range(num_moments):
            scatter_gtg[l, :, :] = data[
                idx, (5 + l * num_groups) : (5 + l * num_groups + num_groups)
            ]

        xs[str(i)] = {
            "total": data[idx, 2],
            "nu_fission": data[idx, 4],
            "scatter_gtg": scatter_gtg,
        }

    # Define assembly
    assembly = np.arange(num_elements).reshape(
        (int(np.sqrt(num_elements)), int(np.sqrt(num_elements)))
    )[:, ::-1]

    # Reflect assembly
    if reflected:
        num_elements_per_cell = np.concatenate(
            (num_elements_per_cell[::-1], num_elements_per_cell)
        )
        assembly = np.block(
            [[assembly, assembly[:, ::-1]], [assembly[::-1, :], assembly[::-1, ::-1]]]
        )

    # Initialize gmsh
    gmsh.initialize()
    gmsh.model.add("BWR Core")
    gmsh.option.setNumber("General.Terminal", 0)

    # Create points between half unit cells
    points = np.zeros((assembly.shape[0] + 1, assembly.shape[1] + 1), dtype=int)
    for i in range(assembly.shape[0] + 1):
        for j in range(assembly.shape[1] + 1):
            points[i, j] = gmsh.model.geo.add_point(
                j * pitch / 2, i * pitch / 2, 0, 0.1, i * (assembly.shape[0] + 1) + j
            )

    # Connect points
    x_lines = np.zeros((assembly.shape[0], assembly.shape[1] + 1), dtype=int)
    y_lines = np.zeros((assembly.shape[0] + 1, assembly.shape[1]), dtype=int)

    for i in range(y_lines.shape[0]):
        for j in range(y_lines.shape[1]):
            y_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i, j + 1])

    for i in range(x_lines.shape[0]):
        for j in range(x_lines.shape[1]):
            x_lines[i, j] = gmsh.model.geo.add_line(points[i, j], points[i + 1, j])

    # Add boundary conditions
    reflective = y_lines[-1, :].tolist() + x_lines[:, -1].tolist()
    vacuum = y_lines[0, :].tolist() + x_lines[:, 0].tolist()

    if reflected:
        gmsh.model.add_physical_group(1, reflective + vacuum, name="vacuum")
    else:
        gmsh.model.add_physical_group(1, reflective, name="reflective")
        gmsh.model.add_physical_group(1, vacuum, name="vacuum")

    # Create surfaces
    faces = np.zeros(assembly.shape, dtype=int)
    surfaces = np.zeros(assembly.shape, dtype=int)

    for i in range(assembly.shape[0]):
        for j in range(assembly.shape[1]):
            faces[i, j] = gmsh.model.geo.add_curve_loop(
                [
                    y_lines[i, j],
                    x_lines[i, j + 1],
                    -y_lines[i + 1, j],
                    -x_lines[i, j],
                ]
            )
            surfaces[i, j] = gmsh.model.geo.add_plane_surface([faces[i, j]])

    # Assign material regions
    for i in range(num_elements):
        idxs = np.argwhere(assembly == i)
        gmsh.model.add_physical_group(
            2, faces[idxs[:, 0], idxs[:, 1]].flatten().tolist(), name=str(i)
        )

    # Sync gmsh model
    gmsh.model.geo.synchronize()

    # Create structured mesh
    for i in range(assembly.shape[0]):
        for j in range(assembly.shape[1]):
            # Transfinite curves
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                y_lines[i + 1, j], num_elements_per_cell[j]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j], num_elements_per_cell[i]
            )
            gmsh.model.mesh.set_transfinite_curve(
                x_lines[i, j + 1], num_elements_per_cell[i]
            )

            # Transfinite surface
            gmsh.model.mesh.set_transfinite_surface(
                surfaces[i, j],
                cornerTags=points[i : i + 2, j : j + 2].flatten().tolist(),
            )

    # Generate mesh and recombine to get 4-point quadrangle structured mesh
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.recombine()

    if return_model:
        return Server(xs), Geometry(gmsh.model), ordinates, gmsh.model
    else:
        geometry = Geometry(gmsh.model)
        gmsh.finalize()

        return Server(xs), geometry, ordinates
