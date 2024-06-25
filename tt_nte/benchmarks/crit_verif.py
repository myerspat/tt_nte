"""
crit_verif.py

All cases below are given in https://www.sciencedirect.com/science/article/pii/S0149197002000987.
All cases are given in a critical configuration.
"""

import numpy as np

from tt_nte.geometry import Geometry, Region
from tt_nte.xs import Material, Nuclide, Server


def pu_brick(num_nodes):
    """
    Pu-239 1D slab problem taken from the Criticality Verification Benchmark Suite.
    The width of the slab is 3.707444 cm with vacuum boundary conditions on either
    side.
    """
    # Nuclides
    pu239 = Nuclide(Z=94, A=239)

    # Materials
    fuel = Material({pu239: 1.0})

    # Cross section data
    xs = {
        "chi": np.array([1.0]),
        fuel: {
            "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
            "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
            "total": np.array([0.32640]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab Geometry
    fuel_region = Region(fuel, 3.707444, num_nodes - 1)
    geometry = Geometry([fuel_region], "vacuum", "vacuum")

    return xs_server, geometry


def pu_brick_multi_region(num_nodes, num_regions):
    """
    Same as pu_brick() but split into several Regions. Regions are
    linearly spaced.
    """
    # Nuclides
    pu239 = Nuclide(Z=94, A=239)

    # Materials
    fuel = Material({pu239: 1.0})

    # Cross section data
    xs = {
        "chi": np.array([1.0]),
        fuel: {
            "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
            "scatter_gtg": np.array([[[0.225216]]]),  # 1/cm
            "total": np.array([0.32640]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab Geometry
    regions = []
    num_nodes = np.linspace(0, num_nodes, num_regions + 1, dtype=int)
    thicknesses = np.linspace(0, 3.707444, num_regions + 1)

    num_nodes[-1] -= 1
    num_nodes = num_nodes[1:] - num_nodes[:-1]
    thicknesses = thicknesses[1:] - thicknesses[:-1]

    for n, t in zip(num_nodes, thicknesses):
        regions.append(Region(fuel, t, n))

    geometry = Geometry(regions, "vacuum", "vacuum")

    return xs_server, geometry


def research_reactor_multi_region(num_nodes, right_bc="reflective"):
    """
    Multi-region case with multiplying medium (fuel) and non-multiplying medium
    (mod). This case is also multi-group.
    """
    # Nuclides
    u235 = Nuclide(Z=92, A=235)
    h1 = Nuclide(Z=1, A=1)
    o16 = Nuclide(Z=8, A=16)

    # Materials
    fuel = Material({u235: 1 / 1501, h1: 1000 / 1501, o16: 500 / 1501})
    mod = Material({h1: 2 / 3, o16: 1 / 3})

    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        fuel: {
            "nu_fission": 2.5 * np.array([0.000836, 0.029564]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83892, 0.000767],
                        [0.04635, 2.918300],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88721, 2.9727]),  # 1/cm
        },
        mod: {
            "nu_fission": np.zeros(2),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.83975, 0.000336],
                        [0.04749, 2.967600],
                    ],
                ]
            ),  # 1/cm
            "total": np.array([0.88798, 2.9865]),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab Geometry
    regions = [Region(mod, 1.126151, num_nodes[0])]

    if right_bc == "reflective":
        regions.append(Region(fuel, 6.696802, num_nodes[1] - 1))
    else:
        regions.append(Region(fuel, 2 * 6.696802, num_nodes[1]))
        regions.append(Region(mod, 1.126151, num_nodes[2] - 1))

    geometry = Geometry(regions, "vacuum", right_bc)

    return xs_server, geometry


def research_reactor_anisotropic(num_nodes, right_bc="reflective"):
    """
    Two-group uranium research reactor (linearly-anisotropic), Tables
    49 and 50.
    """
    # Nuclides
    u235 = Nuclide(Z=92, A=235)
    h1 = Nuclide(Z=1, A=1)
    o16 = Nuclide(Z=8, A=16)

    # Materials
    fuel = Material({u235: 1 / 1501, h1: 1000 / 1501, o16: 500 / 1501})

    # Cross section data
    xs = {
        "chi": np.array([1.0, 0.0]),
        fuel: {
            "total": np.array([0.65696, 2.52025]),  # 1/cm
            "nu_fission": 2.5 * np.array([0.0010484, 0.050632]),  # 1/cm
            "scatter_gtg": np.array(
                [
                    [
                        [0.625680, 0.00000],
                        [0.029227, 2.44383],
                    ],
                    [
                        [0.2745900, 0.00000],
                        [0.0075737, 0.83318],
                    ],
                ]
            ),  # 1/cm
        },
    }
    xs_server = Server(xs)

    # Slab Geometry
    regions = []

    if right_bc == "reflective":
        regions.append(Region(fuel, 9.4959, num_nodes - 1))
    else:
        regions.append(Region(fuel, 2 * 9.4959, num_nodes - 1))

    geometry = Geometry(regions, "vacuum", right_bc)

    return xs_server, geometry
