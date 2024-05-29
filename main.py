"""
main.py

These scripts include an attempt to recreate the results shown in 
https://www.sciencedirect.com/science/article/pii/S002199912400192X
for the one-dimensional case study.
"""

# import sys

import numpy as np

from discrete_ordinates import DiscreteOrdinates
from geometry import Geometry
from material import Material
from nuclide import Nuclide
from xs_server import XSServer

import matplotlib.pyplot as plt
import time

# np.set_printoptions(threshold=sys.maxsize)


# Nuclides
pu239 = Nuclide(Z=94, A=239)

# Materials
fissile_brick = Material({pu239: 1.0})

# Cross section data
one_group_xs = {
    "num_groups": 2,
    "chi": np.array([1.0]),
    fissile_brick: {
        "nu_fission": np.array([3.24 * 0.081600]),  # 1/cm
        "scatter_gtg": np.array([[0.225216]]),  # 1/cm
        "total": np.array([0.32640]),  # 1/cm
    },
}

two_group_xs = {
    "num_groups": 2,
    "chi": np.array([0.575, 0.425]),
    fissile_brick: {
        "nu_fission": np.array([3.10 * 0.0936, 2.93 * 0.08544]),  # 1/cm
        "scatter_gtg": np.array([[0.0792, 0.0],[0.0432, 0.23616]]),  # 1/cm
        "total": np.array([0.2208, 0.3360]),  # 1/cm
    },
}
xs_server = XSServer(two_group_xs)

# Slab Geometry
#thickness = 3.707444  # cm, 1-D 1-group Pu slab critical case
'''thickness = 2 * 1.795602 # cm, 1-D 2-group Pu slab critical case
num_nodes = 127
geometry = Geometry({fissile_brick: {"num_nodes": num_nodes, "thickness": thickness}})

# Solver
tol = 1e-4
num_ordinates = 16

tt = DiscreteOrdinates(
    xs_server=xs_server, geometry=geometry, num_ordinates=num_ordinates, tol=tol, bc = "vacuum"
)

# k = tt.solve_gesv()

k, psi = tt.solve_matrix_power()
print(k)
print(psi)'''

# Slab Geometry
thickness = 2 * 1.795602  # cm
num_nodes = 511
geometry = Geometry({fissile_brick: {"num_nodes": num_nodes, "thickness": thickness}})
tol = 1e-6

num_ordinates_list = [2, 4, 8, 16]

SN = DiscreteOrdinates(
    xs_server=xs_server,
    geometry=geometry,
    num_ordinates=num_ordinates_list[0],
    tol=tol,
)

#solvers = 

ks = {"ISFM": [], "TT/ALS": []}
exec_times = {"ISFM": [], "TT/ALS": []}

expected_k = 1
psi = None

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
ax1.axhline(expected_k, color="k", ls="--", label="Ground Truth")

for method in ks.keys():
    for num_ordinates in num_ordinates_list:
        # Initialize solver
        SN = DiscreteOrdinates(
            xs_server=xs_server, 
            geometry=geometry, 
            num_ordinates=num_ordinates, 
            tol=tol,
        )
    
        # Run and time matrix power iteration
        k = None
        start = None
        stop = None
        if method == "ISFM":
            start = time.time()
            k, psi = SN.solve_matrix_power()
            stop = time.time()
        elif method == "TT/ALS":
            start = time.time()
            k, psi = SN.solve_TT_power(method="als")
            stop = time.time()
    
        ks[method].append(k)
        exec_times[method].append(stop - start)
        
    ax1.scatter(num_ordinates_list, ks[method], label=method)
    ax2.scatter(num_ordinates_list, np.abs(expected_k - np.array(ks[method])) * 1e5, label=method)
    ax3.scatter(num_ordinates_list, exec_times[method], label=method)

ax1.set(xlabel="Ordinate $n$", ylabel="Eigenvalue $k$")
ax2.set(xlabel="Ordinate $n$", ylabel="Eigenvalue Error (pcm)")
ax3.set(xlabel="Ordinate $n$", ylabel="Execution Time (s)")
ax2.set_yscale("log")
plt.legend()
plt.show()

# tt.solve_TT()
