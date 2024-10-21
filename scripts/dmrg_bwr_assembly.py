import time

import cupy as cu

import tt_nte.experimental.solvers as exp
from tt_nte.benchmarks import bwr_assembly
from tt_nte.methods import DiscreteOrdinates

with cu.cuda.Device(1):
    # Get benchmark XSs and geometry
    xs_server, geometry, ordinates = bwr_assembly(64)

    # Build TT operators
    start = time.time()
    SN = DiscreteOrdinates(
        xs_server=xs_server,
        geometry=geometry,
        num_ordinates=4,
        tt_fmt="qtt",
        qtt_threshold=1e-8,
        xs_threshold=1e-5,
    )
    print(f"Setup time = {time.time() - start}")

    SN.H.ortho(1e-8)
    SN.F.ortho(1e-8)
    SN.S.ortho(1e-8)

    size = 0
    for tn in [SN.H, SN.F, SN.S]:
        size += sum([core.nbytes for core in tn.cores])

    print(f"Size of operators = {size * 1e-6} MB")

    print(f"H = {SN.H}")
    print(f"F = {SN.F}")
    print(f"S = {SN.S}")

    # Initialize power iteration solver
    solver = exp.Power(
        SN.H.to_quimb() - SN.S.to_quimb(),
        SN.F.to_quimb(),
        verbose=True,
        print_every=1,
    )

    # Run solver
    linear_solver_opts = {
        "tols": 1e-4,
        "max_ranks": [64],
        "sweep_sequence": "RL",
        "max_sweeps": 1,
    }

    start = time.time()
    k, psi = solver.solve(
        tol=1e-2,
        linear_solver=exp.ALS(verbose=True, use_gpu=False),
        rank=2,
        **linear_solver_opts,
    )
    print(f"Solve time = {time.time() - start}, k = {k}")
