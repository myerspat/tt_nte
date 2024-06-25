import numpy as np

import tt_nte.benchmarks as benchmarks
import tt_nte.solvers as solvers
from tt_nte.methods import DiscreteOrdinates


def test_single_media():
    # Set numpy random seed
    np.random.seed(42)

    # Get single-media problem geometry and XS server
    server, geometry = benchmarks.pu_brick(1024)

    # ----------------------------------------------------------------
    # SN in TT format

    # Initialize SN solver
    SN = DiscreteOrdinates(
        xs_server=server,
        geometry=geometry,
        num_ordinates=2,
        tt_fmt="tt",
    )

    # Assertions
    assert SN.H.matricize().shape == (1024 * 2, 1024 * 2)
    assert SN.S.matricize().shape == (1024 * 2, 1024 * 2)
    assert SN.F.matricize().shape == (1024 * 2, 1024 * 2)

    # Ordinates and solvers to test
    num_ordinates = [2, 8]
    tt_solvers = [solvers.Matrix, solvers.ALS, solvers.AMEn]

    # Expected solutions
    expected_k = [0.80418, 0.99176]
    expected_psi = []

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i])

        # Run matrix generalized eigenvalue solver
        solver = solvers.Matrix(method=SN, verbose=True)
        solver.ges()

        # Assertions
        assert abs(expected_k[i] - solver.k) < 0.00005
        assert solver.psi.shape == (1024 * num_ordinates[i],)

        # Save psi
        expected_psi.append(solver.psi)

        for solver in tt_solvers:
            # Initialize solver
            solver = solver(method=SN, verbose=True)

            # Run power iteration
            solver.power()

            # Assertions
            assert abs(expected_k[i] - solver.k) < 0.00005
            assert (
                solver.psi.shape
                if solver.__class__.__name__ == "Matrix"
                else solver.psi.matricize().shape == (1024 * num_ordinates[i],)
            )
            assert (
                np.linalg.norm(
                    expected_psi[i] - solver.psi
                    if solver.__class__.__name__ == "Matrix"
                    else expected_psi[i] - solver.psi.matricize()
                )
                < 0.002
            )

    # ----------------------------------------------------------------
    # SN in QTT format
    qtt_solvers = [solvers.MALS, solvers.GMRES]
    qtt_solver_configs = [{"max_rank": 20}, {}]

    for i in range(len(num_ordinates)):
        # Change number of ordinates in SN
        SN.update_settings(num_ordinates=num_ordinates[i], tt_fmt="qtt")

        for j in range(len(qtt_solvers)):
            # Initialize solver
            solver = qtt_solvers[j](method=SN, verbose=True, **qtt_solver_configs[j])

            # Run power iteration
            solver.power(ranks=4)

            # Assertions
            assert abs(expected_k[i] - solver.k) < 0.00005
            assert (
                solver.psi.shape
                if solver.__class__.__name__ == "Matrix"
                else solver.psi.matricize().shape == (1024 * num_ordinates[i],)
            )
            assert (
                np.linalg.norm(
                    expected_psi[i] - solver.psi
                    if solver.__class__.__name__ == "Matrix"
                    else expected_psi[i] - solver.psi.matricize()
                )
                < 0.002
            )
