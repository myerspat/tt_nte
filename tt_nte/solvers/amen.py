import numpy as np
import tt
from tt.amen import amen_solve

from tt_nte.solvers._base import Solver
from tt_nte.tensor_train import TensorTrain


class AMEn(Solver):
    def __init__(self, method, verbose=False):
        """
        Eigenvalue solver using the Alternating Minimal Energy Method to solve
        Ax=b.
        """
        # Initialize base class
        super().__init__(
            method.H.train("ttpy"),
            method.S.train("ttpy"),
            method.F.train("ttpy"),
            verbose,
        )

    # =======================================================================
    # Methods

    def power(self, ranks=None, tol=1e-6, max_iter=100, amen_tol=1e-6):
        """
        Power iteration using tt.amen.amen_solve(). ``amen_tol`` controls the
        tolerance of the solution produced by tt.amen.amen_solve(). ``ranks``
        controls the ranks of the cores in the solution.
        """
        # Setup power iteration
        psi0, k0 = self._setup(ranks)

        def solver(A, B, x0):
            return amen_solve(A, tt.matvec(B, x0), x0, amen_tol, verb=self._verbose)

        super()._power(
            psi0=psi0,
            k0=k0,
            solver=solver,
            norm=lambda x, p: TensorTrain(tt.vector.to_list(x)).norm(p),
            matvec=tt.matvec,
            tol=tol,
            max_iter=max_iter,
        )

    def _setup(self, ranks):
        # Get maximum ranks for each core
        ranks = (
            ((np.array([self._H.r, self._S.r, self._F.r])).max(axis=0).tolist())
            if ranks == None
            else ranks
        )

        # Initial guess for psi and k
        psi0 = tt.vector.from_list(
            TensorTrain.rand(self._H.n, [1] * self._H.d, ranks).cores
        )
        k0 = np.random.rand(1)[0]

        return psi0, k0

    # =======================================================================
    # Getters

    @property
    def psi(self):
        return TensorTrain(tt.vector.to_list(super().psi))
