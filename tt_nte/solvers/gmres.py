import numpy as np
import tt

from tt_nte.solvers._base import Solver
from tt_nte.tensor_train import TensorTrain


class GMRES(Solver):
    def __init__(self, method, verbose=False):
        """
        Eigenvalue solver using GMRES to solve Ax=b.
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

    def power(
        self,
        ranks=None,
        tol=1e-6,
        max_iter=100,
        gmres_tol=1e-6,
        gmres_max_iter=100,
        m=20,
    ):
        """
        Power iteration using tt.solvers.GMRES(). ``ranks`` controls the ranks
        of the solution. ``gmres_tol`` controls the tolerance of the solution
        produced by GMRES. ``gmres_max_iter`` controls the maximum number of
        iterations of GMRES. ``m`` controls the number of GMRES iterations
        before restart.
        """
        # Setup power iteration
        psi0, k0 = self._setup(ranks)

        def solver(A, B, x0):
            # Matrix-vector product with A
            def matvec(x, eps):
                return tt.matvec(A, x).round(eps)

            psi, _ = tt.solvers.GMRES(
                A=matvec,
                u_0=x0,
                b=tt.matvec(B, x0),
                eps=gmres_tol,
                maxit=gmres_max_iter,
                m=m,
                verbose=self._verbose,
            )
            return psi

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
