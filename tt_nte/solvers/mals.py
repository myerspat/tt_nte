import copy

import numpy as np
from scikit_tt.solvers.sle import mals

from tt_nte.solvers.als import ALS


class MALS(ALS):
    def __init__(self, method, verbose=False, threshold=1e-8, max_rank=None):
        """
        Eigenvalue solver using Modified ALS (MALS) to solve Ax=b. ``max_rank``
        controls the maximum allowed rank for the cores of the solution.
        ``threshold`` controls the SVD threshold.
        """
        # Initialize base class
        super().__init__(method, verbose)

        self._threshold = threshold
        self._max_rank = max_rank

        # Take maximum rank of operators if not given
        if self._max_rank is None:
            self._max_rank = np.array(
                [self._H.ranks, self._S.ranks, self._F.ranks], dtype=int
            ).max()

    # =======================================================================
    # Methods

    def ges(self, ranks=None, repeats=10):
        raise NotImplementedError()

    def power(self, ranks=None, tol=1e-6, max_iter=100):
        """
        Power iteration using scikit_tt.solvers.sle.mals().
        """
        # Get minimum rank for start if None is given
        if ranks is None:
            ranks = np.array(
                [self._H.ranks[1:-1], self._S.ranks[1:-1], self._F.ranks[1:-1]],
                dtype=int,
            ).min()

        # If verbose we temporarily disable it for 5 ALS iterations
        verbose = copy.deepcopy(self._verbose)
        self._verbose = False

        # Run initial 5 ALS iterations
        super().power(ranks=ranks, tol=tol, max_iter=5)

        # Reset verbose
        self._verbose = verbose

        # scikit_tt.solvers.mals kwargs
        kwargs = {"threshold": self._threshold, "max_rank": ranks}

        # Compute polynomial coefs for ranks
        self._poly = np.polyfit(
            np.log([self.error, tol * 100]), [ranks, self._max_rank], 1
        )

        # MALS function
        def solver(A, B, x0):
            return mals(A, x0, B.dot(x0), **kwargs)

        # Update maximum rank as error goes to tol
        def rank_update(sol):
            kwargs["max_rank"] = np.round(
                self._poly[0] * np.log(sol.error) + self._poly[1]
            ).astype(int)

        # Run power iteration
        super()._power(
            psi0=copy.deepcopy(self._psi),
            k0=copy.deepcopy(self._k),
            solver=solver,
            norm=lambda x, p: x.norm(p),
            matvec=lambda A, x: A @ x,
            tol=tol,
            max_iter=max_iter,
            callback=rank_update,
        )
