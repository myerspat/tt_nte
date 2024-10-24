import copy
import warnings

import numpy as np


class Solver(object):
    def __init__(self, H, S, F, verbose):
        """
        Construct Solver object with the NTE operators.
        """
        self._H = H
        self._S = S
        self._F = F

        self._verbose = verbose

    # =======================================================================
    # Methods

    def _ges(self, solver, norm):
        """
        Base generalized eigenvalue solver algorithm.
        """
        if self._verbose:
            print(f"-- {self.__class__.__name__} Generalized Eigenvalue Solver")

        # Run solver
        self._k, self._psi = solver(self._F, self._H - self._S)

        # Flip eigenvector if negative
        if norm(self._psi, 1) < 0:
            self._psi *= -1

        # Normalize psi
        self._psi *= 1 / norm(self._psi, 2)

    def _power(self, psi0, k0, solver, norm, matvec, tol, max_iter, callback=None):
        """
        Base power iteration algorithm.
        """
        if self._verbose:
            print(f"-- {self.__class__.__name__} Power Iteration")

        self._num_iter = 0
        err0 = 1

        for _ in range(max_iter):
            # Compute x in Ax = b
            self._psi = solver(A=self._H, B=self._S + self._F * (1 / k0), x0=psi0)

            # Compute new eigenvalue and eigenvector L2 error
            self._k = (
                k0
                * norm(matvec(self._F, self._psi), 1)
                / norm(matvec(self._F, psi0), 1)
            )
            self._err = norm(self._psi - psi0, 2) / norm(psi0, 2)

            # Increment iteration number
            self._num_iter += 1

            if self._verbose and self._num_iter % 10 == 0:
                print(
                    f"--   Iteration = {self._num_iter},"
                    + f" k = {np.round(self._k, 5)},"
                    + f" psi error = {self._err}"
                )

            # Run callback if given
            if callback:
                callback(self)

            # Break if tolerance is met
            if self._err < tol:
                if self._verbose:
                    print(f"-- Converged: k = {np.round(self._k, 8)}")

                # Normalize eigenvector
                self._psi *= 1 / norm(self._psi, 2)

                # Flip eigenvector if negative
                if norm(self._psi, 1) < 0:
                    self._psi *= -1

                return

            elif self._err > err0 and self._num_iter > 2:
                if self._verbose:
                    warnings.warn(
                        f"Iteration {self._num_iter} error is greater"
                        + f" than iteration {self._num_iter - 1}"
                    )

            # Copy results for next iteration
            psi0 = copy.deepcopy(self._psi)
            k0 = copy.deepcopy(self._k)
            err0 = copy.deepcopy(self._err)

        # Normalize eigenvector
        self._psi *= 1 / norm(self._psi, 2)

        if self._verbose:
            warnings.warn(
                f"Maximum number of power iterations ({max_iter})"
                + f" without convergence to error < {tol}",
                RuntimeWarning,
            )

    # =======================================================================
    # Getters

    @property
    def k(self):
        return self._k

    @property
    def psi(self):
        return self._psi

    @property
    def error(self):
        return self._err

    @property
    def num_iter(self):
        return self._num_iter
