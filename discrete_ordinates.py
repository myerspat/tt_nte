"""
discrete_ordinates.py
"""
import copy

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs, inv

import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.sle as sle
import scikit_tt.tensor_train as tt
from scikit_tt import TT


class DiscreteOrdinates:
    def __init__(
        self,
        xs_server,
        geometry,
        num_ordinates,
        bc="vacuum",
        tol=1e-6,
        max_iter=1000,
    ):
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
            bc=bc,
            tol=tol,
            max_iter=max_iter,
        )

    def _construct_tensor_trains(self):
        """
        Construct operaotr tensor trains as described in
        LANL TT paper.
        """
        # Use only first material for now
        mat = self._geometry.materials[0]

        dx = self._geometry.mat_dx(mat)
        num_nodes = self._geometry.mat_num_nodes(mat)

        d = np.eye(num_nodes + 1, k=0)
        d_p = np.eye(num_nodes + 1, k=-1)
        d_m = np.eye(num_nodes + 1, k=1)

        # Boundary condition matrix
        bc = [
            np.ones((num_nodes + 1, 1)),
            np.ones((num_nodes + 1, 1)),
        ]
        bc[1][-1, 0] = 0
        bc[0][0, 0] = 0

        # Interpolation
        Ip = [1 / 2 * (d + d_p), 1 / 2 * (d + d_m)]

        # Differentiation matrix
        D = [1 / dx * (d - d_p), 1 / dx * (d_m - d)]

        # Iterate over two half spaces
        # Because we're in 1D we have 2 direction spaces
        # 2D - 4 (quadrants)
        # 3D - 8 (octants)
        self._H = []
        self._F = []
        self._S = []
        for i in range(2):
            C = np.zeros((2, 2), dtype=float)
            C[i, i] = 1.0

            Ig = np.identity(self._xs_server.num_groups)
            IL = np.identity(int(self._num_ordinates / 2))

            # Angular point matrix
            Qu = np.kron(C, (-1) ** (i) * np.diag(self._mu))

            # Streaming
            if self._xs_server.num_groups > 1:
                self._H.append([Ig, Qu, D[i]])
            else:
                self._H.append([Qu, D[i]])

            # Total interaction
            if self._xs_server.num_groups > 1:
                self._H.append(
                    [
                        np.diag(self._xs_server.total(mat)),
                        np.kron(C, IL),
                        Ip[i],
                    ]
                )
            else:
                self._H.append(
                    [
                        self._xs_server.total(mat) * np.kron(C, IL),
                        Ip[i],
                    ]
                )

            if self._bc == "reflective":
                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (1 + i) % 2] = 1

                D_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                D_ref[-i, -i] = 1 / dx

                Ip_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                Ip_ref[-i, -i] = 1 / 2

                Qu_ref = np.kron(C_ref, -np.diag(self._mu))

                if self._xs_server.num_groups > 1:
                    self._H.append([Ig, Qu_ref, D_ref])
                    self._H.append(
                        [
                            np.diag(self._xs_server.total(mat)),
                            np.kron(C_ref, -IL),
                            Ip_ref,
                        ]
                    )
                else:
                    self._H.append([Qu_ref, D_ref])
                    self._H.append(
                        [self._xs_server.total(mat) * np.kron(C_ref, -IL), Ip_ref]
                    )

            # Integral operator matrix
            C[i, :] = 1
            Intg = np.kron(C, np.outer(np.ones(self._w.size), self._w))

            # Fission
            if self._xs_server.num_groups > 1:
                self._F.append(
                    [
                        np.outer(self._xs_server.chi[0], self._xs_server.nu_fission(mat)),
                        Intg,
                        Ip[i] * bc[i],
                    ]
                )
            else:
                self._F.append(
                    [
                        self._xs_server.chi * self._xs_server.nu_fission(mat) * Intg,
                        Ip[i] * bc[i],
                    ]
                )

            # Scatter
            if self._xs_server.num_groups > 1:
                self._S.append(
                    [
                        self._xs_server.scatter_gtg(mat),
                        Intg,
                        Ip[i] * bc[i],
                    ]
                )
            else:
                self._S.append(
                    [
                        self._xs_server.scatter_gtg(mat) * Intg,
                        Ip[i] * bc[i],
                    ]
                )

        # Construct TT objects
        self._H = self._tensor_train(self._H)
        self._F = self._tensor_train(self._F)
        self._S = self._tensor_train(self._S)
        mat_shape = (
            self._xs_server.num_groups * self._num_ordinates * (num_nodes + 1),
            self._xs_server.num_groups * self._num_ordinates * (num_nodes + 1),
        )
        print("H matrix: ", np.round(self._H.full().reshape(mat_shape),3))
        print("S matrix: ", np.round(self._S.full().reshape(mat_shape),3))
        print("F matrix: ", np.round(self._F.full().reshape(mat_shape),3))

    # ==============================================================================
    # Full matrix solvers

    def solve_ges(self):
        """
        Solve SN using scipy.sparse.linalg.eigs.
        """
        # Use only first material for now
        mat = self._geometry.materials[0]
        num_nodes = self._geometry.mat_num_nodes(mat)

        # Reshape operator tensors to 2D matrices
        mat_shape = (
            self._num_ordinates * (num_nodes + 1),
            self._num_ordinates * (num_nodes + 1),
        )

        H = sp.csc_matrix(self._H.full().reshape(mat_shape))
        S = sp.csc_matrix(self._S.full().reshape(mat_shape))
        F = sp.csc_matrix(self._F.full().reshape(mat_shape))

        k, psi = eigs(F, 1, H - S)

        return k, psi

    def solve_matrix_power(self):
        """
        Solve SN using power iteration with full matrices.
        """
        # Use only first material for now
        mat = self._geometry.materials[0]
        num_nodes = self._geometry.mat_num_nodes(mat)

        psi_old = np.random.rand(
            (num_nodes + 1) * self._num_ordinates * self._xs_server.num_groups
        ).reshape((-1, 1))
        k_old = np.random.rand(1)[0]

        err = 1.0

        # Reshape operator tensors to 2D matrices
        mat_shape = (
            self._xs_server.num_groups * self._num_ordinates * (num_nodes + 1),
            self._xs_server.num_groups * self._num_ordinates * (num_nodes + 1),
        )

        H_inv = inv(sp.csc_matrix(self._H.full().reshape(mat_shape)))
        S = sp.csc_matrix(self._S.full().reshape(mat_shape))
        F = sp.csc_matrix(self._F.full().reshape(mat_shape))

        for i in range(self._max_iter):
            psi_new = H_inv.dot((S + 1 / k_old * F).dot(psi_old))

            # Compute new eigenvalue and eigenvector L2 error
            k_new = k_old * np.sum(F.dot(psi_new)) / np.sum(F.dot(psi_old))
            err = np.linalg.norm(psi_new - psi_old, ord=2)

            if err < self._tol:
                return k_new, psi_new

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

            print("Iteration ", i, ",  k: ", k_old)

        raise RuntimeError(
            f"Maximum number of power iteration ({self._max_iter})"
            + f" without convergence to err < {self._tol}"
        )

    # ==============================================================================
    # TT solvers (scikit_tt and power iteration implementation)

    def solve_TT_scikit_als(self, rank=4):
        """
        SN using TT with ALS/generalized eigenvalue solver.
        """
        # Create initial guess
        psi = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=rank)

        # Run ALS GES
        k, psi, _ = evp.als(
            self._F, psi, operator_gevp=(self._H - self._S), conv_eps=self._tol
        )

        return k, psi.full().flatten()

    def solve_TT_power(self, method="als", rank=2, threshold=1e-12, max_rank=np.infty):
        """
        Solve SN using power iteration with TT ALS or MALS
        """
        psi_old = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=rank)

        k_old = np.random.rand(1)[0]
        err = 1.0

        solver = sle.als if method == "als" else sle.mals
        kwargs = (
            {} if method == "als" else {"threshold": threshold, "max_rank": max_rank}
        )

        for i in range(self._max_iter):
            psi_new = solver(
                self._H,
                psi_old,
                (self._S + 1 / k_old * self._F).dot(psi_old),
                **kwargs,
            )

            # Compute new eigenvalue and eigenvector L2 error
            k_new = (
                k_old * (self._F.dot(psi_new)).norm() / (self._F.dot(psi_old)).norm()
            )
            err = (psi_new - psi_old).norm(p=2)

            if err < self._tol:
                return k_new, psi_new.full().reshape(-1)

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

            print("k: ", k_old)

        raise RuntimeError(
            f"Maximum number of power iteration ({self._max_iter})"
            + f" without convergence to err < {self._tol}"
        )

    # ==============================================================================
    # Utility methods

    def _tensor_train(self, elements):
        """
        Create scikit_tt.TT from list of numpy.ndarray(s) or list of list of numpy.ndarray(s).
        """
        if isinstance(elements, list):
            # Ensure elements are 4D
            for i in range(len(elements)):
                if isinstance(elements[i], np.ndarray):
                    elements[i] = elements[i][np.newaxis, :, :, np.newaxis]
                else:
                    elements[i] = self._tensor_train(elements[i])

        else:
            return TT(elements)

        if isinstance(elements[0], np.ndarray):
            return TT(elements)
        else:
            tt = elements[0]
            for array in elements[1:]:
                tt += array

            return tt

    def update_settings(
        self,
        xs_server=None,
        geometry=None,
        num_ordinates=None,
        bc=None,
        tol=None,
        max_iter=None,
    ):
        """
        Update SN settings.
        """
        self._xs_server = self._xs_server if xs_server is None else xs_server
        self._geometry = self._geometry if geometry is None else geometry
        self._num_ordinates = (
            self._num_ordinates if num_ordinates is None else num_ordinates
        )
        self._bc = self._bc if bc is None else bc
        self._tol = self._tol if tol is None else tol
        self._max_iter = self._max_iter if max_iter is None else max_iter

        # Assert supported boundary conditions
        assert self._bc == "vacuum" or self._bc == "reflective"

        # Assert dimensions are power of 2
        self._check_dim_size("ordinates", self._num_ordinates)
        for mat in self._geometry.materials:
            self._check_dim_size("spatial edges", self._geometry.mat_num_nodes(mat) + 1)
        if self._xs_server.num_groups != 1:
            self._check_dim_size("energy groups", self._xs_server.num_groups)

        # Get square quadrature set (in 1D this is Gauss-Legendre)
        self._w, self._mu = self._gauss_legendre(self._num_ordinates)

        # Construct operator tensors
        self._construct_tensor_trains()

    @staticmethod
    def _check_dim_size(name, dim_size):
        """
        Check dimension sizes are powers of 2.
        """
        if not ((dim_size & (dim_size - 1) == 0) and dim_size != 0):
            raise RuntimeError(f"Number of {name} must be a power of 2")

    # ==============================================================================
    # Quadrature sets

    @staticmethod
    def _gauss_legendre(L):
        """
        Gauss Legendre quadrature set. Only given for positive half space.
        """
        mu, w = np.polynomial.legendre.leggauss(L)
        w = w[: int(mu.size / 2)] / 2
        mu = np.abs(mu[: int(mu.size / 2)])

        assert np.round(np.sum(w), 1) == 0.5

        return w, mu

    # ==============================================================================
    # Getters

    @property
    def num_ordinates(self):
        return self._num_ordinates

    @property
    def H(self):
        return self._H

    @property
    def S(self):
        return self._S

    @property
    def F(self):
        return self._F
