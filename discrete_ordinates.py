"""
discrete_ordinates.py
"""

import copy
import matplotlib.pyplot as plt

import numpy as np
import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.sle as sle
import scikit_tt.tensor_train as tt
import scipy.sparse as sp
from scikit_tt import TT
from scipy.sparse.linalg import eigs, inv
from scipy.special import eval_legendre
from scipy.linalg import svd


class DiscreteOrdinates:
    def __init__(
        self,
        xs_server,
        geometry,
        num_ordinates,
        tol=1e-6,
        max_iter=100,
        tt_fmt="tt",
        qtt_threshold=1e-15,
    ):
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
            tol=tol,
            max_iter=max_iter,
            tt_fmt=tt_fmt,
            qtt_threshold=qtt_threshold,
        )

    def _construct_tensor_trains(self):
        """
        Construct operaotr tensor trains as described in
        LANL TT paper.
        """
        num_nodes = self._geometry.num_nodes
        num_groups = self._xs_server.num_groups

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
        dx = self._geometry.dx
        D = [
            1 / np.concatenate((dx[[0],], dx)) * (d - d_p),
            1 / np.concatenate((dx, dx[[-1],])) * (d_m - d),
        ]

        # Group identity matrix
        Ig = np.identity(self._xs_server.num_groups)

        # Ordinate identity matrix
        IL = np.identity(int(self._num_ordinates / 2))

        # Iterate over two half spaces
        # Because we're in 1D we have 2 direction spaces
        # 2D - 4 (quadrants)
        # 3D - 8 (octants)
        self._H = []
        self._F = []
        self._S = []
        bcs = [self._geometry.left_bc, self._geometry.right_bc]

        for i in range(2):
            C = np.zeros((2, 2), dtype=float)
            C[i, i] = 1.0

            # Angular point matrix
            Qu = np.kron(C, (-1) ** (i) * np.diag(self._mu))

            # Streaming operator
            self._H.append([Ig, Qu, D[i]] if num_groups > 1 else [Qu, D[i]])

            # Integral operator
            A = np.zeros((2, 2), dtype=float)
            A[i, :] = 1
            F_Intg = np.kron(A, np.outer(np.ones(self._w.size), self._w))

            # Add reflection
            if bcs[i] == "reflective":
                Ip_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                Ip_ref[-i, -i] = 1 / 2

                D_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                D_ref[-i, -i] = 1 / dx[0,] if i == 0 else 1 / dx[-1,]

                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (i + 1) % 2] = 1

                Qu_ref = np.kron(C_ref, -np.diag(self._mu))

                # Streaming operator (reflective)
                self._H.append(
                    [Ig, Qu_ref, D_ref] if num_groups > 1 else [Qu_ref, D_ref]
                )

                # Total interaction operator (reflective)
                region = (
                    self._geometry.regions[0] if i == 0 else self._geometry.regions[-1]
                )
                self._H.append(
                    [
                        np.diag(self._xs_server.total(region.material)),
                        np.kron(C_ref, -IL),
                        Ip_ref,
                    ]
                    if num_groups > 1
                    else [
                        self._xs_server.total(region.material) * np.kron(C_ref, -IL),
                        Ip_ref,
                    ]
                )

            # Iterate through regions in the problem from left to right
            for region in self._geometry.unique_regions:
                # Define XS matrices
                mat = region.material

                total = np.diag(self._xs_server.total(mat))
                nu_fission = np.outer(
                    self._xs_server.chi, self._xs_server.nu_fission(mat)
                )

                total = np.squeeze(total)
                nu_fission = np.squeeze(nu_fission)

                # Region mask for spatial dependence
                mask = self._geometry.region_mask(region)
                mask = (
                    np.concatenate((mask[[0],], mask))
                    if i == 0
                    else np.concatenate((mask, mask[[-1],]))
                )

                # Total interaction operator
                self._H.append(
                    [total, np.kron(C, IL), mask * Ip[i]]
                    if num_groups > 1
                    else [total * np.kron(C, IL), mask * Ip[i]]
                )

                # Fission operator
                self._F.append(
                    [nu_fission, F_Intg, Ip[i] * bc[i] * mask]
                    if num_groups > 1
                    else [nu_fission * F_Intg, Ip[i] * bc[i] * mask]
                )

                # Iterate through scattering moments
                for l in range(self._xs_server.scatter_gtg(mat).shape[0]):
                    scatter_gtg = np.squeeze(self._xs_server.scatter_gtg(mat)[l,])

                    # Integral operator
                    A = np.zeros((2, 2), dtype=float)
                    A[i, :] = 1
                    A[i, 1 - i] = (-1) ** l
                    S_Intg = np.kron(
                        A,
                        np.outer(
                            (2 * l + 1) * eval_legendre(l, self._mu),
                            self._w * eval_legendre(l, self._mu),
                        ),
                    )

                    # Scattering operator
                    self._S.append(
                        [scatter_gtg, S_Intg, Ip[i] * bc[i] * mask]
                        if num_groups > 1
                        else [scatter_gtg * S_Intg, Ip[i] * bc[i] * mask]
                    )

        # Construct TT objects
        self._H = self._tensor_train(self._H)
        self._F = self._tensor_train(self._F)
        self._S = self._tensor_train(self._S)

        # Convert to QTT if requested
        if self._tt_fmt == "qtt":
            self._H = self.tt2qtt(self._H)
            self._F = self.tt2qtt(self._F)
            self._S = self.tt2qtt(self._S)

    # ==============================================================================
    # Full matrix solvers

    def solve_ges(self):
        """
        Solve SN using scipy.sparse.linalg.eigs.
        """
        # Get operators in CSC format
        H = self.H("csc")
        S = self.S("csc")
        F = self.F("csc")

        k, psi = eigs(F, 1, H - S)
        psi = np.real(psi).flatten()

        if np.sum(psi) < 0:
            psi = -psi

        return np.real(k)[0], psi / np.linalg.norm(psi)

    def solve_matrix_power(self):
        """
        Solve SN using power iteration with full matrices.
        """
        psi_old = np.random.rand(
            (self._geometry.num_nodes + 1)
            * self._num_ordinates
            * self._xs_server.num_groups
        ).reshape((-1, 1))
        k_old = np.random.rand(1)[0]

        err = 1.0

        # Get operators in CSC format
        H_inv = inv(self.H("csc"))
        S = self.S("csc")
        F = self.F("csc")

        for _ in range(self._max_iter):
            psi_new = H_inv.dot((S + 1 / k_old * F).dot(psi_old))

            # Compute new eigenvalue and eigenvector L2 error
            k_new = k_old * np.sum(F.dot(psi_new)) / np.sum(F.dot(psi_old))
            err = np.linalg.norm(psi_new - psi_old, ord=2)

            if err < self._tol:
                return k_new, psi_new.flatten() / np.linalg.norm(psi_new.flatten())

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

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

        psi = psi.full().flatten() / psi.norm()

        if np.sum(psi) < 0:
            psi = -psi

        return k, psi

    def solve_TT_power(self, method="als", rank=4, threshold=1e-12, max_rank=np.infty):
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
                return k_new, psi_new.full().flatten() / psi_new.norm()

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

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

    def tt2qtt(self, tt, cores=None):
        """
        Transform TT formatted operator to QTT format.
        """
        # Reshape and permute dims to list of [2] * l_k
        new_dims = []
        if cores is None:
            for dim_size in tt.row_dims:
                new_dims.append([2] * self._get_degree(dim_size))

        else:
            new_dims = np.copy(tt.row_dims)
            for core in cores:
                new_dims[core] = [2] * self._get_degree(new_dims[core])

        # Transform operator into QTT format with threshold
        # for SVD decomposition
        return tt.tt2qtt(new_dims, new_dims, threshold=self._qtt_threshold)

    def _get_degree(self, dim_size):
        """
        Get degree where the dimension size is 2 ** degree.
        """
        return int(np.round(np.log(dim_size) / np.log(2)))

    def update_settings(
        self,
        xs_server=None,
        geometry=None,
        num_ordinates=None,
        tol=None,
        max_iter=None,
        tt_fmt=None,
        qtt_threshold=None,
    ):
        """
        Update SN settings.
        """
        self._xs_server = self._xs_server if xs_server is None else xs_server
        self._geometry = self._geometry if geometry is None else geometry
        self._num_ordinates = (
            self._num_ordinates if num_ordinates is None else num_ordinates
        )
        self._tol = self._tol if tol is None else tol
        self._max_iter = self._max_iter if max_iter is None else max_iter
        self._tt_fmt = self._tt_fmt if tt_fmt is None else tt_fmt
        self._qtt_threshold = (
            self._qtt_threshold if qtt_threshold is None else qtt_threshold
        )

        # Assert supported TT formats
        assert self._tt_fmt == "tt" or self._tt_fmt == "qtt"

        # Assert dimensions are power of 2
        self._check_dim_size("ordinates", self._num_ordinates)
        self._check_dim_size("spatial edges", self._geometry.num_nodes + 1)
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

    def _format_tt(self, tt, fmt):
        """
        Formatting function to get TT, full matrix, or CSC formatted operators.
        """
        if fmt == "tt":
            return tt
        elif fmt == "full":
            return tt.matricize()
        elif fmt == "csc":
            return sp.csc_matrix(tt.matricize())
        else:
            raise RuntimeError(f"Format requested ({fmt}) is not supported")

    def plot_qtt_svd(self):
        fig, axs = plt.subplots(3)
        tts = [self._H, self._F, self._S]
        tt_names = ["LHS Operator", "Fission Operator", "Scattering Operator"]

        for tt_idx in range(len(tts)):
            tt = tts[tt_idx]
            tt_tensor = tt.copy()

            # QTT shaping
            new_dims = []
            for dim_size in tt.row_dims:
                new_dims.append([2] * self._get_degree(dim_size))

            for i in range(tt.order):
                # Get core features
                core = tt_tensor.cores[i]
                rank = tt_tensor.ranks[i]
                row_dim = tt_tensor.row_dims[i]
                col_dim = tt_tensor.col_dims[i]

                c = plt.cm.rainbow(np.linspace(0, 1, tt.order))[i]

                for j in range(len(new_dims[i]) - 1):
                    # Set new row_dim and col_dim for reshape
                    row_dim = int(row_dim / new_dims[i][j])
                    col_dim = int(col_dim / new_dims[i][j])

                    # Reshape and transpose core
                    core = core.reshape(
                        rank,
                        new_dims[i][j],
                        row_dim,
                        new_dims[i][j],
                        col_dim,
                        tt_tensor.ranks[i + 1],
                    ).transpose([0, 1, 3, 2, 4, 5])

                    # Apply SVD to split core
                    [_, s, _] = svd(
                        core.reshape(
                            rank * new_dims[i][j] ** 2,
                            row_dim * col_dim * tt_tensor.ranks[i + 1],
                        ),
                        full_matrices=False,
                        overwrite_a=True,
                        check_finite=False,
                        lapack_driver="gesvd",
                    )

                    # Plot SVD singular values
                    axs[tt_idx].plot(s, c=c)

        return fig, axs

    # ==============================================================================
    # Quadrature sets

    @staticmethod
    def _gauss_legendre(N):
        """
        Gauss Legendre quadrature set. Only given for positive half space.
        """
        mu, w = np.polynomial.legendre.leggauss(N)
        w = w[: int(mu.size / 2)] / 2
        mu = np.abs(mu[: int(mu.size / 2)])

        assert np.round(np.sum(w), 1) == 0.5

        return w, mu

    # ==============================================================================
    # Getters

    def H(self, fmt="tt"):
        return self._format_tt(self._H, fmt)

    def S(self, fmt="tt"):
        return self._format_tt(self._S, fmt)

    def F(self, fmt="tt"):
        return self._format_tt(self._F, fmt)

    @property
    def num_ordinates(self):
        return self._num_ordinates
