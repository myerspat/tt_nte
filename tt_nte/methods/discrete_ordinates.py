"""
discrete_ordinates.py
"""

import numpy as np
from scipy.special import eval_legendre

from tt_nte.tensor_train import TensorTrain
from tt_nte.utils.utils import check_dim_size


class DiscreteOrdinates:
    _direction_spaces = [
        np.array([[1], [-1]]),
        np.array(np.meshgrid([1, -1], [1, -1])).T.reshape(-1, 2),
        np.array(np.meshgrid([1, -1], [1, -1], [1, -1])).T.reshape(-1, 3),
    ]

    def __init__(
        self,
        xs_server,
        geometry,
        num_ordinates,
        tt_fmt="tt",
        qtt_threshold=1e-15,
        octant_ords=None,
    ):
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
            tt_fmt=tt_fmt,
            qtt_threshold=qtt_threshold,
            octant_ords=octant_ords,
        )

    def _construct_tensor_trains(self):
        """
        Construct operaotr tensor trains as described in
        LANL TT paper.
        """
        num_nodes = self._geometry.num_nodes
        num_groups = self._xs_server.num_groups

        d = np.eye(num_nodes, k=0)
        d_p = np.eye(num_nodes, k=-1)
        d_m = np.eye(num_nodes, k=1)

        # Boundary condition matrix
        bc = [
            np.ones((num_nodes, 1)),
            np.ones((num_nodes, 1)),
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
        bcs = [self._geometry.bcs[0], self._geometry.bcs[3]]

        for i in range(2):
            C = np.zeros((2, 2), dtype=float)
            C[i, i] = 1.0

            # Angular point matrix
            Qu = np.kron(C, (-1) ** (i) * np.diag(self._octant_ords[:, 1]))

            # Streaming operator
            self._H.append([Ig, Qu, D[i]] if num_groups > 1 else [Qu, D[i]])

            # Integral operator
            A = np.zeros((2, 2), dtype=float)
            A[i, :] = 1
            F_Intg = np.kron(
                A,
                np.outer(np.ones(self._octant_ords.shape[0]), self._octant_ords[:, 0]),
            )

            # Add reflection
            if bcs[i] == "reflective":
                Ip_ref = np.zeros((num_nodes, num_nodes))
                Ip_ref[-i, -i] = 1 / 2

                D_ref = np.zeros((num_nodes, num_nodes))
                D_ref[-i, -i] = 1 / dx[0,] if i == 0 else 1 / dx[-1,]

                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (i + 1) % 2] = 1

                Qu_ref = np.kron(C_ref, -np.diag(self._octant_ords[:, 1]))

                # Streaming operator (reflective)
                self._H.append(
                    [Ig, Qu_ref, D_ref] if num_groups > 1 else [Qu_ref, D_ref]
                )

                # Total interaction operator (reflective)
                region = (
                    self._geometry.bc_regions(0)[0]
                    if i == 0
                    else self._geometry.bc_regions(3)[0]
                )
                self._H.append(
                    [
                        np.diag(self._xs_server.total(region)),
                        np.kron(C_ref, -IL),
                        Ip_ref,
                    ]
                    if num_groups > 1
                    else [
                        self._xs_server.total(region) * np.kron(C_ref, -IL),
                        Ip_ref,
                    ]
                )

            # Iterate through regions in the problem from left to right
            for mat in self._geometry.regions:
                # Define XSs
                total = np.diag(self._xs_server.total(mat))
                nu_fission = np.outer(
                    self._xs_server.chi, self._xs_server.nu_fission(mat)
                )

                total = np.squeeze(total)
                nu_fission = np.squeeze(nu_fission)

                # Region mask for spatial dependence
                mask = self._geometry.region_mask(mat)[0]
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
                            (2 * l + 1) * eval_legendre(l, self._octant_ords[:, 1]),
                            self._octant_ords[:, 0]
                            * eval_legendre(l, self._octant_ords[:, 1]),
                        ),
                    )

                    # Scattering operator
                    self._S.append(
                        [scatter_gtg, S_Intg, Ip[i] * bc[i] * mask]
                        if num_groups > 1
                        else [scatter_gtg * S_Intg, Ip[i] * bc[i] * mask]
                    )

        # Construct TT objects
        self._H = TensorTrain(self._H)
        self._F = TensorTrain(self._F)
        self._S = TensorTrain(self._S)

        # Convert to QTT format
        if self._tt_fmt == "qtt":
            self._H.tt2qtt(self._qtt_threshold)
            self._F.tt2qtt(self._qtt_threshold)
            self._S.tt2qtt(self._qtt_threshold)

    # =====================================================================
    # Utility methods

    def update_settings(
        self,
        xs_server=None,
        geometry=None,
        num_ordinates=None,
        tt_fmt=None,
        qtt_threshold=None,
        octant_ords=None,
    ):
        """
        Update SN settings.
        """
        self._xs_server = self._xs_server if xs_server is None else xs_server
        self._geometry = self._geometry if geometry is None else geometry
        self._num_ordinates = (
            self._num_ordinates if num_ordinates is None else num_ordinates
        )
        self._tt_fmt = self._tt_fmt if tt_fmt is None else tt_fmt
        self._qtt_threshold = (
            self._qtt_threshold if qtt_threshold is None else qtt_threshold
        )

        # Assert supported TT formats
        assert self._tt_fmt == "tt" or self._tt_fmt == "qtt"

        # Assert dimensions are power of 2
        check_dim_size("ordinates", self._num_ordinates)
        check_dim_size("spatial edges", self._geometry.num_nodes)
        if self._xs_server.num_groups != 1:
            check_dim_size("energy groups", self._xs_server.num_groups)

        # Get quadrature set
        # 1D = (N, 2): w, mu
        # 2D = (N, 3): w, mu, eta
        # 3D = (N, 4): w, mu, eta, xi
        if octant_ords == None:
            # Square quadrature set
            if self._geometry.num_dim == 1:
                self._octant_ords = self._gauss_legendre(self._num_ordinates)
            elif self._geometry.num_dim == 2:
                self._octant_ords = np.unique(
                    self._chebyshev_legendre(self._num_ordinates * 4), axis=0
                )
            else:
                self._octant_ords = self._chebyshev_legendre(self._num_ordinates)
        else:
            self._octant_ords = octant_ords

        # Construct operator tensors
        self._construct_tensor_trains()

    # =====================================================================
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

        return np.concatenate([w[:, np.newaxis], mu[:, np.newaxis]], axis=1)

    @staticmethod
    def _gauss_chebyshev(N):
        """
        Gauss-Chebyshev quadrature.
        """
        gamma = (2 * np.arange(1, N / 2 + 1) - 1) * np.pi / (2 * N / 2)
        w = np.ones(int(N / 2)) * 1 / N

        return np.concatenate([w[:, np.newaxis], gamma[:, np.newaxis]], axis=1)

    @staticmethod
    def _chebyshev_legendre(N):
        """
        Chebyshev-Legendre (square) qudrature set given by
        https://www.osti.gov/servlets/purl/5958402.
        """
        n = np.round(np.sqrt(N / 2)).astype(int)

        # Compute quadrature
        q_l = DiscreteOrdinates._gauss_legendre(n)
        q_c = DiscreteOrdinates._gauss_chebyshev(n)

        w_l, mu = q_l[:, 0], q_l[:, 1]
        w_c, gamma = q_c[:, 0], q_c[:, 1]

        # Add quadrature on other side
        mu = np.concatenate([mu, -mu])
        w_l = np.concatenate([w_l, w_l])

        gamma = np.concatenate([gamma, -gamma])
        w_c = np.concatenate([w_c, w_c])

        # assert 2 * gamma.size * mu.size == N

        # Compute ordinates
        ordinates = np.zeros((N, 4))
        for i in range(mu.size):
            for j in range(gamma.size):
                ordinates[i * gamma.size + j, 0] = w_l[i] * w_c[j]
                ordinates[i * gamma.size + j, 1] = mu[i]
                ordinates[i * gamma.size + j, 2] = np.sqrt(1 - mu[i] ** 2) * np.cos(
                    gamma[j]
                )
                ordinates[i * gamma.size + j, 3] = np.sqrt(
                    1 - mu[i] ** 2 - ordinates[i * gamma.size + j, 2] ** 2
                )

        ordinates[int(N / 2) :, :] = ordinates[: int(N / 2), :]
        ordinates[int(N / 2) :, 2] *= -1

        # Get first octant ordinates
        ordinates = ordinates[
            np.argwhere(
                (ordinates[:, 1] > 0) & (ordinates[:, 2] > 0) & (ordinates[:, 3] > 0)
            ).flatten(),
            :,
        ]

        return ordinates

    # =====================================================================
    # Getters

    @property
    def H(self):
        return self._H

    @property
    def S(self):
        return self._S

    @property
    def F(self):
        return self._F

    @property
    def geometry(self):
        return self._geometry

    @property
    def xs_server(self):
        return self._xs_server

    @property
    def num_ordinates(self):
        return self._num_ordinates
