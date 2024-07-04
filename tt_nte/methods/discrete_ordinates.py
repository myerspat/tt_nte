"""
discrete_ordinates.py
"""
import math

import numpy as np
from scipy.special import eval_legendre, lpmv

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
        octants = self._direction_spaces[self._geometry.num_dim - 1]
        num_octants = octants.shape[0]
        num_dim = self._geometry.num_dim
        num_groups = self._xs_server.num_groups
        num_ordinates = self._num_ordinates

        # Differential and interpolation operators
        D = []
        Ip = []

        for i in range(num_dim):
            # Get differential length along dimension
            diff = self._geometry.diff[i]

            # Number of nodes along dimension
            dim_num_nodes = diff.size + 1

            d = np.eye(dim_num_nodes, k=0)
            d_p = np.eye(dim_num_nodes, k=-1)
            d_m = np.eye(dim_num_nodes, k=1)

            # Add to differential operator
            D.append(
                [
                    (d - d_p)
                    / np.concatenate(
                        [
                            diff[
                                [
                                    0,
                                ]
                            ],
                            diff,
                        ]
                    ),
                    (d_m - d) / np.concatenate([diff, diff[[-1],]]),
                ]
            )

            # Add to interpolation operator
            Ip.append([(d + d_p) / 2, (d + d_m) / 2])

        # Group identity matrix
        Ig = np.identity(num_groups)

        # Ordinate identity matrix
        IL = np.identity(int(num_ordinates / num_octants))

        # Iterate over half-spaces/quadrants/octants and append
        # TT cores
        self._H = []
        self._F = []
        self._S = []

        bcs = self._geometry.bcs

        for i in range(num_octants):
            octant = octants[i, :]

            # Index into D, Ip, bc based on direction (1 or -1)
            dir_idx = [0 if dir > 0 else 1 for dir in octant]

            C = np.zeros((num_octants, num_octants), dtype=float)
            C[i, i] = 1.0

            for j in range(num_dim):
                # Angular point matrix
                Q = np.kron(C, octant[j] * np.diag(self._octant_ords[:, j + 1]))

                # Get spatial cores in correct order depending on dimension
                spatial_cores = []
                for k in range(num_dim):
                    if j != k:
                        spatial_cores.append(Ip[k][dir_idx[k]])
                    else:
                        spatial_cores.append(D[k][dir_idx[k]])

                # Append dimension streaming operator
                self._H.append(
                    (
                        [
                            Ig,
                            Q,
                        ]
                        if num_groups > 1
                        else [Q]
                    )
                    + spatial_cores
                )

                # Add reflective boundary condition
                if bcs[j + 3 * dir_idx[octant[j]]] == "reflective":
                    # Find reflected octant
                    ref_octant = np.copy(octant)
                    ref_octant[j] *= -1

                    # Place values in reflected octant position
                    C_ref = np.zeros((num_octants, num_octants), dtype=float)
                    C_ref[i, np.where((octants == ref_octant).all(axis=1))[0]] = 1

                    Q_ref = np.kron(
                        C_ref, ref_octant[j] * np.diag(self._octant_ords[:, j + 1])
                    )

                    # Apply boundary mask to spatial cores
                    for k in range(num_dim):
                        bc_mask = np.zeros((spatial_cores[k].shape[0], 1))
                        bc_mask[-dir_idx[k], 0] = 1
                        spatial_cores[k] *= bc_mask

                    # Append boundary condition
                    self._H.append(
                        (
                            [
                                Ig,
                                Q_ref,
                            ]
                            if num_groups > 1
                            else [Q_ref]
                        )
                        + spatial_cores
                    )

            # Fission integral operator
            A = np.zeros((num_octants, num_octants), dtype=float)
            A[i, :] = 1
            F_Intg = np.kron(
                A, np.outer(self._octant_ords.shape[0], self._octant_ords[:, 0])
            )

            # Total interaction, fission, and scattering operators
            # Iterate through each material region
            for mat in self._geometry.regions:
                # Region masks for spatial dependence
                masks = self._geometry.region_mask(mat)

                # Add 0 at boundary condition and get spatial cores
                spatial_cores = []
                for j in range(num_dim):
                    masks[j] = (
                        np.concatenate((0, masks[j]))
                        if dir_idx[j] == 0
                        else np.concatenate((masks[j], 0))
                    )
                    spatial_cores.append(masks[j] * Ip[j][dir_idx[j]])

                # Append total interaction operator
                total = np.squeeze(np.diag(self._xs_server.total(mat)))
                self._H.append(
                    (
                        [total, np.kron(C, IL)]
                        if num_groups > 1
                        else [total * np.kron(C, IL)]
                    )
                    + spatial_cores
                )

                # Append fission operator
                nu_fission = np.squeeze(
                    np.outer(self._xs_server.chi, self._xs_server.nu_fission(mat))
                )
                self._F.append(
                    ([nu_fission, F_Intg] if num_groups > 1 else [nu_fission * F_Intg])
                    + spatial_cores
                )

                # # Number of ordinates in an octant
                # n = int(num_ordinates / num_octants)
                #
                # # Iterate through scattering moments
                # def Y(l, m, ordinates):
                #     y = (
                #         (-1) ** m
                #         * np.sqrt(
                #             (2 * l + 1)
                #             * math.factorial(l - abs(m))
                #             / math.factorial(l + abs(m))
                #         )
                #         * lpmv(m, l, ordinates[:, 1])
                #     )
                #
                #     if m == 0:
                #         return y
                #     elif m % 2 == 0:
                #         return (
                #             y * m * ordinates[:, 2] / np.sqrt(1 - ordinates[:, 1] ** 2)
                #         )
                #     else:
                #         return (
                #             y * m * ordinates[:, 3] / np.sqrt(1 - ordinates[:, 1] ** 2)
                #         )
                #
                # for l in range(self._xs_server.scatter_gtg(mat).shape[0]):
                #
                #     for m in range(l + 1):
                #
                #     # Scattering integral Operator
                #     S_intg = np.zeros(F_Intg.shape)
                #
                #     for k in range(octants.shape[0]):
                #         S_Intg_p = np.zeros((n, n))

                        # S_Intg[
                        #     i
                        #     * num_ordinates
                        #     / num_octants : i
                        #     * num_ordinates
                        #     / num_octants
                        #     + num_ordinates / num_octants,
                        #     k
                        #     * num_ordinates
                        #     / num_octants : k
                        #     * num_ordinates
                        #     / num_octants
                        #     + num_ordinates / num_octants,
                        # ] = np.outer()

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
        # 2/3D = (N, 4): w, mu, eta, xi
        self._octant_ords = (
            self._compute_square_set(self._num_ordinates, self._geometry.num_dim)
            if octant_ords is None
            else octant_ords
        )

        # Construct operator tensors
        self._construct_tensor_trains()

    # =====================================================================
    # Quadrature sets

    @staticmethod
    def _compute_square_set(N, num_dim):
        if num_dim == 1:
            return DiscreteOrdinates._gauss_legendre(N)
        # elif num_dim == 2:
        #     octant_ords = DiscreteOrdinates._chebyshev_legendre(N * 2)
        #     octant_ords[:, 0] *= 2
        #     return octant_ords
        else:
            return DiscreteOrdinates._chebyshev_legendre(N)

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
        gamma = (2 * np.arange(1, int(N / 2) + 1) - 1) * np.pi / (2 * N)
        w = np.ones(int(N / 2)) / (2 * N)

        return np.concatenate([w[:, np.newaxis], gamma[:, np.newaxis]], axis=1)

    @staticmethod
    def _chebyshev_legendre(N):
        """
        Chebyshev-Legendre (square) qudrature set given by
        https://www.osti.gov/servlets/purl/5958402.
        """
        assert N % 8 == 0

        n = np.round(np.sqrt(N / 2)).astype(int)

        # Compute quadrature
        q_l = DiscreteOrdinates._gauss_legendre(n)
        q_c = DiscreteOrdinates._gauss_chebyshev(n)

        w_l, xi = q_l[:, 0], q_l[:, 1]
        w_c, omega = q_c[:, 0], q_c[:, 1]

        # Assert number of ordinates
        assert 8 * omega.size * xi.size == N

        ordinates = np.zeros((int(N / 8), 4))
        for i in range(xi.size):
            for j in range(omega.size):
                k = i * omega.size + j
                ordinates[k, 0] = w_l[i] * w_c[j]
                ordinates[k, 1] = np.sqrt(1 - xi[i] ** 2) * np.cos(omega[j])
                ordinates[k, 2] = np.sqrt(1 - xi[i] ** 2 - ordinates[k, 1] ** 2)
                ordinates[k, 3] = xi[i]

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
