"""
discrete_ordinates.py
"""

import math

import numpy as np
from scipy.special import lpmv

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
        regions=None,
    ):
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
            tt_fmt=tt_fmt,
            qtt_threshold=qtt_threshold,
            octant_ords=octant_ords,
            regions=regions,
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
            diff = self._geometry.diff[i].reshape((-1, 1))

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
                            diff[[0],],
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
                for k in range(num_dim - 1, -1, -1):
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
                if bcs[int(j + 3 * dir_idx[j])] == "reflective":
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
                    sp_idx = num_dim - 1 - j
                    bc_mask = np.zeros((spatial_cores[sp_idx].shape[0], 1))
                    bc_mask[-dir_idx[j], 0] = 1
                    spatial_cores[sp_idx] = bc_mask * np.copy(spatial_cores[sp_idx])

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
                A,
                np.outer(np.ones(self._octant_ords.shape[0]), self._octant_ords[:, 0]),
            )

            # Total interaction, fission, and scattering operators
            # Iterate through each material region
            for region, mat in self._regions.items():
                # Region masks for spatial dependence
                masks = self._geometry.region_mask(region)

                # Add 0 at boundary condition and get spatial cores
                spatial_cores = []
                for j in range(num_dim - 1, -1, -1):
                    masks[j] = (
                        np.concatenate((np.zeros((1, 1)), masks[j]))
                        if dir_idx[j] == 0
                        else np.concatenate((masks[j], np.zeros((1, 1))))
                    )
                    spatial_cores.append(masks[j] * Ip[j][dir_idx[j]])

                # Append total interaction operator
                total = np.squeeze(np.diag(self._xs_server.total(mat)))
                self._H.append(
                    (
                        [
                            total,
                            np.kron(C, IL),
                        ]
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

                # Number of ordinates in an octant
                n = int(num_ordinates / num_octants)

                # Iterate through scattering moments
                def Y(l, m, ordinates, even=True):
                    y = (
                        (-1) ** m
                        * np.sqrt(
                            (2 * l + 1)
                            * math.factorial(l - abs(m))
                            / math.factorial(l + abs(m))
                        )
                        * lpmv(m, l, ordinates[:, 1])
                    )

                    if m == 0:
                        return y
                    elif even:
                        gamma = np.arccos(
                            ordinates[:, 2] / np.sqrt(1 - ordinates[:, 1] ** 2)
                        )
                        return y * np.cos(m * gamma)
                    else:
                        gamma = np.arcsin(
                            ordinates[:, 3] / np.sqrt(1 - ordinates[:, 1] ** 2)
                        )
                        return y * np.sin(m * gamma)

                for l in range(self._xs_server.num_moments):
                    # Scattering integral operator
                    S_Intg = np.zeros(F_Intg.shape)

                    # Outgoing ordinates
                    out_ords = np.copy(self._octant_ords)
                    out_ords[:, 1:] *= octant[np.newaxis, :]

                    for k in range(octants.shape[0]):
                        # Incoming ordinates
                        in_ords = np.copy(self._octant_ords)
                        in_ords[:, 1:] *= octants[[k], :]

                        S_Intg[
                            int(i * n) : int(i * n + n),
                            int(k * n) : int(k * n + n),
                        ] = np.outer(
                            Y(l, 0, out_ords),
                            self._octant_ords[:, 0] * Y(l, 0, in_ords),
                        )

                    if num_dim > 1:
                        for m in range(1, l + 1):
                            for k in range(octants.shape[0]):
                                # Incoming ordinates
                                in_ords = np.copy(self._octant_ords)
                                in_ords[:, 1:] *= octants[[k], :]

                                S_Intg[
                                    int(i * n) : int(i * n + n),
                                    int(k * n) : int(k * n + n),
                                ] += 2 * np.outer(
                                    Y(l, m, out_ords, even=True),
                                    self._octant_ords[:, 0]
                                    * Y(l, m, in_ords, even=True),
                                ) + (
                                    2
                                    * np.outer(
                                        Y(l, m, out_ords, even=False),
                                        self._octant_ords[:, 0]
                                        * Y(l, m, in_ords, even=False),
                                    )
                                    if num_dim > 2
                                    else 0
                                )
                    # Append train to scattering operator
                    scatter_gtg = np.squeeze(self._xs_server.scatter_gtg(mat)[l,])
                    self._S.append(
                        (
                            [
                                scatter_gtg,
                                S_Intg,
                            ]
                            if num_groups > 1
                            else [scatter_gtg * S_Intg]
                        )
                        + spatial_cores
                    )

        # Construct TT objects
        self._H = TensorTrain(self._H, fmt=self._tt_fmt, threshold=self._qtt_threshold)
        self._F = TensorTrain(self._F, fmt=self._tt_fmt, threshold=self._qtt_threshold)
        self._S = TensorTrain(self._S, fmt=self._tt_fmt, threshold=self._qtt_threshold)

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
        regions=None,
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
        for i in range(self._geometry.num_dim):
            check_dim_size("spatial edges", self._geometry.diff[i].size + 1)
        if self._xs_server.num_groups != 1:
            check_dim_size("energy groups", self._xs_server.num_groups)

        # Get quadrature set
        # 1D = (N, 2): w, mu
        # 2D = (N, 3): w, mu, eta
        # 3D = (N, 4): w, mu, eta, xi
        self._octant_ords = (
            self._compute_square_set(self._num_ordinates, self._geometry.num_dim)
            if octant_ords is None
            else octant_ords
        )
        if octant_ords is not None:
            self._num_ordinates = int(
                2**self._geometry.num_dim * self._octant_ords.shape[0]
            )

        self._octant_ords[:, 0] = (
            1
            / self._direction_spaces[self._geometry.num_dim - 1].shape[0]
            * self._octant_ords[:, 0]
            / np.sum(self._octant_ords[:, 0])
        )

        if regions is None:
            self._regions = {region: region for region in self._geometry.regions}
        else:
            self._regions = regions

        # Construct operator tensors
        self._construct_tensor_trains()

    def get_group(self, psi, g):
        """
        Get gth group of TT-decomposed solution.
        """
        return TensorTrain([psi.cores[0][:, [g], :, :]] + psi.cores[1:])

    # =====================================================================
    # Quadrature sets

    @staticmethod
    def _compute_square_set(N, num_dim):
        if num_dim == 1:
            return DiscreteOrdinates._gauss_legendre(N)
        elif num_dim == 2:
            octant_ords = DiscreteOrdinates._chebyshev_legendre(N * 2)[:, :-1]
            octant_ords[:, 0] *= 2
            return octant_ords
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

        w_l, mu = q_l[:, 0], q_l[:, 1]
        w_c, gamma = q_c[:, 0], q_c[:, 1]

        # Assert number of ordinates
        assert 8 * gamma.size * mu.size == N

        ordinates = np.zeros((int(N / 8), 4))
        for i in range(mu.size):
            for j in range(gamma.size):
                k = i * gamma.size + j
                ordinates[k, 0] = w_l[i] * w_c[j]
                ordinates[k, 1] = mu[i]
                ordinates[k, 2] = np.sqrt(1 - mu[i] ** 2) * np.cos(gamma[j])
                ordinates[k, 3] = np.sqrt(1 - mu[i] ** 2 - ordinates[k, 2] ** 2)

        return ordinates

    # =====================================================================
    # Getters

    @property
    def H(self):
        assert isinstance(self._H, TensorTrain)
        return self._H

    @property
    def S(self):
        assert isinstance(self._S, TensorTrain)
        return self._S

    @property
    def F(self):
        assert isinstance(self._F, TensorTrain)
        return self._F

    @property
    def Int_N(self):
        """
        Angular integral operator to get scalar flux from angular flux.
        """
        # Define integration TT
        return TensorTrain(
            (
                [np.identity(self._xs_server.num_groups)]
                if self._xs_server.num_groups > 1
                else []
            )
            + [np.block([2**self._geometry.num_dim * [self._octant_ords[:, 0]]])]
            + [
                np.identity(diff.size + 1)
                for diff in self._geometry.diff[: self._geometry.num_dim]
            ]
        )

    @property
    def geometry(self):
        return self._geometry

    @property
    def xs_server(self):
        return self._xs_server

    @property
    def num_ordinates(self):
        return self._num_ordinates
