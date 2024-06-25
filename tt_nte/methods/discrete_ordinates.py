"""
discrete_ordinates.py
"""

import numpy as np
from scipy.special import eval_legendre

from tt_nte.tensor_train import TensorTrain
from tt_nte.utils.utils import check_dim_size


class DiscreteOrdinates:
    def __init__(
        self,
        xs_server,
        geometry,
        num_ordinates,
        tt_fmt="tt",
        qtt_threshold=1e-15,
    ):
        self.update_settings(
            xs_server=xs_server,
            geometry=geometry,
            num_ordinates=num_ordinates,
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
        check_dim_size("spatial edges", self._geometry.num_nodes + 1)
        if self._xs_server.num_groups != 1:
            check_dim_size("energy groups", self._xs_server.num_groups)

        # Get square quadrature set (in 1D this is Gauss-Legendre)
        self._w, self._mu = self._gauss_legendre(self._num_ordinates)

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

        return w, mu

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
