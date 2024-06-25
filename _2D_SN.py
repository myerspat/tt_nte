"""
2D_SN.py
2D attempt for TT solvers. 6/14/24
Ravi Shastri
"""

import copy
import numpy as np
import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.sle as sle
import scikit_tt.tensor_train as tt
import scipy.sparse as sp
from scikit_tt import TT
from scipy.sparse.linalg import eigs, inv
from scipy.special import eval_legendre
from scipy.special import roots_legendre
import utils


class DiscreteOrdinates2D:
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
        Construct operator tensor trains as described in
        LANL TT paper.
        """
        num_nodes = self._geometry.num_nodes
        num_groups = self._xs_server.num_groups
        # check normalization here. 
        num_ordinates = self._num_ordinates / 2
        
        # init TT matrices
        # Iterate over two half spaces
        # Because we're in 1D we have 2 direction spaces
        # 2D - 4 (quadrants)
        # 3D - 8 (octants)
        self._Hx = []
        self._Fx = []
        self._Sx = []
        self._Hy = []
        self._Fy = []
        self._Sy = []
        self._Hz = []
        self._Fz = []
        self._Sz = []

        d = np.eye(num_nodes + 1, k=0)
        # left and right
        d_r = np.eye(num_nodes + 1, k=-1)
        d_l = np.eye(num_nodes + 1, k=1)
        # up and down
        d_u = np.eye(num_nodes + 1, k=-1)
        d_d = np.eye(num_nodes + 1, k=1)
        # in and out
        d_i = np.eye(num_nodes + 1, k=-1)
        d_o = np.eye(num_nodes + 1, k=1)

        # Boundary condition matrix
        bcx = [
            np.ones((num_nodes + 1, 1)),
            np.ones((num_nodes + 1, 1)),
        ]
        
        bcx[1][-1] = 0
        bcx[0][0 ] = 0
        
        # Interpolation
        Ipx = [(d + d_r)/2,(d + d_l)/2]
        # Differentiation matrix
        dx = self._geometry.dx
        Dx = [
            1 / np.concatenate((dx[[0],], dx)) * (d - d_r),
            1 / np.concatenate((dx, dx[[-1],])) * (d_l - d),
        ]
                
        if num_nodes > 3:
            bcy = [
                np.ones((num_nodes + 1, 1)),
                np.ones((num_nodes + 1, 1)),
            ]
            bcy[1][-2] = 0
            bcy[0][1] = 0
            Ipy = [(d + d_u)/2,(d + d_d)/2]
            dy = self._geometry.dy
            Dy = [
                1 / np.concatenate((dy[[1],], dy)) * (d - d_u),
                1 / np.concatenate((dy, dy[[-2],])) * (d_d - d),
            ]
        
        if num_nodes > 7:
            bcz = [
                np.ones((num_nodes + 1, 1)),
                np.ones((num_nodes + 1, 1)),
            ]
            bcz[1][-3] = 0
            bcz[0][2] = 0    
            Ipz = [(d + d_i)/2,(d + d_o)/2]
            dz = self._geometry.dz
            Dz = [
                1 / np.concatenate((dz[[2],], dz)) * (d - d_i),
                1 / np.concatenate((dz, dz[[-3],])) * (d_o - d),
            ]


        # Group identity matrix
        Ig = np.identity(num_groups)

        # Ordinate identity matrix
        IL = np.identity(int(num_ordinates / 2))

        bcs = [
                self._geometry.left_bc, self._geometry.right_bc, 
                self._geometry.top_bc,  self._geometry.bottom_bc, 
                self.geometry.into_bc,  self.geometry.outof_bc
              ]

        for i in range(2): # x dimension
            C = np.zeros((2, 2), dtype=float)
            C[i, i] = 1.0

            # Angular point matrix
            Qmu = np.kron(C, (-1) ** (i) * np.diag(self._mu))

            # Streaming operator
            self._Hx.append([Ig, Qmu, Dx[i]] if num_groups > 1 else [Qmu, Dx[i]])

            # Integral operator
            A = np.zeros((2, 2), dtype=float)
            A[i, :] = 1
            F_Intg = np.kron(A, np.outer(np.ones(self._wx.size), self._wx))

            # Add reflection
            if bcs[i] == "reflective":
                Ip_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                Ip_ref[-i, -i] = 1 / 2

                D_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                D_ref[-i, -i] = 1 / dx[0,] if i == 0 else 1 / dx[-1,]

                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (i + 1) % 2] = 1

                Qmu_ref = np.kron(C_ref, -np.diag(self._mu))

                # Streaming operator (reflective)
                self._Hx.append(
                    [Ig, Qmu_ref, D_ref] if num_groups > 1 else [Qmu_ref, D_ref]
                )

                # Total interaction operator (reflective)
                region = (
                    self._geometry.regions[0] if i == 0 else self._geometry.regions[-1]
                )
                self._Hx.append(
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
                self._Hx.append(
                    [total, np.kron(C, IL), mask * Ipx[i]]
                    if num_groups > 1
                    else [total * np.kron(C, IL), mask * Ipx[i]]
                )

                # Fission operator
                self._Fx.append(
                    [nu_fission, F_Intg, Ipx[i] * bcx[i] * mask]
                    if num_groups > 1
                    else [nu_fission * F_Intg, Ipx[i] * bcx[i] * mask]
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
                            self._wx * eval_legendre(l, self._mu),
                        ),
                    )

                    # Scattering operator
                    self._Sx.append(
                        [scatter_gtg, S_Intg, Ipx[i] * bcx[i] * mask]
                        if num_groups > 1
                        else [scatter_gtg * S_Intg, Ipx[i] * bcx[i] * mask]
                    )
                    
        for i in range(2): # y dimension
            C = np.zeros((2, 2), dtype=float)
            C[i, i] = 1.0

            # Angular point matrix
            Qet = np.kron(C, (-1) ** (i) * np.diag(self._eta))

            # Streaming operator
            self._Hy.append([Ig, Qet, Dy[i]] if num_groups > 1 else [Qet, Dy[i]])

            # Integral operator
            A = np.zeros((2, 2), dtype=float)
            A[i, :] = 1
            F_Intg = np.kron(A, np.outer(np.ones(self._wy.size), self._wy))

            # Add reflection
            if bcs[i] == "reflective":
                # Ip_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                Ip_ref[-i, -i] = 1 / 2

                D_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                D_ref[-i, -i] = 1 / dy[0,] if i == 0 else 1 / dy[-1,]

                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (i + 1) % 2] = 1

                Qet_ref = np.kron(C_ref, -np.diag(self._eta))

                # Streaming operator (reflective)
                self._Hy.append(
                    [Ig, Qet_ref, D_ref] if num_groups > 1 else [Qet_ref, D_ref]
                )

                # Total interaction operator (reflective)
                region = (
                    self._geometry.regions[0] if i == 0 else self._geometry.regions[-1]
                )
                self._Hy.append(
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
                self._Hy.append(
                    [total, np.kron(C, IL), mask * Ipy[i]]
                    if num_groups > 1
                    else [total * np.kron(C, IL), mask * Ipy[i]]
                )

                # Fission operator
                self._Fy.append(
                    [nu_fission, F_Intg, Ipy[i] * bcy[i] * mask]
                    if num_groups > 1
                    else [nu_fission * F_Intg, Ipy[i] * bcy[i] * mask]
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
                            self._wx * eval_legendre(l, self._mu),
                        ),
                    )

                    # Scattering operator
                    self._Sx.append(
                        [scatter_gtg, S_Intg, Ipx[i] * bcx[i] * mask]
                        if num_groups > 1
                        else [scatter_gtg * S_Intg, Ipx[i] * bcx[i] * mask]
                    )



        # Construct TT objects
        self._Hx = self._tensor_train(self._Hx)
        self._Fx = self._tensor_train(self._Fx)
        self._Sx = self._tensor_train(self._Sx)
        self._Hy = self._tensor_train(self._Hy)
        self._Fy = self._tensor_train(self._Fy)
        self._Sy = self._tensor_train(self._Sy)
        self._Hz = self._tensor_train(self._Hz)
        self._Fz = self._tensor_train(self._Fz)
        self._Sz = self._tensor_train(self._Sz)

        # Convert to QTT if requested
        if self._tt_fmt == "qtt":
            self._Hx = self.tt2qtt(self._Hx)
            self._Fx = self.tt2qtt(self._Fx)
            self._Sx = self.tt2qtt(self._Sx)
            self._Hy = self.tt2qtt(self._Hy)
            self._Fy = self.tt2qtt(self._Fy)
            self._Sy = self.tt2qtt(self._Sy)
            self._Hz = self.tt2qtt(self._Hz)
            self._Fz = self.tt2qtt(self._Fz)
            self._Sz = self.tt2qtt(self._Sz)

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

        return np.real(k)[0], psi / np.linalg.norm(psi, 2)

    def solve_matrix_power(self):
        """
        Solve SN using power iteration with full matrices.
        """
        psi_old = np.random.rand(
            (self._geometry.num_nodes + 1)
            * self._num_ordinates
            * self._xs_server.num_groups
        ).reshape((-1, 1))
        psi_old /= np.linalg.norm(psi_old, 2)

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
            err = np.linalg.norm(psi_new - psi_old, ord=2) / np.linalg.norm(
                psi_old, ord=2
            )

            # Return if tolerance is met
            if err < self._tol:
                return k_new, psi_new.flatten() / np.linalg.norm(psi_new.flatten(), 2)

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

        raise RuntimeError(
            f"Maximum number of power iteration ({self._max_iter})"
            + f" without convergence to err < {self._tol}"
        )

    # ==============================================================================
    # TT solvers (scikit_tt and power iteration implementation)

    def solve_TT_scikit_als(self, ranks=None, repeats=10):
        """
        SN using TT with ALS/generalized eigenvalue solver.
        """
        # Get maximum ranks for each core for possible convergence
        if ranks == None:
            ranks = (
                np.array([self._H.ranks, self._S.ranks, self._F.ranks], dtype=int)
                .max(axis=0)
                .tolist()
            )

        psi = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=ranks)
        psi *= 1 / psi.norm(2)

        # Run ALS GES
        k, psi, _ = evp.als(
            self._F, psi, operator_gevp=(self._H - self._S), repeats=repeats
        )
        psi = psi.full().flatten() / psi.norm(2)

        if np.sum(psi) < 0:
            psi = -psi

        return k, psi

    def solve_TT_power(
        self,
        method="als",
        ranks=None,
        threshold=1e-8,
        start_max_rank=4,
        max_rank=None,
        verbose=False,
    ):
        """
        Solve SN using power iteration with TT ALS or MALS
        """
        assert method == "als" or method == "mals"

        # Get maximum ranks for each core for possible convergence
        if method == "als" and ranks == None:
            ranks = (
                np.array([self._H.ranks, self._S.ranks, self._F.ranks], dtype=int)
                .max(axis=0)
                .tolist()
            )

        elif method == "mals" and max_rank == None:
            # Prepare adaptive rank algorithm
            ranks = start_max_rank
            max_rank = np.array(
                [self._H.ranks, self._S.ranks, self._F.ranks], dtype=int
            ).max()

        # Initial guess for psi and k
        psi_old = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=ranks)
        psi_old *= 1 / psi_old.norm(2)

        k_old = np.random.rand(1)[0]
        err_old = 1.0

        # Run initial 5 iteration of ALS if MALS is chosen
        for i in range(self._max_iter if method == "als" else 5):
            psi_new = sle.als(
                self._H,
                psi_old,
                (self._S + 1 / k_old * self._F).dot(psi_old),
            )

            # Compute new eigenvalue and eigenvector L2 error
            k_new = (
                k_old * (self._F.dot(psi_new)).norm(1) / (self._F.dot(psi_old)).norm(1)
            )
            err_new = (psi_new - psi_old).norm(2) / psi_old.norm(2)

            # Return if tolerance is met
            if err_new < self._tol:
                return k_new, psi_new.full().flatten() / psi_new.norm(2)

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)
            err_old = copy.deepcopy(err_new)

        if method == "mals":
            kwargs = {"threshold": threshold, "max_rank": start_max_rank}

            # Get polynomial coefficients for rank
            self._poly = np.polyfit(
                np.log([err_old, self._tol * 100]),
                [start_max_rank, max_rank],
                1,
            )

            for i in range(self._max_iter - 5):
                psi_new = sle.mals(
                    self._H,
                    psi_old,
                    (self._S + 1 / k_old * self._F).dot(psi_old),
                    **kwargs,
                )

                # Compute new eigenvalue and eigenvector L2 error
                k_new = (
                    k_old
                    * (self._F.dot(psi_new)).norm(1)
                    / (self._F.dot(psi_old)).norm(1)
                )
                err_new = (psi_new - psi_old).norm(2) / psi_old.norm(2)

                # New rank calculation (based on y = A + B * log(x))
                kwargs["max_rank"] = np.round(
                    self._poly[0] * np.log(err_new) + self._poly[1]
                ).astype(int)

                # Return if tolerance is met
                if err_new < self._tol:
                    return k_new, psi_new.full().flatten() / psi_new.norm(2)
                elif err_new > err_old and verbose:
                    print(
                        "Warning: Error increased last iteration, convergence in question"
                    )

                # Copy results for next iteration
                psi_old = copy.deepcopy(psi_new)
                k_old = copy.deepcopy(k_new)
                err_old = copy.deepcopy(err_new)

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
                new_dims.append([2] * utils.get_degree(dim_size))

        else:
            for i in range(len(tt.row_dims)):
                if i in cores:
                    new_dims.append([2] * utils.get_degree(tt.row_dims[i]))
                else:
                    new_dims.append([tt.row_dims[i]])

        # Transform operator into QTT format with threshold
        # for SVD decomposition
        return tt.tt2qtt(new_dims, new_dims, threshold=self._qtt_threshold)

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
        utils.check_dim_size("ordinates", self._num_ordinates)
        utils.check_dim_size("spatial edges", self._geometry.num_nodes + 1)
        if self._xs_server.num_groups != 1:
            utils.check_dim_size("energy groups", self._xs_server.num_groups)

        # Get square quadrature set (in 1D this is Gauss-Legendre)
        # self._w, self._mu = self._gauss_legendre(self._num_ordinates)
        self._mu, self._eta, self._xi, self._wx, self._wy, self._wz \
            = self._3D_chebyshev_quad_set(self._num_ordinates)

        # Construct operator tensors
        self._construct_tensor_trains()

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
    
    @staticmethod
    def _2D_chebyshev_quad_set(self,n):
        # Gauss-Legendre nodes and weights for each dimension
        #ensure power of 2 for group and ordinates     
        self._check_dim_size("ordinates", self._num_ordinates)
        self._check_dim_size("spatial edges", self._geometry.num_nodes + 1)
        if self._xs_server.num_groups != 1:
            self._check_dim_size("energy groups", self._xs_server.num_groups)
        
        mu, w_x = roots_legendre(n)
        eta, w_y = roots_legendre(n)
        x, y = np.meshgrid(mu, eta, indexing='ij')
        
        # Compute the combined weights for the 2D grid
        weight_x = np.outer(w_x, w_y).reshape(n, n)
        weight_y = np.outer(w_x, w_y).reshape(n, n)        
        # normalize to 2*pi
        weight_x *= (1/4) / np.sum(weight_x)
        weight_y *= (1/4) / np.sum(weight_y)
        
        return x, y, weight_x, weight_y
    
    @staticmethod
    def _3D_chebyshev_quad_set(self,n):
        self._check_dim_size("ordinates", self._num_ordinates)
        self._check_dim_size("spatial edges", self._geometry.num_nodes + 1)
        if self._xs_server.num_groups != 1:
            self._check_dim_size("energy groups", self._xs_server.num_groups)
    
        mu, w_x = roots_legendre(n)
        eta, w_y = roots_legendre(n)
        xi, w_z = roots_legendre(n)
        x, y, z = np.meshgrid(mu, eta, xi, indexing='ij')
        
        # Compute the combined weights for the 2D grid
        weight_x = np.outer(w_y, w_z).reshape(n, n)
        weight_y = np.outer(w_x, w_z).reshape(n, n)
        weight_z = np.outer(w_x, w_y).reshape(n, n)   
        # normalize to 2*pi
        weight_x *= (1/8) / np.sum(weight_x)
        weight_y *= (1/8) / np.sum(weight_y)
        weight_z *= (1/8) / np.sum(weight_y)
        
        return x, y, z, weight_x, weight_y, weight_z

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
