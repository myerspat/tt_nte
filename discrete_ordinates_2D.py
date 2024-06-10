"""
discrete_ordinates.py
2D and 3D attempt 6/4/24
"""
import copy
import numpy as np
import scikit_tt.solvers.evp as evp
import scikit_tt.solvers.sle as sle
import scikit_tt.tensor_train as tt
import scipy.sparse as sp
from scikit_tt import TT
from scipy.sparse.linalg import eigs, inv
from scipy.special import roots_legendre


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
        
    def _num_dims(self, dim):
        assert self.dim.upper() in {'1D', '2D', '3D'},\
            "Input string must be '1D', '2D', or '3D'"
        
        if self.dim.lower() == '1d':
            num_dims = 2
        elif self.dim.lower() == '2d':
            num_dims = 4
        elif self.dim.lower() == '3d':
            num_dims = 8
        else:
            return False
            
        return num_dims
    
    def _construct_tensor_trains(self):
        """
        Construct operaotr tensor trains as described in
        LANL TT paper.
        """
        # change number of dimensions here. stick to 2d right now
        num_dims = 2
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
        bc[0][0, 0] = 0
        bc[1][-1, 0] = 0


        # Interpolation
        Ip = [1 / 2 * (d + d_p), 1 / 2 * (d + d_m)]

        # Differentiation matrix
        # potentially change to dy and dz as well
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
        # 2D - 4 (quadrants)
        # 3D - 8 (octants)
        # init operation matrices in each dimension
        self._Hx = []
        self._Fx = []
        self._Sx = []
        self._Hy = []
        self._Fy = []
        self._Sy = []
        # self._Hz = []
        # self._Fz = []
        # self._Sz = []
        
        bcs = [self._geometry.left_bc, self._geometry.right_bc, \
               self._geometry.top_bc, self._geometry.bottom_bc, \
                #self.geometry.into_bc, self.geometry.outof_bc\
                    ]

        for i in range(num_dims):
            C = np.zeros((num_dims, num_dims), dtype=float)
            C[i, i] = 1.0
            
            # Angular point matrix
            Qmu = np.kron(C, (-1) ** (i) * np.diag(self._mu))
            Qet = np.kron(C, (-1) ** (i) * np.diag(self._eta))
            #Qxi = np.kron(C, (-1) ** (i) * np.diag(self._xi))

            # Streaming operator
            self._Hx.append([Ig, Qmu, D[i]] if num_groups > 1 else [Qmu, D[i]])
            self._Hy.append([Ig, Qet, D[i]] if num_groups > 1 else [Qet, D[i]])
            #self._Hx.append([Ig, Qxi, D[i]] if num_groups > 1 else [Qxi, D[i]])

            # Integral operator
            A = np.zeros((num_dims, num_dims), dtype=float)
            A[i, :] = 1 
            Intg_x = np.kron(A, np.outer(np.ones(self._w_x.size), self._w_x))
            Intg_y = np.kron(A, np.outer(np.ones(self._w_y.size), self._w_y))
            #Intg_z = np.kron(A, np.outer(np.ones(self._w_z.size), self._w_z))

            # Add reflection
            if bcs[i] == "reflective":
                Ip_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                Ip_ref[-i, -i] = 1 / 2

                D_ref = np.zeros((num_nodes + 1, num_nodes + 1))
                D_ref[-i, -i] = 1 / dx[0,] if i == 0 else 1 / dx[-1,]

                C_ref = np.zeros((2, 2), dtype=float)
                C_ref[i, (i + 1) % 2] = 1

                Qmu_ref = np.kron(C_ref, -np.diag(self._mu))
                Qet_ref = np.kron(C_ref, -np.diag(self._eta))
                #Qxi_ref = np.kron(C_ref, -np.diag(self._xi))

                # Streaming operator (reflective)
                self._Hx.append(
                    [Ig, Qmu_ref, D_ref] if num_groups > 1 else [Qmu_ref, D_ref]
                )
                self._Hy.append(
                    [Ig, Qet_ref, D_ref] if num_groups > 1 else [Qet_ref, D_ref]
                )
                # self._Hz.append(
                #     [Ig, Qxi_ref, D_ref] if num_groups > 1 else [Qxi_ref, D_ref]
                # )

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
                # self._Hz.append(
                #     [
                #         np.diag(self._xs_server.total(region.material)),
                #         np.kron(C_ref, -IL),
                #         Ip_ref,
                #     ]
                #     if num_groups > 1
                #     else [
                #         self._xs_server.total(region.material) * np.kron(C_ref, -IL),
                #         Ip_ref,
                #     ]
                # )
                assert 1 == 1

            # Iterate through regions in the problem from left to right
            for region in self._geometry.unique_regions:
                # Define XS matrices
                mat = region.material

                total = np.squeeze(np.diag(self._xs_server.total(mat)))
                scatter_gtg = np.squeeze(self._xs_server.scatter_gtg(mat))
                nu_fission = np.squeeze(np.outer(
                    self._xs_server.chi, self._xs_server.nu_fission(mat)))

                # Region mask for spatial dependence
                mask = self._geometry.region_mask(region)
                mask = (
                    np.concatenate((mask[[0],], mask))
                    if i == 0
                    else np.concatenate((mask, mask[[-1],]))
                )

                # Total interaction operator
                self._Hx.append(
                    [total, np.kron(C, IL), mask * Ip[i]]
                    if num_groups > 1
                    else [total * np.kron(C, IL), mask * Ip[i]]
                )
                self._Hy.append(
                    [total, np.kron(C, IL), mask * Ip[i]]
                    if num_groups > 1
                    else [total * np.kron(C, IL), mask * Ip[i]]
                )
                # self._Hz.append(
                #     [total, np.kron(C, IL), mask * Ip[i]]
                #     if num_groups > 1
                #     else [total * np.kron(C, IL), mask * Ip[i]]
                # )

                # Fission operator
                self._Fx.append(
                    [nu_fission, Intg_x, Ip[i] * bc[i] * mask]
                    if num_groups > 1
                    else [nu_fission * Intg_x, Ip[i] * bc[i] * mask]
                )
                self._Fy.append(
                    [nu_fission, Intg_y, Ip[i] * bc[i] * mask]
                    if num_groups > 1
                    else [nu_fission * Intg_y, Ip[i] * bc[i] * mask]
                )
                # self._Fz.append(
                #     [nu_fission, Intg_z, Ip[i] * bc[i] * mask]
                #     if num_groups > 1
                #     else [nu_fission * Intg_z, Ip[i] * bc[i] * mask]
                # )

                # Scattering operator
                self._Sx.append(
                    [scatter_gtg, Intg_x, Ip[i] * bc[i] * mask]
                    if num_groups > 1
                    else [scatter_gtg * Intg_x, Ip[i] * bc[i] * mask]
                )
                self._Sy.append(
                    [scatter_gtg, Intg_y, Ip[i] * bc[i] * mask]
                    if num_groups > 1
                    else [scatter_gtg * Intg_y, Ip[i] * bc[i] * mask]
                )

                # self._Sz.append(
                #     [scatter_gtg, Intg_z, Ip[i] * bc[i] * mask]
                #     if num_groups > 1
                #     else [scatter_gtg * Intg_z, Ip[i] * bc[i] * mask]
                # )
                assert 1 == 1


        # Construct TT objects
        self._Hx = self._tensor_train(self._Hx)
        self._Fx = self._tensor_train(self._Fx)
        self._Sx = self._tensor_train(self._Sx)
        self._Hy = self._tensor_train(self._Hy)
        self._Fy = self._tensor_train(self._Fy)
        self._Sy = self._tensor_train(self._Sy)
        # self._Hz = self._tensor_train(self._Hz)
        # self._Fz = self._tensor_train(self._Fz)
        # self._Sz = self._tensor_train(self._Sz)

        # Convert to QTT if requested
        if self._tt_fmt == "qtt":
            self._Hx = self.tt2qtt(self._Hx)
            self._Fx = self.tt2qtt(self._Fx)
            self._Sx = self.tt2qtt(self._Sx)
            self._Hy = self.tt2qtt(self._Hy)
            self._Fy = self.tt2qtt(self._Fy)
            self._Sy = self.tt2qtt(self._Sy)
            # self._Hz = self.tt2qtt(self._Hz)
            # self._Fz = self.tt2qtt(self._Fz)
            # self._Sz = self.tt2qtt(self._Sz)

    # ==============================================================================
    # Full matrix solvers

    def solve_ges(self):
        """
        Solve SN using scipy.sparse.linalg.eigs.
        """
        # Get operators in CSC format
        Hx = self.Hx("csc")
        Sx = self.Sx("csc")
        Fx = self.Fx("csc")
        Hy = self.Hy("csc")
        Sy = self.Sy("csc")
        Fy = self.Fy("csc")
        # Hz = self.Hz("csc")
        # Sz = self.Sz("csc")
        # Fz = self.Fz("csc")

        kx, psi_x = eigs(Fx, 1, Hx - Sx)
        psix = np.real(psi_x).flatten()
        ky, psi_y = eigs(Fy, 1, Hy - Sy)
        psiy = np.real(psi_y).flatten()
        # kz, psi_z = eigs(Fz, 1, Hz - Sz)
        # psiz = np.real(psi_z).flatten()
        
        k = np.average(kx,ky)
        psi = np.average(psix,psiy)
        #k = np.average(kx,ky,kz)
        #psi = np.average(psix,psiy,psiz)

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
        k_old = float(np.random.rand(1))

        err = 1.0

        # Get operators in CSC format
        Hx_inv = inv(self.Hx("csc"))
        Sx = self.Sx("csc")
        Fx = self.Fx("csc")
        Hy_inv = inv(self.Hy("csc"))
        Sy = self.Sy("csc")
        Fy = self.Fy("csc")
        # Hz_inv = inv(self.H("csc"))
        # Sz = self.S("csc")
        # Fz = self.F("csc")


        for _ in range(self._max_iter):
            psi_new_x = Hx_inv.dot((Sx + 1 / k_old * Fx).dot(psi_old))
            psi_new_y = Hy_inv.dot((Sy + 1 / k_old * Fy).dot(psi_old))

            # Compute new eigenvalue and eigenvector L2 error
            k_new_x = k_old * np.sum(Fx.dot(psi_new_x)) / np.sum(Fx.dot(psi_old))
            k_new_y = k_old * np.sum(Fy.dot(psi_new_y)) / np.sum(Fy.dot(psi_old))
            
            psi_new = np.average(psi_new_x,psi_new_y)
            k_new = (k_new_x + k_new_y)/2
            
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
        psix = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=rank)
        psiy = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=rank)

        # Run ALS GES
        kx, psix, _ = evp.als(
            self._Fx, psix, operator_gevp=(self._Hx - self._Sx), conv_eps=self._tol
        )
        ky, psiy, _ = evp.als(
            self._Fy, psiy, operator_gevp=(self._Hy - self._Sy), conv_eps=self._tol
        )
        k = (kx+ky)/2
        psi = (psix+psiy)/2
        psi = psi.full().flatten() / psi.norm()

        if np.sum(psi) < 0:
            psi = -psi

        return k, psi

    def solve_TT_power(self, method="als", rank=4, threshold=1e-12, max_rank=np.infty):
        """
        Solve SN using power iteration with TT ALS or MALS
        """
        psi_old = tt.rand(self._H.row_dims, [1] * self._H.order, ranks=rank)

        k_old = float(np.random.rand(1))
        err = 1.0

        solver = sle.als if method == "als" else sle.mals
        kwargs = (
            {} if method == "als" else {"threshold": threshold, "max_rank": max_rank}
        )
        
        # x dimension sweep
        for i in range(self._max_iter):
            psi_new = solver(
                self._Hx,
                psi_old,
                (self._Sx + 1 / k_old * self._Fx).dot(psi_old),
                **kwargs,
            )

            # Compute new eigenvalue and eigenvector L2 error
            k_new = (
                k_old * (self._Fx.dot(psi_new)).norm() / (self._Fx.dot(psi_old)).norm()
            )
            err = (psi_new - psi_old).norm(p=2)

            if err < self._tol:
                break

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)
            kx = k_new
            psi_x = psi_new
        
        # y dimension
        for i in range(self._max_iter):
            psi_new = solver(
                self._Hy,
                psi_old,
                (self._Sy + 1 / k_old * self._Fy).dot(psi_old),
                **kwargs,
            )

            # Compute new eigenvalue and eigenvector L2 error
            k_new = (
                k_old * (self._Fy.dot(psi_new)).norm() / (self._Fy.dot(psi_old)).norm()
            )
            err = (psi_new - psi_old).norm(p=2)

            if err < self._tol:
                break

            # Copy results for next iteration
            psi_old = copy.deepcopy(psi_new)
            k_old = copy.deepcopy(k_new)

        raise RuntimeError(
            f"Maximum number of power iteration ({self._max_iter})"
            + f" without convergence to err < {self._tol}"
        )
        
        k = (kx+k_new)/2
        psi = (psi_x + psi_new)/2
        
        return k, psi.full().flatten() / psi.norm()

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

    def tt2qtt(self, tt):
        """
        Transform TT formatted operator to QTT format.
        """
        # Reshape and permute dims to list of [2] * l_k
        new_dims = []
        for dim_size in tt.row_dims:
            new_dims.append([2] * self._get_degree(dim_size))

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
        #self._w, self._mu = self._gauss_legendre(self._num_ordinates)
        self._mu, self._eta, self._w_x, self._w_y = \
            self._2D_chebyshev_quad_set(self,self._num_ordinates)

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

    # ==============================================================================
    # Quadrature sets

    # @staticmethod
    # def _gauss_legendre(N):
    #     """
    #     Gauss Legendre quadrature set. Only given for positive half space.
    #     """
    #     mu, w = np.polynomial.legendre.leggauss(N)
    #     w = w[: int(mu.size / 2)] / 2
    #     mu = np.abs(mu[: int(mu.size / 2)])

    #     assert np.round(np.sum(w), 1) == 0.5

    #     return w, mu
    
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
        weight_x *= (2*np.pi) / np.sum(weight_x)
        weight_y *= (2*np.pi) / np.sum(weight_y)
        
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
        weight_x *= (4*np.pi) / np.sum(weight_x)
        weight_y *= (4*np.pi) / np.sum(weight_y)
        weight_z *= (4*np.pi) / np.sum(weight_y)
        
        return x, y, z, weight_x, weight_y, weight_z

    # =================================================================
    # Getters
    def Hx(self, fmt="tt"):
        return self._format_tt(self._Hx, fmt)

    def Sx(self, fmt="tt"):
        return self._format_tt(self._Sx, fmt)

    def Fx(self, fmt="tt"):
        return self._format_tt(self._Fx, fmt)
    
    def Hy(self, fmt="tt"):
        return self._format_tt(self._Hy, fmt)

    def Sy(self, fmt="tt"):
        return self._format_tt(self._Sy, fmt)

    def Fy(self, fmt="tt"):
        return self._format_tt(self._Fy, fmt)
    
    # def Hz(self, fmt="tt"):
    #     return self._format_tt(self._Hz, fmt)

    # def Sz(self, fmt="tt"):
    #     return self._format_tt(self._Sz, fmt)

    # def Fz(self, fmt="tt"):
    #     return self._format_tt(self._Fz, fmt)

    @property
    def num_ordinates(self):
        return self._num_ordinates
