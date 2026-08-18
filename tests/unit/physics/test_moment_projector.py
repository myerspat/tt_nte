import torch
import pytest
from igakit.cad import line
from igakit import cad

from ttnte.cad.surfaces import circle
from ttnte.cad import Patch
from ttnte.physics import (
    DGTransportAssemblerConfig,
    TTDIGAFirstOrderTransportBackend1D,
    TTDIGAFirstOrderTransportBackend2D,
)
from ttnte.xs import Server, Material
from ttnte.math import ProductQuadrature, QuadratureSet1D
from ttnte.linalg import TTEngine, State, mv

dtype = torch.float64
device = "cpu"


def _filler_xs_and_server():
    mat = Material("mat")
    mat.total = torch.ones((1,), dtype=dtype, device=device)
    mat.scatter_gtg = torch.zeros((1, 1, 1), dtype=dtype, device=device)
    mat.finalize()

    server = Server()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _random_rank1_state(mode_sizes, seed):
    """A rank-1 TT state: cores[k] shape (1, mode_sizes[k], 1, 1). Sufficient
    to exercise the moment projector's angular action directly -- the full
    element (i0, i1, ...) is simply the product of the per-core values, so
    ground-truth moments can be computed independently below with plain
    torch einsums rather than through any of the code under test."""
    g = torch.Generator().manual_seed(seed)
    cores = [torch.rand((1, m, 1, 1), dtype=dtype, generator=g) for m in mode_sizes]
    return State(TTEngine(cores)), cores


def _moment(cores_dense, w_polar, w_azim, b_polar, b_azim):
    """Ground truth moment field (x, y, energy), computed independently of
    assemble_moment_projector()/assemble_ordinates() via a direct weighted contraction
    over the angular axes."""
    return torch.einsum(
        "pa...,p,a->...", cores_dense, w_polar * b_polar, w_azim * b_azim
    )


def test_moment_projector_2d_matches_independent_ground_truth():
    mat_label, server = _filler_xs_and_server()

    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 3, 2)
    qset.to_(torch.device(device), dtype)

    patch = Patch.from_igakit(
        cad.refine(circle(1.0), 2, 1), device=device, dtype=dtype, fill=mat_label
    )

    config = DGTransportAssemblerConfig()
    backend = TTDIGAFirstOrderTransportBackend2D(patch, qset, server, config)

    quads = qset.get_quads()
    w_polar, mu = quads[0].get_weights(), quads[0].get_points()
    w_azim, gamma = quads[1].get_weights(), quads[1].get_points()

    nx = patch.get_ctrlpts_size(0)
    ny = patch.get_ctrlpts_size(1)
    mode_sizes = [
        quads[0].get_num_dofs(),
        quads[1].get_num_dofs(),
        nx,
        ny,
        server.num_groups,
    ]

    psi, cores = _random_rank1_state(mode_sizes, seed=0)
    dense = cores[0].reshape(-1)
    for c in cores[1:]:
        dense = torch.einsum("...,n->...n", dense, c.reshape(-1))

    basis = {
        "P0": (torch.ones_like(mu), torch.ones_like(gamma)),
        "P1z": (mu, torch.ones_like(gamma)),
        "P1x": (torch.sqrt(1 - mu**2), torch.cos(gamma)),
        "P1y": (torch.sqrt(1 - mu**2), torch.sin(gamma)),
    }

    P = backend.assemble_moment_projector(1)
    psi_macro = mv(P, psi)
    dense_macro = psi_macro.to_dense().reshape(dense.shape)

    for name, (b_p, b_a) in basis.items():
        truth = _moment(dense, w_polar, w_azim, b_p, b_a)
        after = _moment(dense_macro, w_polar, w_azim, b_p, b_a)
        torch.testing.assert_close(after, truth, atol=1e-10, rtol=1e-8), name

    # Idempotent: re-projecting the macro part is a no-op.
    psi_macro_twice = mv(P, psi_macro)
    torch.testing.assert_close(
        psi_macro_twice.to_dense(), psi_macro.to_dense(), atol=1e-10, rtol=1e-8
    )

    # The remainder has ~zero moments -- it's orthogonal to every basis
    # function by construction.
    dense_remainder = dense - dense_macro
    for name, (b_p, b_a) in basis.items():
        residual = _moment(dense_remainder, w_polar, w_azim, b_p, b_a)
        assert residual.abs().max().item() < 1e-9, name


def test_moment_projector_1d_matches_independent_ground_truth():
    mat_label, server = _filler_xs_and_server()

    qset = QuadratureSet1D.gauss_legendre(8)
    qset.to_(torch.device(device), dtype)

    curve = line((0, 0), (0, 1))
    c = Patch.from_igakit(curve, device=device, dtype=dtype, fill=mat_label)

    config = DGTransportAssemblerConfig()
    backend = TTDIGAFirstOrderTransportBackend1D(c, qset, server, config)

    w_polar, mu = qset.get_weights(), qset.get_points()
    nx = c.get_ctrlpts_size(0)
    mode_sizes = [qset.get_num_dofs(), nx, server.num_groups]

    psi, cores = _random_rank1_state(mode_sizes, seed=1)
    dense = cores[0].reshape(-1)
    for cc in cores[1:]:
        dense = torch.einsum("...,n->...n", dense, cc.reshape(-1))

    basis = {"P0": torch.ones_like(mu), "P1z": mu}

    P = backend.assemble_moment_projector(1)
    psi_macro = mv(P, psi)
    dense_macro = psi_macro.to_dense().reshape(dense.shape)

    for name, b_p in basis.items():
        truth = torch.einsum("p...,p->...", dense, w_polar * b_p)
        after = torch.einsum("p...,p->...", dense_macro, w_polar * b_p)
        torch.testing.assert_close(after, truth, atol=1e-10, rtol=1e-8), name


def test_moment_projector_rejects_unsupported_order():
    mat_label, server = _filler_xs_and_server()
    qset = ProductQuadrature.gauss_legendre_chebyshev(4, 3, 2)
    qset.to_(torch.device(device), dtype)
    patch = Patch.from_igakit(
        cad.refine(circle(1.0), 2, 1), device=device, dtype=dtype, fill=mat_label
    )
    backend = TTDIGAFirstOrderTransportBackend2D(
        patch, qset, server, DGTransportAssemblerConfig()
    )
    with pytest.raises(RuntimeError):
        backend.assemble_moment_projector(2)
