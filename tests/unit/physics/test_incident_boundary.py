import pytest
import torch
from igakit.cad import refine, line

from ttnte import mpi_context
from ttnte.xs import Server, Material
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    DGTransportAssemblerConfig,
    DIGAFirstOrderTransportAssembler1D,
    FixedSource,
)
from ttnte.math import QuadratureSet1D

test_params = [
    ("cpu", torch.float64),
]


def _pure_absorber(device, dtype):
    """A 1-group, non-fissile material -- fixed/incident sources on a fissile fill are
    rejected at assemble() time (combined fixed-source + fission/subcritical-
    multiplication problems aren't supported yet), so these tests need a genuinely non-
    fissile material rather than the fissile pu239 benchmark."""
    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=dtype, device=device)
    mat.total = torch.tensor([0.32640], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(1, dtype=dtype, device=device)
    mat.fission = torch.zeros(1, dtype=dtype, device=device)
    mat.absorption = torch.tensor([0.101184], dtype=dtype, device=device)
    mat.scatter_gtg = torch.tensor([[[0.225216]]], dtype=dtype, device=device)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _make_slab(device, dtype, configure_boundary):
    """Build a single-patch 1-D slab (purely-absorbing, 1-group), letting
    `configure_boundary(patch)` set up the x_min face before the mesh is
    connected/finalized.

    x_max is left UNKNOWN and auto-assigned VACUUM by `mesh.connect()`.
    """
    mpi_context.init()
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    fill, xs_server = _pure_absorber(device, dtype)

    rc = 2.256751
    patch = Patch.from_igakit(
        refine(line((-rc, 0), (rc, 0)), 10, 3),
        device=device,
        dtype=dtype,
        fill=fill,
    )
    configure_boundary(patch)

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    assembler = DIGAFirstOrderTransportAssembler1D(patch, qset, xs_server, config)
    assembler.assemble()
    return assembler


@pytest.mark.parametrize("device, dtype", test_params)
def test_incident_inflow_operator_matches_internal(device, dtype):
    """BoundaryType.INCIDENT is meant to reuse INTERNAL's own inflow-operator
    construction verbatim (see assemble_inflow_boundary_operator() /
    assemble_boundary_operators() in dg_first_order_transport_backends.cpp) -- the only
    difference is where the RHS value comes from.

    Building the same face as INCIDENT vs. as a (disconnected) INTERNAL boundary
    must therefore produce bit-identical inflow operators.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    Q = 1.0

    def as_incident(patch):
        patch.set_boundary_source(
            0, False, FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
        )

    def as_internal(patch):
        patch.set_boundary_type(0, False, BoundaryType.INTERNAL)

    incident_assembler = _make_slab(device, dtype, as_incident)
    internal_assembler = _make_slab(device, dtype, as_internal)

    incident_bin = incident_assembler.inflow_ops[0]
    internal_bin = internal_assembler.inflow_ops[0]

    assert incident_bin.defined() and internal_bin.defined()
    torch.testing.assert_close(
        incident_bin.as_tt().to_dense(), internal_bin.as_tt().to_dense()
    )


@pytest.mark.parametrize("device, dtype", test_params)
def test_incident_boundary_leaves_lhs_unchanged_from_vacuum(device, dtype):
    """A prescribed incident flux is a known RHS-only value, not this patch's own
    unknown -- it must be excluded from the LHS exactly the way VACUUM is
    (assemble_inflow_boundary_operator() returns nullopt for VACUUM, and the assembler's
    LHS loop now excludes INCIDENT alongside INTERNAL), so the assembled interior
    operator must be identical whether the face is VACUUM or INCIDENT."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    def as_incident(patch):
        patch.set_boundary_source(
            0, False, FixedSource(isotropic_strength=torch.tensor([2.5], dtype=dtype))
        )

    def as_vacuum(patch):
        patch.set_boundary_type(0, False, BoundaryType.VACUUM)

    incident_assembler = _make_slab(device, dtype, as_incident)
    vacuum_assembler = _make_slab(device, dtype, as_vacuum)

    def lhs(assembler):
        A = assembler.interior_loss_op.as_tt() - assembler.scatter_op.as_tt()
        for op in assembler.outflow_ops:
            A = A + op.as_tt()
        return A.round(1e-10, 10**9).to_dense()

    torch.testing.assert_close(lhs(incident_assembler), lhs(vacuum_assembler))

    # The VACUUM face contributes no inflow operator at all; the INCIDENT
    # face's inflow operator is defined but purely RHS-side.
    assert not vacuum_assembler.inflow_ops[0].defined()
    assert incident_assembler.inflow_ops[0].defined()


@pytest.mark.parametrize("device, dtype", test_params)
def test_incident_source_is_finite_and_scales_linearly(device, dtype):
    """The composed RHS contribution (mv(inflow_op, incident_state), folded
    into DGFirstOrderTransportAssembler::source_) must be finite and scale
    linearly with the prescribed strength -- a basic sanity check on
    assemble_incident_source()'s isotropic branch."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    device = torch.device(device)

    def make(Q):
        def as_incident(patch):
            patch.set_boundary_source(
                0, False, FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
            )

        return _make_slab(device, dtype, as_incident)

    Q = 1.7
    assembler_q = make(Q)
    assembler_2q = make(2.0 * Q)

    source_q = assembler_q.source.to_dense()
    source_2q = assembler_2q.source.to_dense()

    assert torch.isfinite(source_q).all()
    assert source_q.abs().sum().item() > 0.0
    torch.testing.assert_close(source_2q, 2.0 * source_q)
