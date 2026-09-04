import pytest
import torch
from igakit.cad import refine, line, circle, ruled

from ttnte import mpi_context
from ttnte.xs import Server, Material
from ttnte.xs.benchmarks import pu239
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import BoundaryType, DGTransportAssemblerConfig, FixedSource
from ttnte.math import QuadratureSet1D, ProductQuadrature
from ttnte.driver import IGATransportDriver1D, IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    IGADDSolver,
    ExecMode,
    CommMode,
)


def _pure_absorber(sigma_t, device, dtype):
    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=dtype, device=device)
    mat.total = torch.tensor([sigma_t], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(1, dtype=dtype, device=device)
    mat.fission = torch.zeros(1, dtype=dtype, device=device)
    mat.absorption = torch.tensor([sigma_t], dtype=dtype, device=device)
    mat.scatter_gtg = torch.zeros((1, 1, 1), dtype=dtype, device=device)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _pure_scatterer(sigma_t, sigma_s, device, dtype):
    server = Server()
    mat = Material("Scatterer")
    mat.chi = torch.zeros(1, dtype=dtype, device=device)
    mat.total = torch.tensor([sigma_t], dtype=dtype, device=device)
    mat.nu_fission = torch.zeros(1, dtype=dtype, device=device)
    mat.fission = torch.zeros(1, dtype=dtype, device=device)
    mat.absorption = torch.tensor([sigma_t - sigma_s], dtype=dtype, device=device)
    mat.scatter_gtg = torch.tensor([[[sigma_s]]], dtype=dtype, device=device)
    mat.finalize()
    server.add_material(mat)
    server.finalize()
    return mat.label, server


def _solve_fixed_source(mesh, server, qset, dd_tol=1e-7, amen_eps=1e-8):
    device = torch.device("cpu")
    dtype = torch.float64
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    driver = IGATransportDriver1D(mesh, server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=dd_tol,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=amen_eps, kickrank=4, local_iterations=80, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)
    result = driver.solve_fixed_source(
        dd_solver, tol=1e-6, max_iter=50, verbose=False, clear_assemblers=False
    )
    return driver, result


def _solve_fixed_source_clear_assemblers(
    mesh, server, qset, dd_tol=1e-7, amen_eps=1e-8
):
    """Same as _solve_fixed_source(), but with the default clear_assemblers=True, so
    driver.get_assemblers() comes back empty and global_balance()/ patch_balance_table()
    must build their own assemblers on demand."""
    device = torch.device("cpu")
    dtype = torch.float64
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = 0
    config.cross_jacobian_inverse = True

    driver = IGATransportDriver1D(mesh, server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=dd_tol,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=amen_eps, kickrank=4, local_iterations=80, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)
    result = driver.solve_fixed_source(dd_solver, tol=1e-6, max_iter=50, verbose=False)
    return driver, result


def test_volumetric_source_balance_closes():
    """Single-patch, pure-absorber slab, uniform volumetric source Q, VACUUM
    on both ends: no scattering/fission, so the balance is trivial --
    fixed_source == absorption + leakage -- and fixed_source itself must
    equal exactly Q * length."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.75
    Q = 2.0
    L = 3.0

    fill, server = _pure_absorber(sigma_t, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 12, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.VACUUM)
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source(mesh, server, qset)

    gb = result.global_balance(driver.get_assemblers(), eps=1e-12, max_rank=10**9)
    torch.testing.assert_close(
        gb.fixed_source, torch.tensor([Q * L], dtype=dtype), rtol=1e-10, atol=1e-10
    )
    residual = gb.fixed_source - gb.absorption - gb.leakage
    assert (residual.abs() / gb.fixed_source).item() < 1e-3
    assert gb.fission_source.item() == 0.0
    assert gb.scatter_in.item() == 0.0
    assert gb.scatter_out.item() == 0.0
    assert gb.dd_residual.item() == 0.0


def test_global_balance_without_assemblers_rebuilds_them():
    """Same problem as test_volumetric_source_balance_closes(), but solved with the
    default clear_assemblers=True and global_balance() called with no `assemblers` at
    all -- it must build its own fresh assemblers on demand (via
    TransportSolution.make_assembler()) and still close the balance correctly,
    matching driver.get_assemblers() being empty."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.75
    Q = 2.0
    L = 3.0

    fill, server = _pure_absorber(sigma_t, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 12, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.VACUUM)
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source_clear_assemblers(mesh, server, qset)
    assert driver.get_assemblers() == {}

    gb = result.global_balance(eps=1e-12, max_rank=10**9)
    torch.testing.assert_close(
        gb.fixed_source, torch.tensor([Q * L], dtype=dtype), rtol=1e-10, atol=1e-10
    )
    residual = gb.fixed_source - gb.absorption - gb.leakage
    assert (residual.abs() / gb.fixed_source).item() < 1e-3


def test_incident_beam_balance_closes():
    """Single-patch pure-absorber slab with a prescribed INCIDENT beam:
    fixed_source (read off the assembled source_ state) must match the
    INCIDENT face's own `incoming` (they're the same physical quantity via
    two independent routes), and the whole-problem balance must close."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.75
    Q = 2.0
    L = 3.0

    fill, server = _pure_absorber(sigma_t, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 12, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.set_boundary_source(
        0, False, FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    )
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source(mesh, server, qset)

    assemblers = driver.get_assemblers()
    table = result.patch_balance_table(assemblers, eps=1e-12, max_rank=10**9)
    assert len(table.patches) == 1
    pb = table.patches[0]

    incident_face = next(f for f in pb.faces if f.type == BoundaryType.INCIDENT)
    vacuum_face = next(f for f in pb.faces if f.type == BoundaryType.VACUUM)
    assert incident_face.incoming is not None
    assert vacuum_face.incoming is None

    torch.testing.assert_close(
        pb.fixed_source, incident_face.incoming, rtol=1e-6, atol=1e-8
    )

    gb = result.global_balance(assemblers, eps=1e-12, max_rank=10**9)
    residual = gb.fixed_source - gb.absorption - gb.leakage
    assert (residual.abs() / gb.fixed_source).item() < 1e-3


def test_reflective_face_incoming_matches_outgoing():
    """A REFLECTIVE face's incoming and outgoing partial currents must match exactly
    (self-consistent by construction, a free consistency check), verified directly at
    the assembler/driver level -- no analytic solution needed, just internal consistency
    of the leakage functional."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.5
    sigma_s = 0.2
    L = 2.0

    fill, server = _pure_scatterer(sigma_t, sigma_s, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 10, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.source = FixedSource(isotropic_strength=torch.tensor([1.0], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.REFLECTIVE)
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source(mesh, server, qset)

    assemblers = driver.get_assemblers()
    table = result.patch_balance_table(assemblers, eps=1e-12, max_rank=10**9)
    pb = table.patches[0]
    reflective_face = next(f for f in pb.faces if f.type == BoundaryType.REFLECTIVE)
    torch.testing.assert_close(
        reflective_face.outgoing, reflective_face.incoming, rtol=1e-3, atol=1e-6
    )

    gb = result.global_balance(assemblers, eps=1e-12, max_rank=10**9)
    residual = gb.fixed_source + gb.scatter_in - gb.absorption
    residual = residual - gb.scatter_out - gb.leakage
    assert (residual.abs() / gb.fixed_source).item() < 1e-3


def test_scattering_balance_closes():
    """Single-patch slab with nonzero scattering and a volumetric source,
    VACUUM both ends: the full per-group balance
    (fixed_source + scatter_in == absorption + scatter_out + leakage) must
    close, exercising scatter_in/scatter_out (built from the isotropic/l=0
    scattering moment only) together rather than in isolation."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.5
    sigma_s = 0.2
    L = 2.0

    fill, server = _pure_scatterer(sigma_t, sigma_s, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (L, 0)), 10, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    patch.source = FixedSource(isotropic_strength=torch.tensor([1.0], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.VACUUM)
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source(mesh, server, qset)

    gb = result.global_balance(driver.get_assemblers(), eps=1e-12, max_rank=10**9)
    assert gb.scatter_in.item() > 0.0
    assert gb.scatter_out.item() > 0.0
    gain = gb.fixed_source + gb.scatter_in
    loss = gb.absorption + gb.scatter_out + gb.leakage
    assert ((gain - loss).abs() / gain).item() < 1e-3


def test_two_patch_internal_dd_residual_near_zero():
    """Two-patch INTERNAL coupling (mirrors test_slab.py's geometry): the two patches'
    independently, locally computed outgoing currents through their shared face must
    match closely -- the direct DD-conservation check ("is the distributed method losing
    particles")."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    fills, xs_server = pu239(num_groups=1, device=device, dtype=dtype)
    rc = 2.256751
    c0 = Patch.from_igakit(
        refine(line((-rc, 0), (0, 0)), 5, 3), device=device, dtype=dtype, fill=fills[0]
    )
    c1 = Patch.from_igakit(
        refine(line((0, 0), (rc, 0)), 5, 3), device=device, dtype=dtype, fill=fills[0]
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(c0)
    mesh.add_block(c1)
    mesh.connect()
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = True

    driver = IGATransportDriver1D(mesh, xs_server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=1e-7,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=2, eps=1e-8, kickrank=2, local_iterations=60, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)
    result = driver.solve_eigenvalue(
        dd_solver, tol=1e-6, max_iter=100, clear_assemblers=False, verbose=False
    )

    gb = result.global_balance(driver.get_assemblers(), eps=1e-10, max_rank=10**9)
    assert gb.dd_residual.item() < 1e-6


def test_two_patch_internal_leakage_closes_per_patch():
    """Two-patch, 1D pure-absorber slab, fixed source only in the first patch, VACUUM at
    both outer ends, INTERNAL where the patches meet: resolve_internal_faces() must fill
    in the shared face's `incoming` from the neighbor's own `outgoing` (checked.

    directly, both directions) and fold it into `leakage`, so each patch's own balance
    -- fixed_source + scatter_in == absorption + scatter_out + leakage -- closes on its
    own, including the second patch, which has no fixed_source of its own and would show
    a wildly wrong residual under the old (INTERNAL-excluded) leakage.
    """
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.75
    Q = 2.0
    L = 3.0

    fill, server = _pure_absorber(sigma_t, device, dtype)
    p0 = Patch.from_igakit(
        refine(line((0, 0), (L / 2, 0)), 8, 3), device=device, dtype=dtype, fill=fill
    )
    p1 = Patch.from_igakit(
        refine(line((L / 2, 0), (L, 0)), 8, 3), device=device, dtype=dtype, fill=fill
    )

    mesh = IGAMesh(mpi_context)
    mesh.add_block(p0)
    mesh.add_block(p1)
    mesh.connect()

    p0.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    p0.set_boundary_type(0, False, BoundaryType.VACUUM)
    p1.set_boundary_type(0, True, BoundaryType.VACUUM)
    mesh.finalize()

    qset = QuadratureSet1D.gauss_legendre(64)
    qset.to_(device, dtype)

    driver, result = _solve_fixed_source(mesh, server, qset)
    assemblers = driver.get_assemblers()

    table = result.patch_balance_table(assemblers, eps=1e-12, max_rank=10**9)
    assert len(table.patches) == 2

    internal_face = {
        pb.gid: next(f for f in pb.faces if f.type == BoundaryType.INTERNAL)
        for pb in table.patches
    }
    gid0, gid1 = sorted(internal_face.keys())
    face0, face1 = internal_face[gid0], internal_face[gid1]

    # incoming must now be resolved for INTERNAL faces, and match the
    # neighbor's own outgoing exactly -- resolve_internal_faces() just copies
    # it across the already-gathered table, no solve tolerance involved.
    assert face0.incoming is not None and face1.incoming is not None
    torch.testing.assert_close(face0.incoming, face1.outgoing, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(face1.incoming, face0.outgoing, rtol=1e-12, atol=1e-14)

    gb = result.global_balance(assemblers, eps=1e-12, max_rank=10**9)
    for pb in table.patches:
        residual = (
            pb.fixed_source
            + pb.scatter_in
            - pb.absorption
            - pb.scatter_out
            - pb.leakage
        )
        assert (residual.abs() / gb.fixed_source).item() < 1e-3

    # INTERNAL contributions to leakage are exact opposites on the two sides
    # (resolve_internal_faces() derives both from the same pair of outgoing
    # tensors), so they must cancel exactly when summed -- matching
    # gb.leakage, which never includes them in the first place.
    patch_leakage_sum = sum(
        (pb.leakage for pb in table.patches[1:]), table.patches[0].leakage
    )
    torch.testing.assert_close(patch_leakage_sum, gb.leakage, rtol=1e-10, atol=1e-12)

    # NOTE: gb.dd_residual is NOT checked here. It compares each side's own
    # bare `outgoing` (the Omega.n>0 hemisphere for THAT patch's own normal)
    # directly against the other side's bare `outgoing` (the Omega.n>0
    # hemisphere for the OPPOSITE normal) -- physically disjoint angular sets
    # for a one-directional-streaming problem like this one (all flow is
    # rightward: GID0's internal outgoing is a real, nonzero rightward
    # current; GID1's internal outgoing is genuinely ~0, since nothing flows
    # back left). dd_residual only comes out near-zero when the two
    # directions happen to carry equal current (e.g. the symmetric
    # homogeneous-slab eigenvalue problem in
    # test_two_patch_internal_dd_residual_near_zero) -- a coincidence of that
    # problem's symmetry, not a general DD-conservation check. This is a
    # pre-existing property of dd_residual's definition, not something
    # resolve_internal_faces() introduced.


def test_2d_reflective_anisotropic_balance_closes():
    """Single 2D patch, anisotropic (P1) scatterer, REFLECTIVE on x_min/y_min
    (mirroring the quarter-symmetry corner used throughout the C5G7/cruciform
    benchmarks) and VACUUM on x_max/y_max, with a uniform volumetric source.
    No DD coupling at all (a single patch has no INTERNAL faces here), so
    this isolates the local operator + reflective boundary from block-Jacobi
    dynamics entirely: if the angle-integrated balance still closes here, an
    outer-loop/DD explanation for a real anisotropic-scattering divergence is
    much harder to sustain, since nothing DD-related is exercised at all.

    scatter_in only ever uses the l=0 moment (see compute_balance() -- this
    is not a shortcut, it's exact: integrating any l>=1 term over the full
    receiving solid angle is exactly zero by Legendre orthogonality with
    P_0=1, so l=0 is the only moment that can appear in an angle-integrated
    group-transfer balance). That means a real bug in the l>=1 machinery
    that corrupts the *solved* angular flux psi (not just the reported
    balance) still shows up here: a wrong psi feeds back into every term of
    the balance (absorption, scatter_out, leakage) even though scatter_in's
    own formula only reads the l=0 moment directly."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.5
    sigma_s0 = 0.3
    sigma_s1 = 0.1
    Q = 1.0
    L = 2.0

    server = Server()
    mat = Material("AnisotropicScatterer")
    mat.chi = torch.zeros(1, dtype=dtype)
    mat.total = torch.tensor([sigma_t], dtype=dtype)
    mat.nu_fission = torch.zeros(1, dtype=dtype)
    mat.fission = torch.zeros(1, dtype=dtype)
    mat.absorption = torch.tensor([sigma_t - sigma_s0], dtype=dtype)
    mat.scatter_gtg = torch.tensor([[[sigma_s0]], [[sigma_s1]]], dtype=dtype)
    mat.finalize()
    server.add_material(mat)
    server.finalize()

    # dim=0 is the line's own parametrization (x, 0..L); dim=1 is the ruling
    # direction (y, 0..L) -- is_upper=False/True give the x_min/x_max and
    # y_min/y_max edges respectively.
    bottom = line((0, 0), (L, 0))
    top = line((0, L), (L, L))
    patch = Patch.from_igakit(
        refine(ruled(bottom, top), [10, 10], [2, 2]),
        device=device,
        dtype=dtype,
        fill=mat.label,
    )
    patch.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.REFLECTIVE)  # x_min
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)  # x_max
    patch.set_boundary_type(1, False, BoundaryType.REFLECTIVE)  # y_min
    patch.set_boundary_type(1, True, BoundaryType.VACUUM)  # y_max

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = ProductQuadrature.gauss_legendre_chebyshev(8, 8, 2)
    qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-10
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = True

    driver = IGATransportDriver2D(mesh, server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=1e-8,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=6, eps=1e-9, kickrank=4, local_iterations=200, resets=6)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)
    result = driver.solve_fixed_source(
        dd_solver, tol=1e-7, max_iter=50, verbose=False, clear_assemblers=False
    )

    gb = result.global_balance(driver.get_assemblers(), eps=1e-12, max_rank=10**9)
    assert gb.scatter_in.item() > 0.0
    assert gb.scatter_out.item() > 0.0
    assert gb.fission_source.item() == 0.0
    gain = gb.fixed_source + gb.scatter_in
    loss = gb.absorption + gb.scatter_out + gb.leakage
    assert ((gain - loss).abs() / gain).item() < 1e-3

    # Both REFLECTIVE faces are self-consistent by construction (a free
    # internal-consistency check, same as test_reflective_face_incoming_
    # matches_outgoing() -- not itself proof the *physics* is right, but a
    # real anisotropic-scattering bug corrupting psi would very plausibly
    # break this too).
    table = result.patch_balance_table(
        driver.get_assemblers(), eps=1e-12, max_rank=10**9
    )
    pb = table.patches[0]
    reflective_faces = [f for f in pb.faces if f.type == BoundaryType.REFLECTIVE]
    assert len(reflective_faces) == 2
    for f in reflective_faces:
        torch.testing.assert_close(f.outgoing, f.incoming, rtol=1e-3, atol=1e-6)


def test_to_dataframe_smoke():
    """to_dataframe() must not raise on either a global_balance() dict or a
    patch_balance_table() list."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    fill, server = _pure_absorber(0.5, device, dtype)
    patch = Patch.from_igakit(
        refine(line((0, 0), (1.0, 0)), 8, 3), device=device, dtype=dtype, fill=fill
    )
    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()
    patch.source = FixedSource(isotropic_strength=torch.tensor([1.0], dtype=dtype))
    patch.set_boundary_type(0, False, BoundaryType.VACUUM)
    patch.set_boundary_type(0, True, BoundaryType.VACUUM)

    qset = QuadratureSet1D.gauss_legendre(32)
    qset.to_(device, dtype)
    driver, result = _solve_fixed_source(mesh, server, qset)

    assemblers = driver.get_assemblers()
    gb = result.global_balance(assemblers, eps=1e-10, max_rank=10**9)
    gb.to_dataframe()
    table = result.patch_balance_table(assemblers, eps=1e-10, max_rank=10**9)
    table.to_dataframe()


def test_2d_curved_boundary_balance_closes():
    """Single-patch 2D annulus (pure absorber, uniform volumetric source,
    VACUUM at both the inner and outer curved boundaries): every existing
    balance test up to this point is 1D, which never exercises
    DGFirstOrderTransportBackend::assemble_leakage_functional()'s NumDim > 1
    branch. A flat-boundary 2D patch wouldn't catch a real bug here either --
    a flat boundary's (Omega . n) mask doesn't depend on tangential position,
    so its TT rank there is trivially 1, which happens to match a wrong
    hardcoded rank-1 assumption. This annulus's inner/outer boundaries are
    genuinely curved (the mask has real, non-trivial tangential rank there),
    which is what actually exercises -- and previously exposed -- a TT bond-
    rank mismatch bug in assemble_leakage_functional()'s face-core insertion
    for every dim except the last one (only the last dim's insertion happens
    to be a plain append, which is safe regardless of rank; every other dim
    inserts into the middle of the TT chain, where a wrong hardcoded rank-1
    core silently corrupts the whole functional)."""
    mpi_context.init()
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)
    device = torch.device("cpu")

    sigma_t = 0.75
    Q = 2.0
    r_in = 2.0
    r_out = 5.0

    server = Server()
    mat = Material("Absorber")
    mat.chi = torch.zeros(1, dtype=dtype)
    mat.total = torch.tensor([sigma_t], dtype=dtype)
    mat.nu_fission = torch.zeros(1, dtype=dtype)
    mat.fission = torch.zeros(1, dtype=dtype)
    mat.absorption = torch.tensor([sigma_t], dtype=dtype)
    mat.scatter_gtg = torch.zeros((1, 1, 1), dtype=dtype)
    mat.finalize()
    server.add_material(mat)
    server.finalize()

    patch = Patch.from_igakit(
        refine(ruled(circle(r_in), circle(r_out)), [10, 4], [3, 3]),
        device=device,
        dtype=dtype,
        fill=mat.label,
    )
    patch.source = FixedSource(isotropic_strength=torch.tensor([Q], dtype=dtype))
    # dim=0 is the closed azimuthal direction (periodic, INTERNAL after
    # connect()); dim=1 is radial -- is_upper=False is the inner circle,
    # is_upper=True is the outer circle.
    patch.set_boundary_type(1, False, BoundaryType.VACUUM)
    patch.set_boundary_type(1, True, BoundaryType.VACUUM)

    mesh = IGAMesh(mpi_context)
    mesh.add_block(patch)
    mesh.connect()
    mesh.finalize()

    qset = ProductQuadrature.gauss_legendre_chebyshev(8, 8, 2)
    qset.to_(device, dtype)

    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-8
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = False

    driver = IGATransportDriver2D(mesh, server, mpi_context)
    driver.assemble(qset, config)

    dd_config = DDSolverConfig(
        tol=1e-7,
        max_iter=10,
        use_gpu=False,
        memory_policy=MemoryPolicy.OUT_OF_CORE,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=False,
    )
    strategy = BlockJacobiStrategy(dd_config)
    strategy.set_local_solver(
        AMEnSolver(nswp=4, eps=1e-7, kickrank=4, local_iterations=80, resets=4)
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)
    result = driver.solve_fixed_source(
        dd_solver, tol=1e-6, max_iter=50, verbose=False, clear_assemblers=False
    )

    gb = result.global_balance(driver.get_assemblers(), eps=1e-10, max_rank=10**9)
    residual = gb.fixed_source - gb.absorption - gb.leakage
    assert (residual.abs() / gb.fixed_source).item() < 1e-3

    # The periodic azimuthal (dim=0) direction is INTERNAL to this single
    # patch -- both sides of that self-connection should agree closely,
    # exactly the kind of curved-boundary, non-last-dim face insertion the
    # original bug corrupted.
    table = result.patch_balance_table(
        driver.get_assemblers(), eps=1e-10, max_rank=10**9
    )
    pb = table.patches[0]
    internal_outgoing = [
        f.outgoing for f in pb.faces if f.type == BoundaryType.INTERNAL
    ]
    assert len(internal_outgoing) == 2
    torch.testing.assert_close(
        internal_outgoing[0], internal_outgoing[1], rtol=1e-3, atol=1e-6
    )
