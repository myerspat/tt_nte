import multiprocessing

import torch
from igakit.cad import circle, line, ruled, refine, coons
import matplotlib.pyplot as plt

from ttnte import mpi_context
from ttnte.visualization.style import get_patch_style
from ttnte.parallel import IGADofHeuristic
from ttnte.xs.benchmarks import c5g7
from ttnte.cad import Patch
from ttnte.mesh import IGAMesh
from ttnte.physics import (
    BoundaryType,
    BCPlane,
    DGTransportAssemblerConfig,
)
from ttnte.math import ProductQuadrature
from ttnte.linalg import Operator, TTEngine, mm
from ttnte.driver import IGATransportDriver2D
from ttnte.solvers import (
    DDSolverConfig,
    MemoryPolicy,
    AMEnSolver,
    BlockJacobiStrategy,
    ExecMode,
    IGADDSolver,
    CommMode,
)


def warmup_all_gpus():
    """Forces PyTorch to eagerly initialize the CUDA context, memory allocator, cuBLAS
    (matmul), and cuSOLVER (linalg) on all available GPUs."""
    if not torch.cuda.is_available():
        return

    num_gpus = torch.cuda.device_count()

    for i in range(num_gpus):
        device = torch.device(f"cuda:{i}")

        # 1. Warm up the primary CUDA context and memory allocator
        # A simple allocation and pointwise operation
        x = torch.ones(256, 256, device=device)
        y = x * 2.0

        # 2. Warm up cuBLAS (Dense Matrix Multiplication backend)
        _ = torch.matmul(x, y)

        # 3. Warm up cuSOLVER (Linear Algebra backend)
        # This is CRITICAL to prevent the "lazy wrapper should be called at most once"
        # fatal C++ error when background threads do SVD or QR for the first time.
        small_mat = torch.randn(2, 2, device=device)
        _ = torch.linalg.qr(small_mat)
        _ = torch.linalg.svd(small_mat)

        # 4. Synchronize to ensure all asynchronous GPU setup tasks are totally finished
        # before the main Python thread continues.
        torch.cuda.synchronize(device=device)


if __name__ == "__main__":
    # Settings
    dtype = torch.float64
    cpu = torch.device("cpu")

    # Initialize MPI
    mpi_context.init()
    num_threads_per_rank = min(multiprocessing.cpu_count() // mpi_context.world_size, 4)
    torch.set_num_threads(num_threads_per_rank)

    # Set defaults for PyTorch
    torch.set_default_dtype(dtype)
    torch.autograd.set_grad_enabled(False)

    # Get XS information
    fills, xs_server = c5g7(device=cpu, dtype=dtype)

    # Create quarter circle NURBS surface
    radius = 0.54  # cm
    pitch = 1.26  # cm
    c0 = circle(radius, angle=torch.pi / 4)
    c1 = circle(radius, angle=-torch.pi / 4).rotate(torch.pi / 2)
    l0 = line((0, 0), (radius, 0))
    l1 = line((0, 0), (0, radius))

    s0 = coons([[l1, c0], [l0, c1]])
    c0 = s0.boundary(0, 1)
    c1 = s0.boundary(1, 1)

    # Create water patch
    l2 = line(p0=(pitch / 2, 0), p1=(pitch / 2, pitch / 2))
    l3 = line(p0=(0, pitch / 2), p1=(pitch / 2, pitch / 2))

    # Create NURBS surfaces
    fuel = [s0]
    water = [ruled(c0, l2), ruled(c1, l3)]

    # Create the mesh
    mesh = IGAMesh(mpi_context)

    for fill, surface in zip([fills[0], fills[-2], fills[-2]], fuel + water):
        mesh.add_block(
            Patch.from_igakit(
                refine(surface, 12, 2), device=cpu, dtype=dtype, fill=fill
            )
        )

    # Connect patches
    mesh.connect()

    # Set the boundary conditions
    mesh.set_axis_aligned_conditions(
        BCPlane(x_min=True, y_min=True, x_max=True, y_max=True),
        BoundaryType.REFLECTIVE,
        tol=1e-6,
    )
    mesh.finalize()

    backend = "matplotlib"
    style = get_patch_style(backend)
    style.mesh.cmap = {
        fills[0].to_string(): "maroon",
        fills[-2].to_string(): "cornflowerblue",
    }
    mesh.plot(
        resolution=25,
        show_ctrlpts=True,
        show_ctrlnet=True,
        show_boundary=True,
        backend="matplotlib",
        filename="figs/c5g7_pincell_2d.png",
        style=style,
    )

    # Create angular quadrature
    qset = ProductQuadrature.gauss_legendre_chebyshev(16, 16, 2)
    qset.to_(torch.device("cpu"), dtype)

    # Create assembly backend
    config = DGTransportAssemblerConfig()
    config.rounding.eps = 1e-6
    config.cross.eps = config.rounding.eps
    config.max_dense_size = int(1e10)
    config.cross_jacobian_inverse = False

    # Create the transport driver
    driver = IGATransportDriver2D(mesh, xs_server, mpi_context)

    # Distribute patches among MPI ranks
    driver.distribute([IGADofHeuristic()])

    # Run the assembler
    driver.assemble(qset, config)

    for s in driver.mesh.blocks:
        assembler = driver.get_assembler(s.gid)
        output = f"GID: {s.gid}\n"

        H = assembler.interior_loss_op.as_tt()
        output += f"  H: Ranks {H.ranks}, CR: {H.compression}\n"
        S = assembler.scatter_op.as_tt()
        output += f"  S: Ranks {S.ranks}, CR: {S.compression}\n"

        if assembler.fission_op.defined():
            F = assembler.fission_op.as_tt()
            output += f"  F: Ranks {F.ranks}, CR: {F.compression}\n"

        for op in assembler.inflow_ops:
            Bin = op.as_tt()
            output += f"  Bin: Ranks {Bin.ranks}, CR: {Bin.compression}\n"

        for op in assembler.outflow_ops:
            Bout = op.as_tt()
            output += f"  Bout: Ranks {Bout.ranks}, CR: {Bout.compression}\n"

        print(output, end="")

    # Warmup GPUs
    warmup_all_gpus()

    # Create Block-Jacobi DD strategy
    outer_tol = 1e-4
    inner_tol = 5e-5
    eps = 1e-5

    config = DDSolverConfig(
        tol=inner_tol,
        tol_forcing=0.1,
        max_iter=100,
        use_gpu=True,
        memory_policy=MemoryPolicy.RESIDENT,
        exec_mode=ExecMode.ASYNC,
        comm_mode=CommMode.ASYNC,
        verbose=True,
    )
    strategy = BlockJacobiStrategy(config)
    strategy.set_local_solver(
        AMEnSolver(
            nswp=2,
            eps=eps,
            eps_forcing=0.01,
            kickrank=4,
            local_iterations=200,
            resets=4,
            max_rank=500,
        )
    )
    dd_solver = IGADDSolver(driver.mesh, strategy)

    # Run DD eigenvalue solver
    result = driver.solve_eigenvalue(
        dd_solver, tol=outer_tol, max_iter=500, verbose=True
    )
