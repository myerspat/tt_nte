import atexit
import os
import sys

from ttnte.cpp.ttnte_python.parallel import *

mpi_context = ParallelContext.instance()


def _finalize_mpi_and_exit():
    """Finalize MPI explicitly, then hard-exit to skip Python's normal
    interpreter teardown.

    The C++/Torch backend holds MPI communicators in objects whose
    destruction order relative to MPI_Finalize is otherwise unspecified at
    interpreter shutdown -- letting that teardown run as usual can trigger
    "MPI_Comm_f2c() called after MPI_FINALIZE" aborts. Registered once here
    so every script that imports ttnte gets this safety net automatically,
    without needing to add anything itself (mirrors the pattern already used
    by ttnte-vnv/conftest.py's pytest_unconfigure hook).
    """
    exit_code = 1 if sys.exc_info()[0] is not None else 0
    try:
        from mpi4py import MPI

        if MPI.Is_initialized() and not MPI.Is_finalized():
            MPI.COMM_WORLD.Barrier()
            MPI.Finalize()
    except Exception:
        pass
    os._exit(exit_code)


atexit.register(_finalize_mpi_and_exit)
