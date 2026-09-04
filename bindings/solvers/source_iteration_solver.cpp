#include "ttnte/solvers/source_iteration_solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_SourceIterationSolver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<SourceIterationSolver, LocalSolver,
    std::shared_ptr<SourceIterationSolver>>(m, "SourceIterationSolver")
    // =================================================================
    // Public constructors
    .def(py::init([](int nswp, double eps, double eps_forcing, int max_rank,
                    int max_full, int kickrank, int kick2, int local_iterations,
                    int resets, bool verbose,
                    ttnte::linalg::AMEnPreconditioner prec,
                    ttnte::linalg::AMEnBackend backend,
                    ttnte::linalg::AMEnNativeOptions native_opts,
                    int max_si_sweeps, double eps_forcing_si) {
      return SourceIterationSolver::create(nswp, eps, eps_forcing, max_rank,
        max_full, kickrank, kick2, local_iterations, resets, verbose, prec,
        backend, native_opts, max_si_sweeps, eps_forcing_si);
    }),
      py::arg("nswp") = 22, py::arg("eps") = 1e-10,
      py::arg("eps_forcing") = 0.01,
      py::arg("max_rank") = std::numeric_limits<int>::max(),
      py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
      py::arg("local_iterations") = 40, py::arg("resets") = 2,
      py::arg("verbose") = false,
      py::arg("prec") = ttnte::linalg::AMEnPreconditioner::NONE,
      py::arg("backend") = ttnte::linalg::AMEnBackend::NATIVE,
      py::arg("native_opts") = ttnte::linalg::AMEnNativeOptions {},
      py::arg("max_si_sweeps") = 25, py::arg("eps_forcing_si") = 0.1)
    .def("solve", &SourceIterationSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>())
    .def_property_readonly(
      "last_si_sweeps", &SourceIterationSolver::get_last_si_sweeps)
    .def_property_readonly(
      "last_si_residual", &SourceIterationSolver::get_last_si_residual);
}
