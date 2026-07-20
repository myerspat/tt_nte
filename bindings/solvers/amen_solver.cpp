#include "ttnte/solvers/amen_solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_AMEnSolver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<AMEnSolver, LocalSolver, std::shared_ptr<AMEnSolver>>(
    m, "AMEnSolver")
    // =================================================================
    // Public constructors
    .def(py::init([](int nswp, double eps, double eps_forcing, int max_rank,
                    int max_full, int kickrank, int kick2, int local_iterations,
                    int resets, bool verbose,
                    ttnte::linalg::AMEnPreconditioner prec,
                    ttnte::linalg::AMEnBackend backend,
                    ttnte::linalg::AMEnNativeOptions native_opts) {
      return AMEnSolver::create(nswp, eps, eps_forcing, max_rank, max_full,
        kickrank, kick2, local_iterations, resets, verbose, prec, backend,
        native_opts);
    }),
      py::arg("nswp") = 22, py::arg("eps") = 1e-10,
      py::arg("eps_forcing") = 0.01,
      py::arg("max_rank") = std::numeric_limits<int>::max(),
      py::arg("max_full") = 500, py::arg("kickrank") = 4, py::arg("kick2") = 0,
      py::arg("local_iterations") = 40, py::arg("resets") = 2,
      py::arg("verbose") = false,
      py::arg("prec") = ttnte::linalg::AMEnPreconditioner::NONE,
      py::arg("backend") = ttnte::linalg::AMEnBackend::NATIVE,
      py::arg("native_opts") = ttnte::linalg::AMEnNativeOptions{})
    .def("solve", &AMEnSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>())
    .def("is_rank_frozen", &AMEnSolver::is_rank_frozen);
}
