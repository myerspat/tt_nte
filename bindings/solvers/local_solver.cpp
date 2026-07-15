#include "ttnte/solvers/local_solver.hpp"
#include "ttnte/solvers/solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

// Trampoline class for virtual method redirection
class PyLocalSolver : public ttnte::solvers::LocalSolver {
public:
  using LocalSolver::LocalSolver; // Inherit constructors

  // Redirect C++ virtual calls to Python
  void solve(const ttnte::linalg::LinearSystem::Ptr& local_system) override
  {
    PYBIND11_OVERRIDE(void, /* Return type */
      LocalSolver,          /* Parent class */
      solve,                /* Name of function */
      local_system          /* Argument(s) */
    );
  }

  // LocalSolver leaves get_state_format() unimplemented (pure virtual on
  // Solver), so the trampoline must override it too to remain concrete.
  ttnte::linalg::FormatType get_state_format() override
  {
    PYBIND11_OVERRIDE_PURE(
      ttnte::linalg::FormatType, LocalSolver, get_state_format);
  }
};

void register_LocalSolver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<LocalSolver, Solver, PyLocalSolver, std::shared_ptr<LocalSolver>>(
    m, "LocalSolver")
    // =================================================================
    // Public methods
    .def("solve", &LocalSolver::solve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>())
    .def("presolve", &LocalSolver::presolve, py::arg("local_system"),
      py::call_guard<py::gil_scoped_release>())
    .def("postsolve", &LocalSolver::postsolve, py::arg("local_system"),
      py::arg("x"), py::call_guard<py::gil_scoped_release>());
}
