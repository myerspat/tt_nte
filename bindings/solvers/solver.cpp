#include "ttnte/solvers/solver.hpp"
#include <torch/extension.h>

namespace py = pybind11;

// Trampoline class for virtual method redirection
class PySolver : public ttnte::solvers::Solver {
public:
  using Solver::Solver;

  void init(const Systems& local_systems) override
  {
    PYBIND11_OVERRIDE_PURE(void, Solver, init, local_systems);
  }
  void wait_for_thread_init() override
  {
    PYBIND11_OVERRIDE(void, Solver, wait_for_thread_init);
  }
  void step() override { PYBIND11_OVERRIDE_PURE(void, Solver, step); }
  void finalize() override { PYBIND11_OVERRIDE(void, Solver, finalize); }
  void update_convergence_criteria(double error) override
  {
    PYBIND11_OVERRIDE(void, Solver, update_convergence_criteria, error);
  }
  const Systems& get_local_systems() const override
  {
    PYBIND11_OVERRIDE_PURE(const Systems&, Solver, get_local_systems);
  }
  void set_local_systems(const Systems& local_systems) override
  {
    PYBIND11_OVERRIDE_PURE(void, Solver, set_local_systems, local_systems);
  }
  ttnte::linalg::FormatType get_state_format() override
  {
    PYBIND11_OVERRIDE_PURE(ttnte::linalg::FormatType, Solver, get_state_format);
  }
  double get_eps() const override
  {
    PYBIND11_OVERRIDE(double, Solver, get_eps);
  }
  int64_t get_max_rank() const override
  {
    PYBIND11_OVERRIDE(int64_t, Solver, get_max_rank);
  }
};

void register_Solver(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<Solver, PySolver, std::shared_ptr<Solver>>(m, "Solver")
    // =================================================================
    // Public methods
    .def("init", &Solver::init, py::arg("local_systems"),
      py::call_guard<py::gil_scoped_release>())
    .def("wait_for_thread_init", &Solver::wait_for_thread_init,
      py::call_guard<py::gil_scoped_release>())
    .def("step", &Solver::step, py::call_guard<py::gil_scoped_release>())
    .def(
      "finalize", &Solver::finalize, py::call_guard<py::gil_scoped_release>())
    .def("update_convergence_criteria", &Solver::update_convergence_criteria,
      py::arg("error"))

    // =================================================================
    // Public getters / setters
    .def_property(
      "local_systems", &Solver::get_local_systems, &Solver::set_local_systems)
    .def_property_readonly("state_format", &Solver::get_state_format)
    .def_property_readonly("eps", &Solver::get_eps)
    .def_property_readonly("max_rank", &Solver::get_max_rank);
}
