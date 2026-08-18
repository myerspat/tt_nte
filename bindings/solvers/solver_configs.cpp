#include "ttnte/solvers/solver_configs.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_DDSolverConfig(py::module_& m)
{
  using namespace ttnte::solvers;

  py::enum_<ExecMode>(m, "ExecMode")
    .value("SYNC", ExecMode::SYNC)
    .value("ASYNC", ExecMode::ASYNC)
    .export_values();

  py::enum_<CommMode>(m, "CommMode")
    .value("SYNC", CommMode::SYNC)
    .value("ASYNC", CommMode::ASYNC)
    .export_values();

  py::class_<DDSolverConfig>(m, "DDSolverConfig")
    .def(py::init<double, int, ExecMode, CommMode, int, bool, MemoryPolicy,
           double, bool>(),
      py::arg("tol") = 1e-8, py::arg("max_iter") = 100,
      py::arg("exec_mode") = ExecMode::ASYNC,
      py::arg("comm_mode") = CommMode::ASYNC, py::arg("num_threads") = 4,
      py::arg("use_gpu") = DEFAULT_USE_GPU,
      py::arg("memory_policy") = DEFAULT_MEMORY_POLICY,
      py::arg("tol_forcing") = 0.1, py::arg("verbose") = false)

    // =================================================================
    // Fields
    .def_readwrite("tol", &DDSolverConfig::tol)
    .def_readwrite("max_iter", &DDSolverConfig::max_iter)
    .def_readwrite("exec_mode", &DDSolverConfig::exec_mode)
    .def_readwrite("comm_mode", &DDSolverConfig::comm_mode)
    .def_readwrite("num_threads", &DDSolverConfig::num_threads)
    .def_readwrite("use_gpu", &DDSolverConfig::use_gpu)
    .def_readwrite("memory_policy", &DDSolverConfig::memory_policy)
    .def_readwrite("verbose", &DDSolverConfig::verbose)
    .def_readwrite("tol_forcing", &DDSolverConfig::tol_forcing);
}
