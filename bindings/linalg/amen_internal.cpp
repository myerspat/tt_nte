// Bindings for the low-level building blocks of the native AMEn solver
// (ttnte::linalg::amen). These are not part of the public solver API -- they
// are exposed so the primitives can be unit-tested directly against
// dense/torch ground truth (see tests/unit/linalg/).
#include "ttnte/linalg/amen/gmres_local.hpp"
#include "ttnte/linalg/amen/local_operator.hpp"
#include "ttnte/linalg/amen/local_preconditioner.hpp"
#include "ttnte/linalg/amen/qless_tsqr.hpp"
#include "ttnte/linalg/amen/rank1_preconditioner.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_amen_internal(py::module_& m)
{
  using namespace ttnte::linalg::amen;

  m.def("tsqr_r", &tsqr_r, py::arg("M"), py::arg("block_size"),
    "Q-less, blocked TSQR: compute only the R factor of M = QR.",
    py::call_guard<py::gil_scoped_release>());

  m.def("qless_orthogonalize", &qless_orthogonalize, py::arg("M"),
    py::arg("block_size"),
    "Q-less orthogonalization: returns (Q, R) with Q = M @ R^{-1}.",
    py::call_guard<py::gil_scoped_release>());

  py::class_<Rank1Preconditioner>(m, "Rank1Preconditioner")
    .def_static("build", &Rank1Preconditioner::build, py::arg("A"),
      py::arg("sv_floor_ratio") = 1e-10)
    .def(
      "apply_left", &Rank1Preconditioner::apply_left, py::arg("vector_cores"))
    .def(
      "apply_right", &Rank1Preconditioner::apply_right, py::arg("vector_cores"))
    .def("apply_right_inverse", &Rank1Preconditioner::apply_right_inverse,
      py::arg("vector_cores"))
    .def("sandwich_operator", &Rank1Preconditioner::sandwich_operator,
      py::arg("operator_cores"));

  py::class_<FoldedLocalOperator>(m, "FoldedLocalOperator")
    .def_static("build", &FoldedLocalOperator::build, py::arg("phi_left"),
      py::arg("a_core"), py::arg("phi_right"), py::arg("regularization") = 0.0)
    .def("apply", &FoldedLocalOperator::apply, py::arg("y"))
    .def("to_dense", &FoldedLocalOperator::to_dense);

  py::class_<LocalPreconditioner>(m, "LocalPreconditioner")
    .def_static("build", &LocalPreconditioner::build, py::arg("phi_left"),
      py::arg("a_core"), py::arg("phi_right"), py::arg("mode"))
    .def("apply_forward", &LocalPreconditioner::apply_forward, py::arg("x"))
    .def("apply_inverse", &LocalPreconditioner::apply_inverse, py::arg("y"));

  m.def("gmres_solve", &gmres_solve, py::arg("op"), py::arg("rhs"),
    py::arg("x0"), py::arg("max_iterations"), py::arg("restarts"),
    py::arg("rel_tol"), py::arg("prefer_incremental") = false,
    py::arg("prec") = static_cast<const LocalPreconditioner*>(nullptr),
    py::call_guard<py::gil_scoped_release>());
  m.def("gmres_solve_cpu", &gmres_solve_cpu, py::arg("op"), py::arg("rhs"),
    py::arg("x0"), py::arg("max_iterations"), py::arg("restarts"),
    py::arg("rel_tol"),
    py::arg("prec") = static_cast<const LocalPreconditioner*>(nullptr),
    py::call_guard<py::gil_scoped_release>());
  m.def("gmres_solve_gpu", &gmres_solve_gpu, py::arg("op"), py::arg("rhs"),
    py::arg("x0"), py::arg("max_iterations"), py::arg("restarts"),
    py::arg("rel_tol"),
    py::arg("prec") = static_cast<const LocalPreconditioner*>(nullptr),
    py::call_guard<py::gil_scoped_release>());
}
