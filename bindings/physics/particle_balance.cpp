#include "ttnte/physics/particle_balance.hpp"
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

void register_ParticleBalance(py::module_& m)
{
  using namespace ttnte::physics;

  py::class_<FaceBalance>(m, "FaceBalance")
    .def_readonly("dim", &FaceBalance::dim)
    .def_readonly("is_upper", &FaceBalance::is_upper)
    .def_readonly("type", &FaceBalance::type)
    .def_readonly("neighbor_gid", &FaceBalance::neighbor_gid)
    .def_readonly("neighbor_dim", &FaceBalance::neighbor_dim)
    .def_readonly("neighbor_is_upper", &FaceBalance::neighbor_is_upper)
    .def_readonly("outgoing", &FaceBalance::outgoing)
    .def_readonly("incoming", &FaceBalance::incoming);

  py::class_<PatchBalance>(m, "PatchBalance")
    .def_readonly("gid", &PatchBalance::gid)
    .def_readonly("fixed_source", &PatchBalance::fixed_source)
    .def_readonly("fission_source", &PatchBalance::fission_source)
    .def_readonly("absorption", &PatchBalance::absorption)
    .def_readonly("scatter_in", &PatchBalance::scatter_in)
    .def_readonly("scatter_out", &PatchBalance::scatter_out)
    .def_readonly("leakage", &PatchBalance::leakage)
    .def_readonly("faces", &PatchBalance::faces);

  // to_dataframe() is attached in ttnte/driver/_balance.py, the same way
  // plotting is attached to ttnte.cad.Patch.
  py::class_<PatchBalanceTable>(m, "PatchBalanceTable")
    .def_readonly("patches", &PatchBalanceTable::patches);

  py::class_<GlobalBalance>(m, "GlobalBalance")
    .def_readonly("fixed_source", &GlobalBalance::fixed_source)
    .def_readonly("fission_source", &GlobalBalance::fission_source)
    .def_readonly("absorption", &GlobalBalance::absorption)
    .def_readonly("scatter_in", &GlobalBalance::scatter_in)
    .def_readonly("scatter_out", &GlobalBalance::scatter_out)
    .def_readonly("leakage", &GlobalBalance::leakage)
    .def_readonly("dd_residual", &GlobalBalance::dd_residual);
}
