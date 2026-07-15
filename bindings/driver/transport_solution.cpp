#include "ttnte/driver/transport_solution.hpp"
#include "ttnte/cad/patch.hpp"
#include <c10/util/SmallVector.h>
#include <limits>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

template<typename BlockType>
static void register_TransportSolution_impl(
  py::module_& m, const std::string& typestr)
{
  using TransportSolution = ttnte::driver::TransportSolution<BlockType>;
  using SolutionPtr = typename TransportSolution::Ptr;

  std::string class_name = typestr + "TransportSolution";

  py::class_<TransportSolution, SolutionPtr>(m, class_name.c_str())
    // =================================================================
    // Public methods
    .def("compute_scalar_flux", &TransportSolution::compute_scalar_flux,
      "Reduce every local field to its scalar flux (0th angular moment). "
      "Returns a NEW TransportSolution; this instance is untouched.",
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("select_group", &TransportSolution::select_group,
      "Select a single energy group, narrowing every local field's energy "
      "axis down to size 1. Returns a NEW TransportSolution; this instance "
      "is untouched. Works on either the raw angular flux or a "
      "compute_scalar_flux()'d solution.",
      py::arg("group"), py::call_guard<py::gil_scoped_release>())
    .def("get_local_field", &TransportSolution::get_local_field,
      "Get this rank's own local field for a GID (no MPI). Call .to_dense() "
      "on the result for a dense tensor.",
      py::arg("gid"), py::return_value_policy::reference_internal)
    .def("compute_errors", &TransportSolution::compute_errors,
      "Per-patch, per-energy-group relative L2 error against a reference "
      "TransportSolution (which may live on a differently-refined NURBS "
      "discretization of the same geometry), properly volume- and "
      "angle-weighted. This rank's own local patches only -- no MPI. "
      "Returns a dict of GID -> length-num_groups tensor.",
      py::arg("reference"), py::call_guard<py::gil_scoped_release>())
    .def("error_norm", &TransportSolution::error_norm,
      "Relative L2 error per energy group against a reference "
      "TransportSolution, aggregated over every local patch on every rank, "
      "properly volume- and angle-weighted. Returns a length-num_groups "
      "tensor. Always collective -- every rank must call this.",
      py::arg("reference"), py::call_guard<py::gil_scoped_release>())
    .def(
      "regular_mesh_average",
      [](const TransportSolution& self, const std::vector<int64_t>& shape,
        const std::vector<int64_t>& n, int64_t max_iter, double tol,
        int64_t seed_resolution) {
        return self.regular_mesh_average(
          c10::SmallVector<int64_t, 3>(shape.begin(), shape.end()),
          c10::SmallVector<int64_t, 3>(n.begin(), n.end()), max_iter, tol,
          seed_resolution);
      },
      "Volume-averaged field on a regular Cartesian grid via composite "
      "trapezoidal integration. Requires this solution to be spatial-only "
      "(compute_scalar_flux()'s result). Returns a tensor of shape "
      "(*shape, num_groups). Always collective -- every rank must call "
      "this. Raise seed_resolution if it errors with 'not covered by any "
      "patch' for a point that's genuinely on the mesh -- this can happen "
      "near a coordinate singularity or a multi-patch corner.",
      py::arg("shape"), py::arg("n"), py::arg("max_iter") = 10,
      py::arg("tol") = 1e-8,
      py::arg("seed_resolution") =
        ttnte::cad::DEFAULT_INVERSE_MAP_SEED_RESOLUTION,
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Public getters / setters
    .def_property_readonly("k_eff", &TransportSolution::get_k_eff)
    .def_property_readonly("gid2rank", &TransportSolution::get_gid2rank);
}

void register_TransportSolution(py::module_& m)
{
  register_TransportSolution_impl<ttnte::cad::Patch>(m, "IGA");
}
