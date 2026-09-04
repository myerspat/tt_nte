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
    .def("compute_patch_balances", &TransportSolution::compute_patch_balances,
      "Compute every one of this rank's own local patches' own "
      "particle-balance diagnostics (source/absorption/scatter/leakage per "
      "group, per face) from this solution's own field. Purely local -- no "
      "MPI. `assemblers` is GID -> assembler, optional and may be partial or "
      "omitted entirely -- e.g. TransportDriver.get_assemblers() (requires "
      "clear_assemblers=False when solving); any local GID missing from it "
      "gets a fresh assembler built on demand (re-running that patch's "
      "assemble(), so passing existing assemblers is cheaper when you have "
      "them). Returns a dict of GID -> PatchBalance.",
      py::arg("assemblers") =
        std::unordered_map<int64_t, typename TransportSolution::AssemblerPtr>(),
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("patch_balance_table", &TransportSolution::patch_balance_table,
      "Every patch's own particle-balance diagnostics, gathered across "
      "every rank and sorted by GID. Always collective -- every rank must "
      "call this; the result is identical on every rank. `assemblers` is "
      "GID -> assembler, optional (see compute_patch_balances()). Returns a "
      "PatchBalanceTable.",
      py::arg("assemblers") =
        std::unordered_map<int64_t, typename TransportSolution::AssemblerPtr>(),
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())
    .def("global_balance", &TransportSolution::global_balance,
      "Particle-balance diagnostics summed over the whole problem. Always "
      "collective -- every rank must call this; the result is identical on "
      "every rank. `dd_residual` is the direct 'is the distributed method "
      "losing particles' diagnostic (~0 for a lossless, fully converged "
      "distributed solve). `assemblers` is GID -> assembler, optional (see "
      "compute_patch_balances()). Returns a GlobalBalance.",
      py::arg("assemblers") =
        std::unordered_map<int64_t, typename TransportSolution::AssemblerPtr>(),
      py::arg("eps") = 1e-10,
      py::arg("max_rank") = std::numeric_limits<int64_t>::max(),
      py::call_guard<py::gil_scoped_release>())

    // =================================================================
    // Public getters / setters
    .def_property_readonly("k_eff", &TransportSolution::get_k_eff)
    .def_property_readonly(
      "num_outer_iterations", &TransportSolution::get_num_outer_iterations)
    .def_property_readonly(
      "total_inner_iterations", &TransportSolution::get_total_inner_iterations)
    .def_property_readonly("outer_k", &TransportSolution::get_outer_k)
    .def_property_readonly(
      "outer_k_error", &TransportSolution::get_outer_k_error)
    .def_property_readonly(
      "outer_flux_error", &TransportSolution::get_outer_flux_error)
    .def_property_readonly(
      "inner_outer_iter", &TransportSolution::get_inner_outer_iter)
    .def_property_readonly(
      "inner_schwarz_error", &TransportSolution::get_inner_schwarz_error)
    .def_property_readonly("gid2rank", &TransportSolution::get_gid2rank);
}

void register_TransportSolution(py::module_& m)
{
  register_TransportSolution_impl<ttnte::cad::Patch>(m, "IGA");
}
