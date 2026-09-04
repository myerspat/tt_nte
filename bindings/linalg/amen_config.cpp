#include "ttnte/linalg/amen/amen_config.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_AMEnConfig(py::module_& m)
{
  using namespace ttnte::linalg;

  py::enum_<AMEnBackend>(m, "AMEnBackend")
    .value("TORCHTT", AMEnBackend::TORCHTT)
    .value("NATIVE", AMEnBackend::NATIVE)
    .export_values();

  py::enum_<AMEnEnrichmentMode>(m, "AMEnEnrichmentMode")
    .value("FULL", AMEnEnrichmentMode::FULL)
    .value("SIMPLIFIED", AMEnEnrichmentMode::SIMPLIFIED)
    .value("ALS_FIXED_RANK", AMEnEnrichmentMode::ALS_FIXED_RANK)
    .export_values();

  py::enum_<AMEnPreconditioner>(m, "AMEnPreconditioner")
    .value("NONE", AMEnPreconditioner::NONE)
    .value("LOCAL_C_PREC", AMEnPreconditioner::LOCAL_C_PREC)
    .value("LOCAL_R_PREC", AMEnPreconditioner::LOCAL_R_PREC)
    .value("RANK1", AMEnPreconditioner::RANK1)
    .export_values();

  py::class_<AMEnNativeOptions>(m, "AMEnNativeOptions")
    .def(py::init<bool, AMEnEnrichmentMode, int, int, bool, double, bool, bool,
           double, double>(),
      py::arg("use_qless_tsqr") = true,
      py::arg("enrichment_mode") = AMEnEnrichmentMode::ALS_FIXED_RANK,
      py::arg("als_residual_rank") = 4,
      py::arg("tsqr_block_size") = DEFAULT_TSQR_BLOCK_SIZE,
      py::arg("use_local_forcing") = false,
      py::arg("gmres_forcing_ceiling") = 1e-1,
      py::arg("use_gpu_batched_gmres") = false,
      py::arg("gmres_mixed_precision") = false,
      py::arg("proximal_regularization") = 0.0, py::arg("resid_damp") = 2.0)

    // =================================================================
    // Fields
    .def_readwrite("use_qless_tsqr", &AMEnNativeOptions::use_qless_tsqr)
    .def_readwrite("enrichment_mode", &AMEnNativeOptions::enrichment_mode)
    .def_readwrite("als_residual_rank", &AMEnNativeOptions::als_residual_rank)
    .def_readwrite("tsqr_block_size", &AMEnNativeOptions::tsqr_block_size)
    .def_readwrite("use_local_forcing", &AMEnNativeOptions::use_local_forcing)
    .def_readwrite(
      "gmres_forcing_ceiling", &AMEnNativeOptions::gmres_forcing_ceiling)
    .def_readwrite(
      "use_gpu_batched_gmres", &AMEnNativeOptions::use_gpu_batched_gmres)
    .def_readwrite(
      "gmres_mixed_precision", &AMEnNativeOptions::gmres_mixed_precision)
    .def_readwrite(
      "proximal_regularization", &AMEnNativeOptions::proximal_regularization)
    .def_readwrite("resid_damp", &AMEnNativeOptions::resid_damp);
}
