#include "ttnte/solvers/enrichment_policy.hpp"
#include <torch/extension.h>

namespace py = pybind11;

void register_EnrichmentPolicy(py::module_& m)
{
  using namespace ttnte::solvers;

  py::class_<EnrichmentPolicy, EnrichmentPolicy::Ptr>(m, "EnrichmentPolicy")
    // =================================================================
    // Public methods
    .def("should_enrich", &EnrichmentPolicy::should_enrich, py::arg("eps"),
      py::arg("error"), py::arg("rank_metric") = 0.0)

    // =================================================================
    // Public getters / setters
    .def("has_frozen", &EnrichmentPolicy::has_frozen);

  py::class_<StaticFreezePolicy, EnrichmentPolicy, StaticFreezePolicy::Ptr>(
    m, "StaticFreezePolicy")
    // =================================================================
    // Public constructors
    .def(py::init([](double freeze_eps) {
      return StaticFreezePolicy::create(freeze_eps);
    }),
      py::arg("freeze_eps"))

    // =================================================================
    // Public getters / setters
    .def_property_readonly("freeze_eps", &StaticFreezePolicy::get_freeze_eps);

  py::class_<AdaptiveRevalidationPolicy, EnrichmentPolicy,
    AdaptiveRevalidationPolicy::Ptr>(m, "AdaptiveRevalidationPolicy")
    // =================================================================
    // Public constructors
    .def(py::init([](int64_t initial_period, int64_t probe_iterations,
                    double growth_factor, int64_t max_period,
                    double growth_tolerance) {
      return AdaptiveRevalidationPolicy::create(initial_period,
        probe_iterations, growth_factor, max_period, growth_tolerance);
    }),
      py::arg("initial_period") = 1, py::arg("probe_iterations") = 1,
      py::arg("growth_factor") = 2.0, py::arg("max_period") = 64,
      py::arg("growth_tolerance") = 0.01)

    // =================================================================
    // Public getters / setters
    .def_property_readonly(
      "initial_period", &AdaptiveRevalidationPolicy::get_initial_period)
    .def_property_readonly(
      "probe_iterations", &AdaptiveRevalidationPolicy::get_probe_iterations)
    .def_property_readonly(
      "growth_factor", &AdaptiveRevalidationPolicy::get_growth_factor)
    .def_property_readonly(
      "max_period", &AdaptiveRevalidationPolicy::get_max_period)
    .def_property_readonly(
      "growth_tolerance", &AdaptiveRevalidationPolicy::get_growth_tolerance)
    .def_property_readonly("period", &AdaptiveRevalidationPolicy::get_period);

  py::class_<HardFreezeWrapper, EnrichmentPolicy, HardFreezeWrapper::Ptr>(
    m, "HardFreezeWrapper")
    // =================================================================
    // Public constructors
    .def(py::init([](EnrichmentPolicy::Ptr inner, double freeze_eps) {
      return HardFreezeWrapper::create(std::move(inner), freeze_eps);
    }),
      py::arg("inner"), py::arg("freeze_eps"))

    // =================================================================
    // Public getters / setters
    .def_property_readonly("freeze_eps", &HardFreezeWrapper::get_freeze_eps)
    .def_property_readonly("inner", &HardFreezeWrapper::get_inner);
}
