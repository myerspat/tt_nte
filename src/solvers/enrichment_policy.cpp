#include "ttnte/solvers/enrichment_policy.hpp"
#include "ttnte/utils/exception.hpp"

#include <algorithm>
#include <cmath>

namespace ttnte::solvers {

// =================================================================
// StaticFreezePolicy

StaticFreezePolicy::StaticFreezePolicy(double freeze_eps)
  : freeze_eps_(freeze_eps)
{
  if (freeze_eps_ <= 0) {
    throw utils::runtime_error(
      "ttnte::solvers::StaticFreezePolicy::StaticFreezePolicy",
      "`freeze_eps` must be greater than 0");
  }
}

bool StaticFreezePolicy::should_enrich(
  double eps, double /*error*/, double /*rank_metric*/)
{
  if (!frozen_ && eps <= freeze_eps_) {
    frozen_ = true;
  }
  return !frozen_;
}

// =================================================================
// AdaptiveRevalidationPolicy

AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy(int64_t initial_period,
  int64_t probe_iterations, double growth_factor, int64_t max_period,
  double growth_tolerance)
  : initial_period_(initial_period), probe_iterations_(probe_iterations),
    growth_factor_(growth_factor), max_period_(max_period),
    growth_tolerance_(growth_tolerance), period_(initial_period)
{
  if (initial_period_ < 1) {
    throw utils::runtime_error(
      "ttnte::solvers::AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy",
      "`initial_period` must be greater than or equal to 1");
  }
  if (probe_iterations_ < 1 || probe_iterations_ > initial_period_) {
    throw utils::runtime_error(
      "ttnte::solvers::AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy",
      "`probe_iterations` must be between 1 and `initial_period`");
  }
  if (growth_factor_ <= 1.0) {
    throw utils::runtime_error(
      "ttnte::solvers::AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy",
      "`growth_factor` must be greater than 1");
  }
  if (max_period_ < initial_period_) {
    throw utils::runtime_error(
      "ttnte::solvers::AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy",
      "`max_period` must be greater than or equal to `initial_period`");
  }
  if (growth_tolerance_ < 0.0) {
    throw utils::runtime_error(
      "ttnte::solvers::AdaptiveRevalidationPolicy::AdaptiveRevalidationPolicy",
      "`growth_tolerance` must be greater than or equal to 0");
  }
}

bool AdaptiveRevalidationPolicy::should_enrich(
  double /*eps*/, double /*error*/, double rank_metric)
{
  if (!started_) {
    started_ = true;
    period_ = initial_period_;
    probe_calls_remaining_ = probe_iterations_;
    baseline_metric_ = rank_metric;
  }

  if (probe_calls_remaining_ > 0) {
    --probe_calls_remaining_;
    if (probe_calls_remaining_ == 0) {
      // The probe window just closed as of this call's result -- decide
      // backoff/reset from the growth observed since it opened.
      double rel_growth =
        (baseline_metric_ > 0.0)
          ? (rank_metric - baseline_metric_) / baseline_metric_
          : 1.0;
      period_ =
        (rel_growth > growth_tolerance_)
          ? initial_period_
          : std::min(max_period_,
              static_cast<int64_t>(std::llround(period_ * growth_factor_)));

      als_calls_remaining_ = period_ - probe_iterations_;
      if (als_calls_remaining_ <= 0) {
        // period_ collapsed to <= probe_iterations_ (e.g. right after a
        // reset) -- go straight back into another probe window.
        probe_calls_remaining_ = probe_iterations_;
        baseline_metric_ = rank_metric;
      }
    }
    return true;
  }

  // In the cheap ALS stretch of the cycle.
  --als_calls_remaining_;
  if (als_calls_remaining_ <= 0) {
    probe_calls_remaining_ = probe_iterations_;
    baseline_metric_ = rank_metric;
  }
  return false;
}

// =================================================================
// HardFreezeWrapper

HardFreezeWrapper::HardFreezeWrapper(
  EnrichmentPolicy::Ptr inner, double freeze_eps)
  : inner_(std::move(inner)), freeze_eps_(freeze_eps)
{
  if (!inner_) {
    throw utils::runtime_error(
      "ttnte::solvers::HardFreezeWrapper::HardFreezeWrapper",
      "`inner` must not be null");
  }
  if (freeze_eps_ <= 0) {
    throw utils::runtime_error(
      "ttnte::solvers::HardFreezeWrapper::HardFreezeWrapper",
      "`freeze_eps` must be greater than 0");
  }
}

bool HardFreezeWrapper::should_enrich(
  double eps, double error, double rank_metric)
{
  if (!frozen_ && eps <= freeze_eps_) {
    frozen_ = true;
  }
  if (frozen_) {
    return false;
  }
  return inner_->should_enrich(eps, error, rank_metric);
}

} // namespace ttnte::solvers
