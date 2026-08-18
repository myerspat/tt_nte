#pragma once

#include <cstdint>
#include <memory>

namespace ttnte::solvers {

/// @brief Decides, once per outer DD (Schwarz) iteration, whether an
/// AMEnSolver should run with enrichment active (adaptive rank, full AMEn)
/// or disabled (fixed rank, pure ALS + proximal regularization). Queried
/// from `AMEnSolver::update_convergence_criteria()` after each iteration's
/// global, MPI-reduced, current-weighted Schwarz error is available (see
/// `DDSolver::step()`).
///
/// Every implementation must key off that same global signal (or a direct,
/// also-global-aggregated probe of the solver's own enrichment machinery, as
/// `AdaptiveRevalidationPolicy` does via `rank_metric`) -- never a local
/// per-patch heuristic (e.g. one patch's own rank-growth history in
/// isolation). A local signal can plateau simply because a patch's still-
/// wrong incoming boundary data happens to be well represented at its
/// current rank, then break once a neighbor's correction actually arrives;
/// see `ttnte/CLAUDE.md`'s DD/AMEn notes.
class EnrichmentPolicy {
public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<EnrichmentPolicy>;

  virtual ~EnrichmentPolicy() = default;

  // =================================================================
  // Public methods
  /// @brief Called once per outer DD iteration with this iteration's forced
  /// local tolerance, raw global error, and an aggregate rank/size metric.
  /// Returns whether enrichment should be active for the NEXT `solve()`
  /// call.
  /// @param eps `max(eps_floor_, eps_forcing_ * min_error_)` -- AMEnSolver's
  /// own forced truncation tolerance, already tightened toward eps_floor_ as
  /// the global error improves.
  /// @param error This iteration's raw global, MPI-reduced, current-weighted
  /// Schwarz error (`DDSolver::step()`).
  /// @param rank_metric An aggregate measure of solution size (e.g. total TT
  /// state element count, summed across every local patch and MPI-reduced --
  /// see `DDSolver::step()`) as of the solve that produced `error`.
  /// Policies that don't need it may ignore it.
  virtual bool should_enrich(double eps, double error, double rank_metric) = 0;

  // =================================================================
  // Public getters / setters
  /// @return Whether this policy has ever disabled enrichment (diagnostic --
  /// distinct from `should_enrich`'s current per-iteration answer, since a
  /// policy may re-enable enrichment later).
  virtual bool has_frozen() const noexcept = 0;
};

/// @brief One-shot, sticky freeze: once `eps` drops to or below `freeze_eps`,
/// permanently disables enrichment. Never re-enables, even if `eps` were to
/// somehow increase again.
class StaticFreezePolicy final : public EnrichmentPolicy {
protected:
  // =================================================================
  // Protected constructors
  explicit StaticFreezePolicy(double freeze_eps);

public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<StaticFreezePolicy>;

  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new StaticFreezePolicy.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new StaticFreezePolicy(std::forward<Args>(args)...));
  }

  bool should_enrich(
    double eps, double error, double rank_metric) override final;

  // =================================================================
  // Public getters / setters
  bool has_frozen() const noexcept override final { return frozen_; }
  /// @return The threshold `eps` must drop to or below to trigger freezing.
  double get_freeze_eps() const noexcept { return freeze_eps_; }

private:
  // =================================================================
  // Private data
  double freeze_eps_;
  bool frozen_ = false;
};

/// @brief Self-pacing, threshold-free enrichment scheduler: no `eps`-based
/// trigger at all. Starts checking (enriching) every iteration -- identical
/// to plain, always-enrich AMEn, so it carries no more risk than the
/// no-policy default -- then widens the gap between checks once a check
/// finds that `rank_metric` didn't meaningfully grow, and snaps back to
/// checking frequently the moment a check finds real growth again.
///
/// This exists because a fixed cadence (the earlier, now-removed
/// PeriodicRevalidationPolicy) has to guess a cadence that's safe for
/// whatever the DD iteration's dynamics happen to be doing -- too sparse
/// while things are still actively evolving, and oscillation followed even
/// through later re-enrichment. Backing off only once the data itself says
/// nothing changed removes that guess.
class AdaptiveRevalidationPolicy final : public EnrichmentPolicy {
protected:
  // =================================================================
  // Protected constructors
  AdaptiveRevalidationPolicy(int64_t initial_period = 1,
    int64_t probe_iterations = 1, double growth_factor = 2.0,
    int64_t max_period = 64, double growth_tolerance = 0.01);

public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<AdaptiveRevalidationPolicy>;

  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new AdaptiveRevalidationPolicy.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new AdaptiveRevalidationPolicy(std::forward<Args>(args)...));
  }

  bool should_enrich(
    double eps, double error, double rank_metric) override final;

  // =================================================================
  // Public getters / setters
  /// @return Always false -- this policy never permanently freezes on its
  /// own; compose it with `HardFreezeWrapper` for that.
  bool has_frozen() const noexcept override final { return false; }
  /// @return The starting/reset check period (iterations per cycle).
  int64_t get_initial_period() const noexcept { return initial_period_; }
  /// @return The number of consecutive enriching calls per cycle.
  int64_t get_probe_iterations() const noexcept { return probe_iterations_; }
  /// @return The multiplicative factor the period widens by after a cycle
  /// finds no meaningful growth.
  double get_growth_factor() const noexcept { return growth_factor_; }
  /// @return The upper bound the period is capped at.
  int64_t get_max_period() const noexcept { return max_period_; }
  /// @return The relative `rank_metric` increase, below which a cycle counts
  /// as "no growth" (backoff widens rather than resets).
  double get_growth_tolerance() const noexcept { return growth_tolerance_; }
  /// @return The current (adaptive) period -- diagnostic.
  int64_t get_period() const noexcept { return period_; }

private:
  // =================================================================
  // Private data
  int64_t initial_period_;
  int64_t probe_iterations_;
  double growth_factor_;
  int64_t max_period_;
  double growth_tolerance_;

  bool started_ = false;
  int64_t period_;
  int64_t probe_calls_remaining_ = 0;
  int64_t als_calls_remaining_ = 0;
  double baseline_metric_ = -1.0;
};

/// @brief Wraps another EnrichmentPolicy, forcing a permanent, sticky freeze
/// once `eps` drops to or below `freeze_eps` -- regardless of what the
/// wrapped policy would otherwise decide. Lets an always-active, self-pacing
/// policy (e.g. AdaptiveRevalidationPolicy) benefit from the same one-shot,
/// proven-safe hard stop `StaticFreezePolicy` uses on its own, without
/// duplicating that logic into every policy that wants it -- e.g.
/// `HardFreezeWrapper(AdaptiveRevalidationPolicy(...), freeze_eps)` gets
/// cheap self-paced checking during the still-dynamic early phase, and the
/// same known-safe permanent freeze late, without indefinite occasional
/// re-checking cost once it's no longer needed.
class HardFreezeWrapper final : public EnrichmentPolicy {
protected:
  // =================================================================
  // Protected constructors
  HardFreezeWrapper(EnrichmentPolicy::Ptr inner, double freeze_eps);

public:
  // =================================================================
  // Public types
  using Ptr = std::shared_ptr<HardFreezeWrapper>;

  // =================================================================
  // Public methods
  /// @brief Create a shared pointer to a new HardFreezeWrapper.
  template<typename... Args>
  static Ptr create(Args&&... args)
  {
    return Ptr(new HardFreezeWrapper(std::forward<Args>(args)...));
  }

  bool should_enrich(
    double eps, double error, double rank_metric) override final;

  // =================================================================
  // Public getters / setters
  bool has_frozen() const noexcept override final
  {
    return frozen_ || inner_->has_frozen();
  }
  /// @return The threshold `eps` must drop to or below to trigger the
  /// permanent freeze.
  double get_freeze_eps() const noexcept { return freeze_eps_; }
  /// @return The wrapped policy.
  const EnrichmentPolicy::Ptr& get_inner() const noexcept { return inner_; }

private:
  // =================================================================
  // Private data
  EnrichmentPolicy::Ptr inner_;
  double freeze_eps_;
  bool frozen_ = false;
};

} // namespace ttnte::solvers
