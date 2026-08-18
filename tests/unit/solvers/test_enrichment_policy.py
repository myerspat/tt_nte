import pytest

from ttnte.solvers import (
    StaticFreezePolicy,
    AdaptiveRevalidationPolicy,
    HardFreezeWrapper,
)


def test_static_freeze_policy_enriches_until_threshold():
    policy = StaticFreezePolicy(freeze_eps=1e-3)
    assert not policy.has_frozen()
    assert policy.should_enrich(eps=1e-1, error=1.0, rank_metric=0.0)
    assert policy.should_enrich(eps=1e-2, error=0.5, rank_metric=0.0)
    assert not policy.has_frozen()

    assert not policy.should_enrich(
        eps=1e-3, error=0.1, rank_metric=0.0
    )  # eps <= freeze_eps
    assert policy.has_frozen()


def test_static_freeze_policy_never_reenables():
    """Sticky: once tripped, must not re-enable even if eps rises back above
    freeze_eps on a later call."""
    policy = StaticFreezePolicy(freeze_eps=1e-3)
    assert not policy.should_enrich(eps=1e-4, error=0.1, rank_metric=0.0)
    assert policy.has_frozen()

    assert not policy.should_enrich(eps=1.0, error=10.0, rank_metric=0.0)
    assert policy.has_frozen()


def test_static_freeze_policy_rejects_non_positive_freeze_eps():
    with pytest.raises(RuntimeError):
        StaticFreezePolicy(freeze_eps=0.0)
    with pytest.raises(RuntimeError):
        StaticFreezePolicy(freeze_eps=-1.0)


def test_adaptive_revalidation_policy_starts_enriching_every_call():
    """initial_period=1 (default) means the first call is always a probe -- identical to
    plain always-enrich AMEn until growth stops."""
    policy = AdaptiveRevalidationPolicy(initial_period=1, probe_iterations=1)
    assert policy.should_enrich(eps=1.0, error=1.0, rank_metric=100.0)


def test_adaptive_revalidation_policy_widens_when_growth_stops():
    """Repeated no-growth probes should widen the period geometrically, capped at
    max_period."""
    policy = AdaptiveRevalidationPolicy(
        initial_period=1,
        probe_iterations=1,
        growth_factor=2.0,
        max_period=8,
        growth_tolerance=0.01,
    )
    # Constant rank_metric -- every probe finds no growth.
    decisions = [
        policy.should_enrich(eps=1.0, error=1.0, rank_metric=100.0) for _ in range(20)
    ]
    assert policy.period > 1
    assert policy.period <= 8
    # Some cheap-ALS (False) decisions must have accumulated once the period
    # widened past 1.
    assert not all(decisions)


def test_adaptive_revalidation_policy_resets_on_growth():
    """A probe that finds real growth should snap the period back down to
    initial_period -- verified by driving continuous growth and checking the
    period was observed back at its minimum at some point, rather than
    assuming it stays there (steady growth keeps re-triggering the reset
    every cycle, which is itself the correct behavior)."""
    policy = AdaptiveRevalidationPolicy(
        initial_period=1,
        probe_iterations=1,
        growth_factor=2.0,
        max_period=64,
        growth_tolerance=0.01,
    )
    for _ in range(15):
        policy.should_enrich(eps=1.0, error=1.0, rank_metric=100.0)
    widened_period = policy.period
    assert widened_period > 1

    periods_seen = []
    metric = 100.0
    for _ in range(2 * widened_period + 5):
        metric *= 1.5  # steady growth every call -> every probe sees real growth
        policy.should_enrich(eps=1.0, error=1.0, rank_metric=metric)
        periods_seen.append(policy.period)

    assert min(periods_seen) == 1


def test_adaptive_revalidation_policy_small_growth_treated_as_noise():
    """Growth below growth_tolerance must NOT reset the period -- avoids resetting the
    backoff on SVD-noise-level upticks."""
    policy = AdaptiveRevalidationPolicy(
        initial_period=1,
        probe_iterations=1,
        growth_factor=2.0,
        max_period=64,
        growth_tolerance=0.05,
    )
    for _ in range(5):
        policy.should_enrich(eps=1.0, error=1.0, rank_metric=100.0)
    widened_period = policy.period
    assert widened_period > 1

    # 1% growth -- below the 5% tolerance -- should not reset.
    for _ in range(widened_period):
        policy.should_enrich(eps=1.0, error=1.0, rank_metric=101.0)
    assert policy.period >= widened_period


def test_adaptive_revalidation_policy_never_freezes_on_its_own():
    policy = AdaptiveRevalidationPolicy()
    for _ in range(50):
        policy.should_enrich(eps=1e-10, error=1e-10, rank_metric=100.0)
    assert not policy.has_frozen()


def test_adaptive_revalidation_policy_rejects_invalid_args():
    with pytest.raises(RuntimeError):
        AdaptiveRevalidationPolicy(initial_period=0)
    with pytest.raises(RuntimeError):
        AdaptiveRevalidationPolicy(initial_period=1, probe_iterations=2)
    with pytest.raises(RuntimeError):
        AdaptiveRevalidationPolicy(growth_factor=1.0)
    with pytest.raises(RuntimeError):
        AdaptiveRevalidationPolicy(initial_period=4, max_period=1)
    with pytest.raises(RuntimeError):
        AdaptiveRevalidationPolicy(growth_tolerance=-0.1)


def test_hard_freeze_wrapper_delegates_before_threshold():
    inner = AdaptiveRevalidationPolicy(initial_period=1, probe_iterations=1)
    wrapper = HardFreezeWrapper(inner=inner, freeze_eps=1e-3)
    assert wrapper.should_enrich(eps=1e-1, error=1.0, rank_metric=100.0)
    assert not wrapper.has_frozen()


def test_hard_freeze_wrapper_freezes_permanently_past_threshold():
    inner = AdaptiveRevalidationPolicy(initial_period=1, probe_iterations=1)
    wrapper = HardFreezeWrapper(inner=inner, freeze_eps=1e-3)
    assert not wrapper.should_enrich(eps=1e-4, error=0.1, rank_metric=100.0)
    assert wrapper.has_frozen()
    # Sticky -- even if eps rises back above freeze_eps.
    assert not wrapper.should_enrich(eps=1.0, error=1.0, rank_metric=100.0)


def test_hard_freeze_wrapper_rejects_null_inner_or_bad_freeze_eps():
    with pytest.raises(RuntimeError):
        HardFreezeWrapper(inner=None, freeze_eps=1e-3)
    with pytest.raises(RuntimeError):
        HardFreezeWrapper(inner=AdaptiveRevalidationPolicy(), freeze_eps=0.0)
