from __future__ import annotations

from strategy2048.experiments.metrics import PHASE_TIMERS, Metrics


def test_metrics_rates_use_explicit_phase_denominators() -> None:
    metrics = Metrics()
    metrics.increment("env_steps", 20)
    metrics.increment("games", 2)
    metrics.increment("action_value_calls", 40)
    metrics.increment("tuple_lookups", 320)
    metrics.increment("updates", 10)
    metrics.add_wall("rules", 2.0)
    metrics.add_wall("action_selection", 4.0)
    metrics.add_wall("feature_value_lookup", 8.0)
    metrics.add_wall("td_update", 5.0)
    metrics.add_wall("end_to_end", 10.0)

    record = metrics.snapshot()

    assert record["rates"]["rules_env_steps_per_second"] == 10.0
    assert record["rates"]["end_to_end_env_steps_per_second"] == 2.0
    assert record["rates"]["action_value_calls_per_second"] == 10.0
    assert record["rates"]["tuple_lookups_per_second"] == 40.0
    assert record["rates"]["updates_per_second"] == 2.0
    assert record["rates"]["games_per_hour"] == 720.0
    assert set(PHASE_TIMERS) <= set(record["wall_seconds"])
    assert record["rate_semantics"]["tuple_lookups_per_second"]["denominator"] == (
        "wall_seconds.feature_value_lookup"
    )
