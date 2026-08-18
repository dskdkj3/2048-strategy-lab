from __future__ import annotations

from strategy2048.agents.baselines import RandomAgent
from strategy2048.profiling.profile import benchmark, profile_training


def test_benchmark_separates_rules_and_end_to_end_metrics() -> None:
    factory_calls: list[int] = []

    def factory(episode_id: int) -> RandomAgent:
        factory_calls.append(episode_id)
        return RandomAgent("profile", f"agent-{episode_id}")

    result = benchmark(factory, episodes=2, root_seed="profile", max_steps=20)

    assert factory_calls == [0, 1]
    assert result["metrics"]["counters"]["games"] == 2
    assert result["metrics"]["counters"]["env_steps"] > 0
    assert result["metrics"]["rates"]["env_steps_per_second"] > 0


def test_cprofile_artifacts_are_written(tmp_path) -> None:
    result = profile_training(
        lambda: {"work": "bounded"},
        output_dir=tmp_path,
        name="smoke",
    )

    assert (tmp_path / "smoke.prof").is_file()
    assert (tmp_path / "summary.json").is_file()
    assert result["native_core_gate"]["recommendation"] == "continue-python-reference"
