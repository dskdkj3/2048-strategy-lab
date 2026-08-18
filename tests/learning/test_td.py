from __future__ import annotations

import numpy as np

from strategy2048.experiments.training import train_td
from strategy2048.learning.td import TD1PAgent, TDLearner, TupleValueFunction


def _agent(*, initial_value: float = 0.0) -> TD1PAgent:
    value_function = TupleValueFunction(
        tuples=((0, 1), (4, 5)),
        value_cardinality=8,
        symmetry=False,
        initial_value=initial_value,
    )
    return TD1PAgent(
        TDLearner(
            value_function=value_function,
            alpha=0.1,
            gamma=1.0,
            optimistic_initialization=initial_value,
        )
    )


def test_zero_and_optimistic_initialization_are_separate_configs() -> None:
    zero = _agent()
    optimistic = _agent(initial_value=1.0)

    assert np.all(zero.learner.value_function.tables == 0.0)
    assert np.all(optimistic.learner.value_function.tables == 1.0)
    assert zero.knowledge_manifest().initialization["source"] == "zero"
    assert optimistic.knowledge_manifest().initialization["source"] == "optimistic"


def test_checkpoint_round_trip_uses_explicit_arrays(tmp_path) -> None:
    agent = _agent()
    train_td(agent, episodes=1, root_seed="checkpoint", max_steps=20)
    expected_hash = agent.learner.value_function.state_hash()
    expected_counters = agent.counters.to_json()
    agent.save_checkpoint(tmp_path, 1, config_hash="config")

    restored = _agent()
    restored.restore_checkpoint(tmp_path, 1, config_hash="config")

    assert restored.learner.value_function.state_hash() == expected_hash
    assert restored.counters.to_json() == expected_counters


def test_episode_boundary_resume_matches_uninterrupted_training(tmp_path) -> None:
    uninterrupted = _agent()
    train_td(uninterrupted, episodes=4, root_seed="resume", max_steps=20)

    partial = _agent()
    train_td(partial, episodes=2, root_seed="resume", max_steps=20)
    partial.save_checkpoint(tmp_path, 2, config_hash="resume-config")
    resumed = _agent()
    resumed.restore_checkpoint(tmp_path, 2, config_hash="resume-config")
    train_td(resumed, episodes=2, root_seed="resume", max_steps=20, start_episode=2)

    assert (
        resumed.learner.value_function.state_hash()
        == uninterrupted.learner.value_function.state_hash()
    )
    assert resumed.counters.to_json() == uninterrupted.counters.to_json()
