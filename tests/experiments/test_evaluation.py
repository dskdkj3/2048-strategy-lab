from __future__ import annotations

import pytest

from strategy2048.experiments.evaluation import (
    FrozenPolicyAgent,
    evaluate_frozen,
)
from strategy2048.learning.td import TDLearner, TupleValueFunction


def _frozen_agent() -> FrozenPolicyAgent:
    value_function = TupleValueFunction(
        tuples=((0, 1), (4, 5)),
        value_cardinality=8,
        symmetry=False,
        optimistic_total_value=20.0,
    )
    learner = TDLearner(
        value_function=value_function,
        alpha=0.1,
        gamma=1.0,
        optimistic_initialization=value_function.initial_feature_value,
    )
    return FrozenPolicyAgent(learner)


def test_frozen_evaluation_uses_separate_counters_and_preserves_state() -> None:
    agent = _frozen_agent()
    before_state_hash = agent.state_hash()
    before_table_hash = agent.table_hash()
    before_counters = agent.counters.to_json()

    result = evaluate_frozen(
        agent,
        episodes=2,
        root_seed="evaluation-root",
        purpose="eval-env",
        max_steps=10,
        episode_ids=(4, 9),
    )

    assert result["completed_episodes"] == 2
    assert result["state_unchanged"] is True
    assert result["state_hash_before"] == before_state_hash
    assert result["state_hash_after"] == before_state_hash
    assert result["table_hash_before"] == before_table_hash
    assert result["table_hash_after"] == before_table_hash
    assert result["counters_before"] == before_counters
    assert result["counters_after"] == before_counters
    counters = result["evaluation_counters"]
    assert isinstance(counters, dict)
    assert counters["action_value_calls"] > 0
    assert counters["tuple_lookups"] > 0
    assert result["metrics"]["wall_seconds"]["evaluation"] > 0
    assert result["metrics"]["wall_seconds"]["feature_value_lookup"] > 0


def test_frozen_evaluation_drops_incomplete_episode_at_deadline() -> None:
    agent = _frozen_agent()
    result = evaluate_frozen(
        agent,
        episodes=2,
        root_seed="deadline",
        max_steps=10,
        clock=lambda: 1.0,
        deadline=0.0,
    )

    assert result["completed_episodes"] == 0
    assert result["count"] == 0
    assert result["stop_reason"] == "budget_exhausted"
    assert result["partial_episode"] == 0
    assert result["state_unchanged"] is True


def test_frozen_evaluation_rejects_a_mutating_scoring_path(monkeypatch) -> None:
    agent = _frozen_agent()
    original = type(agent.learner).choose_action_read_only

    def mutate_then_score(learner, observation, *, feature_timer=None):
        learner.counters.action_value_calls += 1
        return original(learner, observation, feature_timer=feature_timer)

    monkeypatch.setattr(type(agent.learner), "choose_action_read_only", mutate_then_score)

    with pytest.raises(RuntimeError, match="frozen evaluation mutated"):
        evaluate_frozen(agent, episodes=1, root_seed="mutation", max_steps=2)
