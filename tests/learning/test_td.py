from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.training import train_td
from strategy2048.learning.td import TD1PAgent, TDLearner, TupleValueFunction

REPOSITORY_ROOT = Path(__file__).parents[2]


def _agent(*, initial_value: float = 0.0, alpha: float = 0.1) -> TD1PAgent:
    value_function = TupleValueFunction(
        tuples=((0, 1), (4, 5)),
        value_cardinality=8,
        symmetry=False,
        initial_value=initial_value,
    )
    return TD1PAgent(
        TDLearner(
            value_function=value_function,
            alpha=alpha,
            gamma=1.0,
            optimistic_initialization=initial_value,
        )
    )


def _snapshot(seed: str, episode_id: int = 0):
    env = OracleEnv(root_seed=seed, environment_id=f"checkpoint-{episode_id}")
    env.reset(episode_id=episode_id)
    return env.snapshot()


def _advance(agent: TD1PAgent, env: OracleEnv, steps: int) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    observation = env.observation()
    for _ in range(steps):
        if observation.terminated or observation.truncated:
            break
        action = agent.act(observation)
        result = env.step(action)
        agent.observe(result, result.observation)
        results.append(result.to_json())
        observation = result.observation
    return results


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
    expected_hash = agent.learner.state_hash()
    expected_counters = agent.counters.to_json()
    expected_environment = _snapshot("checkpoint")
    _, metadata_path = agent.save_checkpoint(
        tmp_path,
        1,
        config_hash="config",
        environment_snapshot=expected_environment,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    schema = json.loads(
        (REPOSITORY_ROOT / "schemas/checkpoint-meta.v2.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(metadata, schema)

    restored = _agent()
    restored_environment = restored.restore_checkpoint(tmp_path, 1, config_hash="config")

    assert restored.learner.state_hash() == expected_hash
    assert restored.counters.to_json() == expected_counters
    assert restored_environment == expected_environment


def test_checkpoint_v2_schema_rejects_invalid_learner_config_type(tmp_path) -> None:
    agent = _agent()
    _, metadata_path = agent.save_checkpoint(
        tmp_path,
        1,
        config_hash="schema-learner-config",
        environment_snapshot=_snapshot("schema-learner-config"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["learner_config"]["alpha"] = []
    schema = json.loads(
        (REPOSITORY_ROOT / "schemas/checkpoint-meta.v2.schema.json").read_text(encoding="utf-8")
    )

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(metadata, schema)


def test_episode_boundary_resume_matches_uninterrupted_training(tmp_path) -> None:
    uninterrupted = _agent()
    train_td(uninterrupted, episodes=4, root_seed="resume", max_steps=20)

    partial = _agent()
    train_td(partial, episodes=2, root_seed="resume", max_steps=20)
    partial.save_checkpoint(
        tmp_path,
        2,
        config_hash="resume-config",
        environment_snapshot=_snapshot("resume", 1),
    )
    resumed = _agent()
    resumed.restore_checkpoint(tmp_path, 2, config_hash="resume-config")
    train_td(resumed, episodes=2, root_seed="resume", max_steps=20, start_episode=2)

    assert resumed.learner.state_hash() == uninterrupted.learner.state_hash()
    assert resumed.counters.to_json() == uninterrupted.counters.to_json()


def test_mid_episode_checkpoint_restores_environment_rng_and_learner(tmp_path) -> None:
    original_agent = _agent()
    original_env = OracleEnv(root_seed="mid-episode", environment_id="train-mid", max_steps=50)
    original_env.reset(episode_id=3)
    _advance(original_agent, original_env, 4)
    original_agent.save_checkpoint(
        tmp_path,
        4,
        config_hash="mid-config",
        environment_snapshot=original_env.snapshot(),
    )

    expected_results = _advance(original_agent, original_env, 6)

    restored_agent = _agent()
    restored_snapshot = restored_agent.restore_checkpoint(tmp_path, 4, config_hash="mid-config")
    restored_env = OracleEnv(root_seed="other", environment_id="other", max_steps=50)
    restored_env.restore(restored_snapshot)
    actual_results = _advance(restored_agent, restored_env, 6)

    assert actual_results == expected_results
    assert restored_agent.learner.state_hash() == original_agent.learner.state_hash()


def test_checkpoint_restore_failure_does_not_mutate_learner(tmp_path) -> None:
    source = _agent()
    train_td(source, episodes=1, root_seed="corrupt-source", max_steps=10)
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="atomic-config",
        environment_snapshot=_snapshot("corrupt-source"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["state_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target = _agent()
    train_td(target, episodes=1, root_seed="target-state", max_steps=6)
    before_hash = target.learner.state_hash()
    before_tables = target.learner.value_function.tables.copy()

    with pytest.raises(ValueError, match="state hash"):
        target.restore_checkpoint(tmp_path, 1, config_hash="atomic-config")

    assert target.learner.state_hash() == before_hash
    assert np.array_equal(target.learner.value_function.tables, before_tables)


def test_checkpoint_restore_rejects_learner_config_before_mutation(tmp_path) -> None:
    source = _agent()
    source.save_checkpoint(
        tmp_path,
        1,
        config_hash="config-match",
        environment_snapshot=_snapshot("config-match"),
    )
    target = _agent(alpha=0.2)
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError, match="learner config"):
        target.restore_checkpoint(tmp_path, 1, config_hash="config-match")

    assert target.learner.state_hash() == before_hash


def test_checkpoint_restore_rejects_environment_schema_before_mutation(tmp_path) -> None:
    source = _agent()
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="environment-schema",
        environment_snapshot=_snapshot("environment-schema"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["environment"]["schema_version"] = "engine-snapshot-unknown"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = _agent()
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError, match="environment schema"):
        target.restore_checkpoint(tmp_path, 1, config_hash="environment-schema")

    assert target.learner.state_hash() == before_hash


def test_checkpoint_restore_rejects_invalid_rng_state_before_mutation(tmp_path) -> None:
    source = _agent()
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="environment-rng",
        environment_snapshot=_snapshot("environment-rng"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["environment"]["rng"]["state"] = {}
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = _agent()
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError, match="invalid RNG bit-generator state"):
        target.restore_checkpoint(tmp_path, 1, config_hash="environment-rng")

    assert target.learner.state_hash() == before_hash


def test_checkpoint_restore_rejects_legacy_rng_schema_in_v2_metadata(tmp_path) -> None:
    source = _agent()
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="legacy-rng-schema",
        environment_snapshot=_snapshot("legacy-rng-schema"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["environment"]["rng"]["schema_version"] = "rng-v1"
    metadata["environment"]["rng"].pop("lineage")
    with np.load(tmp_path / "1.npz", allow_pickle=False) as archive:
        tables = archive["tables"].copy()
    hash_payload = dict(metadata)
    hash_payload.pop("checkpoint_hash")
    metadata["checkpoint_hash"] = TDLearner._checkpoint_hash(hash_payload, tables)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    target = _agent()
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError, match="checkpoint RNG schema"):
        target.restore_checkpoint(tmp_path, 1, config_hash="legacy-rng-schema")

    assert target.learner.state_hash() == before_hash


@pytest.mark.parametrize(
    ("counter_name", "invalid_value"),
    [("updates", 1.5), ("tuple_updates", True), ("td_error_abs_sum", "0.0")],
)
def test_checkpoint_restore_rejects_coerced_counter_types_before_mutation(
    tmp_path, counter_name, invalid_value
) -> None:
    source = _agent()
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="strict-counters",
        environment_snapshot=_snapshot("strict-counters"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["counters"][counter_name] = invalid_value
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = _agent()
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError):
        target.restore_checkpoint(tmp_path, 1, config_hash="strict-counters")

    assert target.learner.state_hash() == before_hash


def test_checkpoint_restore_rejects_float_array_shape_before_mutation(tmp_path) -> None:
    source = _agent()
    _, metadata_path = source.save_checkpoint(
        tmp_path,
        1,
        config_hash="strict-shape",
        environment_snapshot=_snapshot("strict-shape"),
    )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["array_shape"] = [float(dimension) for dimension in metadata["array_shape"]]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    target = _agent()
    before_hash = target.learner.state_hash()

    with pytest.raises(ValueError, match="shape must contain integers"):
        target.restore_checkpoint(tmp_path, 1, config_hash="strict-shape")

    assert target.learner.state_hash() == before_hash
