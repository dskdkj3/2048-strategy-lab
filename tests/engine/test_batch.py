from __future__ import annotations

import pytest

from strategy2048.engine.batch import ReferenceBatch


def test_environment_identity_is_independent_of_batch_order() -> None:
    first = ReferenceBatch(("a", "b"), root_seed="batch-root")
    second = ReferenceBatch(("b", "a"), root_seed="batch-root")

    first_observations = first.reset(episode_ids={"a": 3, "b": 4})
    second_observations = second.reset(episode_ids={"a": 3, "b": 4})

    assert first_observations["a"].board == second_observations["a"].board
    assert first_observations["b"].board == second_observations["b"].board
    assert first.envs["a"].rng.seed == second.envs["a"].rng.seed
    assert first.envs["b"].rng.seed == second.envs["b"].rng.seed


def test_batch_snapshot_round_trip() -> None:
    batch = ReferenceBatch(("a", "b"), root_seed="batch-snapshot")
    batch.reset()
    snapshot = batch.snapshot()
    expected = batch.observations()

    batch.reset(episode_ids={"a": 2, "b": 2})
    batch.restore(snapshot.to_json())

    assert batch.observations() == expected


def test_batch_snapshot_json_key_order_is_not_significant() -> None:
    batch = ReferenceBatch(("a", "b"), root_seed="batch-snapshot-order")
    batch.reset()
    snapshot = batch.snapshot().to_json()
    snapshot["environments"] = dict(reversed(list(snapshot["environments"].items())))

    batch.reset(episode_ids={"a": 2, "b": 2})
    batch.restore(snapshot)

    assert batch.observations()["a"].board == tuple(snapshot["environments"]["a"]["board"])


def test_batch_reset_failure_is_atomic() -> None:
    batch = ReferenceBatch(("a", "b"), root_seed="batch-reset-atomic")
    batch.reset(episode_ids={"a": 1, "b": 1})
    before = batch.snapshot()

    with pytest.raises(ValueError, match="episode_id"):
        batch.reset(episode_ids={"a": 2, "b": -1})

    assert batch.snapshot() == before


def test_batch_restore_failure_is_atomic() -> None:
    target = ReferenceBatch(("a", "b"), root_seed="batch-restore-target")
    target.reset(episode_ids={"a": 1, "b": 1})
    before = target.snapshot()

    source = ReferenceBatch(("a", "b"), root_seed="batch-restore-source")
    source.reset(episode_ids={"a": 2, "b": 2})
    invalid = source.snapshot().to_json()
    invalid["environments"]["b"]["terminated"] = not invalid["environments"]["b"]["terminated"]

    with pytest.raises(ValueError, match="termination flag"):
        target.restore(invalid)

    assert target.snapshot() == before
