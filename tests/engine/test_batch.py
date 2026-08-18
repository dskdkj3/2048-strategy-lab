from __future__ import annotations

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
