from __future__ import annotations

from collections import Counter

import pytest

from strategy2048.rng.stream import RNGSnapshot, ScientificRNG, derive_seed, rng_for


def test_domain_and_environment_derivation_are_stable_and_distinct() -> None:
    seed_a = derive_seed("root", "train-env", "env-a", 7)

    assert seed_a == derive_seed("root", "train-env", "env-a", 7)
    assert seed_a != derive_seed("root", "eval-env", "env-a", 7)
    assert seed_a != derive_seed("root", "train-env", "env-b", 7)


def test_snapshot_restore_replays_raw_stream_exactly() -> None:
    rng = rng_for("root", "stat-test", "snapshot", 0)
    prefix = [rng.raw_uint64() for _ in range(4)]
    snapshot = rng.snapshot()
    continuation = [rng.raw_uint64() for _ in range(10)]

    restored = ScientificRNG(snapshot.seed, purpose="stat-test")
    restored.restore(snapshot)

    assert prefix
    assert [restored.raw_uint64() for _ in range(10)] == continuation
    assert restored.lineage == snapshot.lineage


def test_legacy_rng_snapshot_without_lineage_remains_replayable() -> None:
    rng = ScientificRNG(7)
    rng.raw_uint64()
    value = rng.snapshot().to_json()
    value["schema_version"] = "rng-v1"
    value.pop("lineage")

    snapshot = RNGSnapshot.from_json(value)
    restored = ScientificRNG(snapshot.seed)
    restored.restore(snapshot)

    assert restored.counter == 1
    assert restored.lineage["purpose"] == "legacy-snapshot"


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [("seed", "7"), ("counter", 1.5), ("counter", True), ("state", [])],
)
def test_rng_snapshot_rejects_coerced_json_types(field_name, invalid_value) -> None:
    value = ScientificRNG(7).snapshot().to_json()
    value[field_name] = invalid_value

    with pytest.raises(ValueError):
        RNGSnapshot.from_json(value)


def test_rng_v2_snapshot_requires_complete_lineage() -> None:
    value = ScientificRNG(7).snapshot().to_json()
    value["lineage"] = {}

    with pytest.raises(ValueError, match="complete field set"):
        RNGSnapshot.from_json(value)


def test_rng_snapshot_rejects_unknown_fields() -> None:
    value = ScientificRNG(7).snapshot().to_json()
    value["unexpected"] = True

    with pytest.raises(ValueError, match="unknown fields"):
        RNGSnapshot.from_json(value)


def test_randbelow_is_bounded_and_statistically_uniform() -> None:
    rng = rng_for("root", "stat-test", "uniform", 0)
    counts = Counter(rng.randbelow(16) for _ in range(16_000))

    assert set(counts) == set(range(16))
    assert max(abs(count - 1000) for count in counts.values()) < 140


def test_tile_probability_statistical_smoke() -> None:
    rng = rng_for("root", "stat-test", "tile-probability", 0)
    twos = sum(rng.randbelow(10) < 9 for _ in range(20_000))

    assert abs(twos / 20_000 - 0.9) < 0.015


@pytest.mark.parametrize("upper", [0, -1, (1 << 64) + 1])
def test_randbelow_rejects_invalid_bounds(upper: int) -> None:
    with pytest.raises(ValueError):
        ScientificRNG(1).randbelow(upper)
