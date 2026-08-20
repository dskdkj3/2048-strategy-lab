"""Correctness-first multi-environment adapter.

This intentionally remains a collection of independent oracle environments;
it is not presented as a vectorized native kernel.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from strategy2048.engine.oracle import EngineSnapshot, Observation, OracleEnv, StepResult
from strategy2048.rules.core import Action, ChanceEvent


@dataclass(frozen=True, slots=True)
class BatchSnapshot:
    schema_version: str
    environments: dict[str, EngineSnapshot]

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environments": {key: value.to_json() for key, value in self.environments.items()},
        }


class ReferenceBatch:
    """Batch facade whose RNG identity is derived from environment id."""

    SNAPSHOT_SCHEMA_VERSION = "batch-snapshot-v1"

    def __init__(
        self,
        environment_ids: Iterable[str],
        *,
        root_seed: int | str = 0,
        purpose: str = "train-env",
        win_tile: int = 2048,
        max_steps: int | None = None,
    ) -> None:
        ids = tuple(str(environment_id) for environment_id in environment_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("environment ids must be unique")
        self.root_seed = root_seed
        self.purpose = purpose
        self.environment_ids = ids
        self.win_tile = win_tile
        self.max_steps = max_steps
        self.envs = {environment_id: self._new_env(environment_id) for environment_id in ids}

    def _new_env(self, environment_id: str) -> OracleEnv:
        return OracleEnv(
            root_seed=self.root_seed,
            environment_id=environment_id,
            win_tile=self.win_tile,
            max_steps=self.max_steps,
        )

    def reset(self, *, episode_ids: dict[str, int] | None = None) -> dict[str, Observation]:
        episode_ids = episode_ids or {}
        candidates = {
            environment_id: self._new_env(environment_id) for environment_id in self.environment_ids
        }
        observations = {
            environment_id: candidates[environment_id].reset(
                episode_id=episode_ids.get(environment_id, 0), purpose=self.purpose
            )
            for environment_id in self.environment_ids
        }
        self.envs = candidates
        return observations

    def step(
        self,
        actions: dict[str, Action | int | str],
        chance_events: dict[str, ChanceEvent | None] | None = None,
    ) -> dict[str, StepResult]:
        unknown = set(actions) - set(self.environment_ids)
        if unknown:
            raise KeyError(f"unknown environment ids: {sorted(unknown)}")
        events = chance_events or {}
        return {
            environment_id: self.envs[environment_id].step(
                actions[environment_id], events.get(environment_id)
            )
            for environment_id in self.environment_ids
            if environment_id in actions
        }

    def observations(self) -> dict[str, Observation]:
        return {environment_id: env.observation() for environment_id, env in self.envs.items()}

    def snapshot(self) -> BatchSnapshot:
        return BatchSnapshot(
            schema_version=self.SNAPSHOT_SCHEMA_VERSION,
            environments={key: value.snapshot() for key, value in self.envs.items()},
        )

    def restore(self, snapshot: BatchSnapshot | dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            if set(snapshot) != {"schema_version", "environments"}:
                raise ValueError("batch snapshot must contain the complete field set")
            if snapshot.get("schema_version") != self.SNAPSHOT_SCHEMA_VERSION:
                raise ValueError("unsupported batch snapshot schema")
            raw_envs = snapshot["environments"]
            if not isinstance(raw_envs, dict):
                raise ValueError("batch environments must be an object")
            snapshot = BatchSnapshot(
                schema_version=snapshot["schema_version"],
                environments={
                    key: EngineSnapshot.from_json(value) for key, value in raw_envs.items()
                },
            )
        elif not isinstance(snapshot, BatchSnapshot):
            raise ValueError("batch snapshot must be a BatchSnapshot or object")
        if snapshot.schema_version != self.SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported batch snapshot schema")
        if set(snapshot.environments) != set(self.environment_ids):
            raise ValueError("batch environment ids differ from snapshot")
        candidates = {
            environment_id: self._new_env(environment_id) for environment_id in self.environment_ids
        }
        for environment_id in self.environment_ids:
            candidates[environment_id].restore(snapshot.environments[environment_id])
        self.envs = candidates
