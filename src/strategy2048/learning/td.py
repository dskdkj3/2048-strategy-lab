"""Configurable afterstate n-tuple value function and TD(0) learner."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from strategy2048.agents.protocol import EvaluationMode
from strategy2048.engine.oracle import EngineSnapshot, Observation, OracleEnv, StepResult
from strategy2048.experiments.artifacts import KnowledgeManifest, canonical_json
from strategy2048.rng.stream import RNG_SCHEMA_VERSION
from strategy2048.rules.core import Action, Board, legal_actions, move_without_spawn, validate_board

CoordinateTuple = tuple[int, ...]
DEFAULT_TUPLES: tuple[CoordinateTuple, ...] = (
    (0, 1, 4, 5),
    (1, 2, 5, 6),
    (2, 3, 6, 7),
    (4, 5, 8, 9),
    (5, 6, 9, 10),
    (8, 9, 12, 13),
    (9, 10, 13, 14),
    (0, 4, 8, 12),
)


def _coordinate(row: int, column: int) -> int:
    return row * 4 + column


def _transform_index(index: int, transform: int) -> int:
    row, column = divmod(index, 4)
    if transform >= 4:
        column = 3 - column
        transform -= 4
    for _ in range(transform):
        row, column = column, 3 - row
    return _coordinate(row, column)


def _transform_board(board: Board, transform: int) -> Board:
    output = [0] * 16
    for index, value in enumerate(board):
        output[_transform_index(index, transform)] = value
    return tuple(output)


@dataclass(slots=True)
class LearningCounters:
    action_value_calls: int = 0
    tuple_lookups: int = 0
    updates: int = 0
    tuple_updates: int = 0
    td_error_abs_sum: float = 0.0

    def to_json(self) -> dict[str, object]:
        return {
            "action_value_calls": self.action_value_calls,
            "tuple_lookups": self.tuple_lookups,
            "updates": self.updates,
            "tuple_updates": self.tuple_updates,
            "td_error_abs_sum": self.td_error_abs_sum,
        }

    @classmethod
    def from_json(cls, value: object) -> LearningCounters:
        if not isinstance(value, dict):
            raise ValueError("learning counters must be an object")
        required = {
            "action_value_calls",
            "tuple_lookups",
            "updates",
            "tuple_updates",
            "td_error_abs_sum",
        }
        if set(value) != required:
            raise ValueError("learning counters must contain the complete field set")
        integer_fields = (
            "action_value_calls",
            "tuple_lookups",
            "updates",
            "tuple_updates",
        )
        if any(type(value[field_name]) is not int for field_name in integer_fields):
            raise ValueError("learning count fields must be integers")
        error_sum = value["td_error_abs_sum"]
        if isinstance(error_sum, bool) or not isinstance(error_sum, (int, float)):
            raise ValueError("learning td_error_abs_sum must be a number")
        counters = cls(
            action_value_calls=value["action_value_calls"],
            tuple_lookups=value["tuple_lookups"],
            updates=value["updates"],
            tuple_updates=value["tuple_updates"],
            td_error_abs_sum=float(error_sum),
        )
        if (
            counters.action_value_calls < 0
            or counters.tuple_lookups < 0
            or counters.updates < 0
            or counters.tuple_updates < 0
            or counters.td_error_abs_sum < 0
            or not math.isfinite(counters.td_error_abs_sum)
        ):
            raise ValueError("learning counters must be finite and non-negative")
        return counters


class TupleValueFunction:
    """Sum of explicit NumPy tuple tables with optional D4 board orbit."""

    def __init__(
        self,
        tuples: Sequence[CoordinateTuple] = DEFAULT_TUPLES,
        *,
        value_cardinality: int = 16,
        symmetry: bool = True,
        initial_value: float = 0.0,
    ) -> None:
        if not tuples:
            raise ValueError("at least one tuple is required")
        normalized = tuple(tuple(int(index) for index in item) for item in tuples)
        if any(not item or any(index < 0 or index >= 16 for index in item) for item in normalized):
            raise ValueError("tuple coordinates must contain board indices 0..15")
        if value_cardinality < 2:
            raise ValueError("value_cardinality must be at least two")
        self.tuples = normalized
        self.value_cardinality = value_cardinality
        self.symmetry = symmetry
        self.initial_value = float(initial_value)
        table_size = value_cardinality ** len(normalized[0])
        if any(len(item) != len(normalized[0]) for item in normalized):
            raise ValueError("all tuples must have the same length")
        self.tables = np.full((len(normalized), table_size), self.initial_value, dtype=np.float64)
        self.counters = LearningCounters()

    @property
    def tuple_length(self) -> int:
        return len(self.tuples[0])

    @property
    def symmetry_orbit_size(self) -> int:
        return 8 if self.symmetry else 1

    def config(self) -> dict[str, Any]:
        return {
            "tuple_source": "default-general-local"
            if self.tuples == DEFAULT_TUPLES
            else "explicit",
            "tuples": [list(item) for item in self.tuples],
            "tuple_length": self.tuple_length,
            "value_cardinality": self.value_cardinality,
            "symmetry": self.symmetry,
            "initial_value": self.initial_value,
        }

    def _index(self, board: Board, coordinates: CoordinateTuple) -> int:
        index = 0
        multiplier = 1
        for coordinate in coordinates:
            exponent = min(board[coordinate], self.value_cardinality - 1)
            index += exponent * multiplier
            multiplier *= self.value_cardinality
        return index

    def _feature_indices(self, board: Sequence[int]) -> list[tuple[int, int]]:
        normalized = validate_board(board)
        transforms = range(8) if self.symmetry else range(1)
        features: list[tuple[int, int]] = []
        for transform in transforms:
            transformed = _transform_board(normalized, transform)
            for tuple_index, coordinates in enumerate(self.tuples):
                features.append((tuple_index, self._index(transformed, coordinates)))
        return features

    def value(self, board: Sequence[int]) -> float:
        total = 0.0
        for tuple_index, feature_index in self._feature_indices(board):
            total += float(self.tables[tuple_index, feature_index])
            self.counters.tuple_lookups += 1
        return total

    def update(self, board: Sequence[int], target: float, alpha: float) -> float:
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        features = self._feature_indices(board)
        current = sum(
            float(self.tables[tuple_index, feature_index])
            for tuple_index, feature_index in features
        )
        error = float(target) - current
        per_feature = alpha * error / len(features)
        for tuple_index, feature_index in features:
            self.tables[tuple_index, feature_index] += per_feature
            self.counters.tuple_updates += 1
        self.counters.updates += 1
        self.counters.td_error_abs_sum += abs(error)
        return error

    def state_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(canonical_json(self.config()).encode("utf-8"))
        digest.update(self.tables.tobytes(order="C"))
        return digest.hexdigest()


@dataclass(slots=True)
class TDLearner:
    value_function: TupleValueFunction = field(default_factory=TupleValueFunction)
    alpha: float = 0.1
    gamma: float = 1.0
    optimistic_initialization: float = 0.0
    agent_type: str = "discovery"

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.alpha > 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.gamma < 0 or self.gamma > 1:
            raise ValueError("gamma must be in [0, 1]")
        if self.optimistic_initialization != self.value_function.initial_value:
            raise ValueError("optimistic initialization must match value table initialization")

    @property
    def counters(self) -> LearningCounters:
        return self.value_function.counters

    def action_value(self, board: Sequence[int], action: Action) -> float:
        self.counters.action_value_calls += 1
        move = move_without_spawn(board, action)
        return self.value_function.value(move.afterstate)

    def choose_action(self, observation: Observation) -> Action:
        if not observation.legal_actions:
            raise RuntimeError("TD learner received an observation with no legal action")
        scored = [
            (self.action_value(observation.board, action), -int(action), action)
            for action in observation.legal_actions
        ]
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def next_value(self, board: Board) -> float:
        actions = legal_actions(board)
        if not actions:
            return 0.0
        return max(self.action_value(board, action) for action in actions)

    def observe(self, transition: StepResult, next_observation: Observation) -> float:
        if not transition.valid:
            return 0.0
        continuation = (
            0.0
            if next_observation.terminated or next_observation.truncated
            else self.next_value(next_observation.board)
        )
        target = float(transition.score_delta) + self.gamma * continuation
        return self.value_function.update(transition.afterstate, target, self.alpha)

    def knowledge_manifest(self) -> KnowledgeManifest:
        return KnowledgeManifest(
            experiment_kind=self.agent_type,
            observation={"source": "official_board", "fields": ["board", "legal_actions"]},
            reward={"source": "official_score_delta", "terminal_target": "zero_continuation"},
            features={"source": "tuple_value_function", **self.value_function.config()},
            initialization={
                "source": "optimistic" if self.optimistic_initialization else "zero",
                "value": self.optimistic_initialization,
            },
            curriculum={"source": "none"},
            checkpoint={
                "source": "learner_and_environment_state",
                "format": "npz-json",
                "schema_version": "checkpoint-meta-v2",
            },
            demonstrations={"source": "none"},
            search={"source": "none", "nodes": 0},
            tablebase={"source": "none"},
            detectors={"source": "none"},
        )

    def learner_config(self) -> dict[str, Any]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "optimistic_initialization": self.optimistic_initialization,
            "value_function": self.value_function.config(),
        }

    def _state_hash_for(self, tables: np.ndarray, counters: LearningCounters) -> str:
        digest = hashlib.sha256()
        digest.update(
            canonical_json(
                {
                    "agent_type": self.agent_type,
                    "learner_config": self.learner_config(),
                    "counters": counters.to_json(),
                }
            ).encode("utf-8")
        )
        digest.update(tables.tobytes(order="C"))
        return digest.hexdigest()

    def state_hash(self) -> str:
        return self._state_hash_for(self.value_function.tables, self.counters)

    @staticmethod
    def _checkpoint_hash(metadata: dict[str, Any], tables: np.ndarray) -> str:
        digest = hashlib.sha256()
        digest.update(canonical_json(metadata).encode("utf-8"))
        digest.update(tables.tobytes(order="C"))
        return digest.hexdigest()

    def checkpoint_metadata(
        self,
        config_hash: str,
        environment_snapshot: EngineSnapshot,
    ) -> dict[str, Any]:
        metadata = {
            "schema_version": "checkpoint-meta-v2",
            "agent_type": self.agent_type,
            "config_hash": config_hash,
            "learner_config": self.learner_config(),
            "array_shape": list(self.value_function.tables.shape),
            "array_dtype": str(self.value_function.tables.dtype),
            "counters": self.counters.to_json(),
            "state_hash": self.state_hash(),
            "environment": environment_snapshot.to_json(),
        }
        metadata["checkpoint_hash"] = self._checkpoint_hash(metadata, self.value_function.tables)
        return metadata

    def save_checkpoint(
        self,
        directory: str | Path,
        step: int,
        *,
        config_hash: str,
        environment_snapshot: EngineSnapshot,
    ) -> tuple[Path, Path]:
        if step < 0:
            raise ValueError("checkpoint step must be non-negative")
        destination = Path(directory)
        destination.mkdir(parents=True, exist_ok=True)
        array_path = destination / f"{step}.npz"
        metadata_path = destination / f"{step}.json"
        temporary_array = destination / f".{step}.tmp-{os.getpid()}.npz"
        temporary_metadata = destination / f".{step}.tmp-{os.getpid()}.json"
        np.savez_compressed(temporary_array, tables=self.value_function.tables)
        temporary_metadata.write_text(
            json.dumps(
                self.checkpoint_metadata(config_hash, environment_snapshot),
                sort_keys=True,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary_array.replace(array_path)
        temporary_metadata.replace(metadata_path)
        return array_path, metadata_path

    def restore_checkpoint(
        self, directory: str | Path, step: int, *, config_hash: str = ""
    ) -> EngineSnapshot:
        destination = Path(directory)
        array_path = destination / f"{step}.npz"
        metadata_path = destination / f"{step}.json"
        if not array_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(f"checkpoint pair not found for step {step}")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            raise ValueError("checkpoint metadata must be an object")
        required_fields = {
            "schema_version",
            "agent_type",
            "config_hash",
            "learner_config",
            "array_shape",
            "array_dtype",
            "counters",
            "state_hash",
            "environment",
            "checkpoint_hash",
        }
        if set(metadata) != required_fields:
            raise ValueError("checkpoint metadata must contain the complete v2 field set")
        if metadata.get("schema_version") != "checkpoint-meta-v2":
            raise ValueError("unsupported checkpoint schema")
        if metadata.get("config_hash", "") != config_hash:
            raise ValueError("checkpoint config hash mismatch")
        if metadata.get("agent_type") != self.agent_type:
            raise ValueError("checkpoint agent type mismatch")
        learner_config = metadata.get("learner_config")
        if not isinstance(learner_config, dict) or canonical_json(learner_config) != canonical_json(
            self.learner_config()
        ):
            raise ValueError("checkpoint learner config mismatch")
        array_shape = metadata.get("array_shape")
        if not isinstance(array_shape, list) or any(
            type(dimension) is not int for dimension in array_shape
        ):
            raise ValueError("checkpoint table shape must contain integers")
        if array_shape != list(self.value_function.tables.shape):
            raise ValueError("checkpoint table shape mismatch")
        array_dtype = metadata.get("array_dtype")
        if not isinstance(array_dtype, str) or array_dtype != str(self.value_function.tables.dtype):
            raise ValueError("checkpoint table dtype mismatch")
        candidate_counters = LearningCounters.from_json(metadata["counters"])
        candidate_environment = EngineSnapshot.from_json(metadata["environment"])
        if candidate_environment.schema_version != OracleEnv.SNAPSHOT_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint environment schema")
        if candidate_environment.rng.schema_version != RNG_SCHEMA_VERSION:
            raise ValueError("unsupported checkpoint RNG schema")
        environment_validator = OracleEnv(root_seed=candidate_environment.rng.seed)
        environment_validator.restore(candidate_environment)
        with np.load(array_path, allow_pickle=False) as archive:
            if set(archive.files) != {"tables"}:
                raise ValueError("checkpoint contains unexpected arrays")
            tables = archive["tables"].copy()
            if (
                tables.shape != self.value_function.tables.shape
                or tables.dtype != self.value_function.tables.dtype
            ):
                raise ValueError("checkpoint array shape or dtype mismatch")
        if metadata.get("state_hash") != self._state_hash_for(tables, candidate_counters):
            raise ValueError("checkpoint state hash mismatch")
        hash_payload = dict(metadata)
        checkpoint_hash = hash_payload.pop("checkpoint_hash")
        if not isinstance(checkpoint_hash, str):
            raise ValueError("checkpoint content hash must be a string")
        if checkpoint_hash != self._checkpoint_hash(hash_payload, tables):
            raise ValueError("checkpoint content hash mismatch")

        self.value_function.tables[...] = tables
        self.value_function.counters = candidate_counters
        return candidate_environment


@dataclass(slots=True)
class TD1PAgent:
    learner: TDLearner
    agent_type: str = "discovery"
    last_learning_seconds: float = field(default=0.0, init=False)

    def act(self, observation: Observation, mode: EvaluationMode = EvaluationMode.TRAIN) -> Action:
        del mode
        return self.learner.choose_action(observation)

    def observe(self, transition: StepResult, next_observation: Observation) -> None:
        started = time.perf_counter()
        self.learner.observe(transition, next_observation)
        self.last_learning_seconds = time.perf_counter() - started

    def knowledge_manifest(self) -> KnowledgeManifest:
        self.learner.agent_type = self.agent_type
        return self.learner.knowledge_manifest()

    @property
    def counters(self) -> LearningCounters:
        return self.learner.counters

    def save_checkpoint(
        self,
        directory: str | Path,
        step: int,
        *,
        config_hash: str,
        environment_snapshot: EngineSnapshot,
    ) -> tuple[Path, Path]:
        return self.learner.save_checkpoint(
            directory,
            step,
            config_hash=config_hash,
            environment_snapshot=environment_snapshot,
        )

    def restore_checkpoint(
        self, directory: str | Path, step: int, *, config_hash: str = ""
    ) -> EngineSnapshot:
        return self.learner.restore_checkpoint(directory, step, config_hash=config_hash)
