"""Two-stage calibration of optimistic initialization under one wall budget.

The protocol is intentionally separate from ``discovery-pilot-v1``.  It reuses
the Discovery runner's durable episode/checkpoint primitives, while owning its
candidate matrix, seed separation, stage decisions, reducer, and verifier.
"""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import tomllib
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jsonschema  # type: ignore[import-untyped]

from strategy2048.engine.oracle import EngineSnapshot, StepResult
from strategy2048.experiments.artifacts import (
    ArtifactError,
    ArtifactStore,
    KnowledgeManifest,
    canonical_json,
    config_hash,
)
from strategy2048.experiments.discovery import (
    REPOSITORY_ROOT,
    DiscoveryArmConfig,
    DiscoveryLearnerConfig,
    DiscoveryPilotConfig,
    _build_agent,
    _checkpoint_clone,
    _DiscoveryInterrupted,
    _initial_snapshot,
    _InterruptController,
    _path_in_artifact,
    _persist_interrupt_boundaries,
    _read_json,
    _read_jsonl,
    _restore_metrics,
    _resume_episode_boundary,
    _RunState,
    _save_checkpoint,
    _save_resume_checkpoint,
    _train_one_episode,
    _validated_checkpoint_directory,
    _write_run_manifest,
)
from strategy2048.experiments.evaluation import FrozenPolicyAgent, evaluate_frozen
from strategy2048.learning.td import DEFAULT_TUPLES
from strategy2048.rules.core import Action

CALIBRATION_SCHEMA_VERSION = "algorithm-calibration-v1"
CALIBRATION_SHARED_WALL_SECONDS = 600
CALIBRATION_FINALIZATION_RESERVE_SECONDS = 10.0
CALIBRATION_SCREEN_WALL_SECONDS = 270
CALIBRATION_SCREEN_EPISODE = 40
CALIBRATION_CONFIRM_EPISODE = 200
CALIBRATION_SCREEN_EVALUATION_EPISODES = 20
CALIBRATION_AUDIT_EVALUATION_EPISODES = 50
CALIBRATION_CANDIDATES = (
    ("td0_zero", 0.0),
    ("td0_oi_300", 300.0),
    ("td0_oi_1000", 1000.0),
    ("td0_oi_3000", 3000.0),
    ("td0_oi_10000", 10000.0),
)
CALIBRATION_GATES = (
    "oi-candidate-recommended",
    "zero-retained",
    "inconclusive",
    "performance-blocked",
    "contract-failed",
)
CALIBRATION_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/algorithm-calibration.v1.schema.json"

CalibrationGate = Literal[
    "oi-candidate-recommended",
    "zero-retained",
    "inconclusive",
    "performance-blocked",
    "contract-failed",
]
EvaluationSuite = Literal["selection", "audit"]
Clock = Callable[[], float]


class CalibrationConfigError(ValueError):
    """The calibration configuration is malformed or changes the approved matrix."""


@dataclass(frozen=True, slots=True)
class CalibrationCandidateConfig:
    id: str
    initialization: Literal["zero", "optimistic"]
    optimistic_total_value: float

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "initialization": self.initialization,
            "optimistic_total_value": self.optimistic_total_value,
        }


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    experiment_id: str
    output_root: str
    training_seeds: tuple[str, str]
    selection_evaluation_root_seed: str
    audit_evaluation_root_seed: str
    exploration_sample_stride: int
    incumbent_candidate_id: str
    parent_calibration_id: str
    candidate_generation_rule: str
    tuning_context_fingerprint: str
    learner: DiscoveryLearnerConfig
    candidates: tuple[
        CalibrationCandidateConfig,
        CalibrationCandidateConfig,
        CalibrationCandidateConfig,
        CalibrationCandidateConfig,
        CalibrationCandidateConfig,
    ]
    max_steps_per_episode: int | None = None
    schema_version: str = CALIBRATION_SCHEMA_VERSION
    shared_wall_seconds: int = CALIBRATION_SHARED_WALL_SECONDS
    finalization_reserve_seconds: float = CALIBRATION_FINALIZATION_RESERVE_SECONDS
    screen_wall_seconds: int = CALIBRATION_SCREEN_WALL_SECONDS
    round_robin_training_chunk: int = 10
    screen_target_episode: int = CALIBRATION_SCREEN_EPISODE
    confirm_target_episode: int = CALIBRATION_CONFIRM_EPISODE
    screen_evaluation_episodes: int = CALIBRATION_SCREEN_EVALUATION_EPISODES
    audit_evaluation_episodes: int = CALIBRATION_AUDIT_EVALUATION_EPISODES
    exploration_overhead_limit: float = 0.02

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "output_root": self.output_root,
            "shared_wall_seconds": self.shared_wall_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "screen_wall_seconds": self.screen_wall_seconds,
            "round_robin_training_chunk": self.round_robin_training_chunk,
            "training_seeds": list(self.training_seeds),
            "selection_evaluation_root_seed": self.selection_evaluation_root_seed,
            "audit_evaluation_root_seed": self.audit_evaluation_root_seed,
            "screen_target_episode": self.screen_target_episode,
            "confirm_target_episode": self.confirm_target_episode,
            "screen_evaluation_episodes": self.screen_evaluation_episodes,
            "audit_evaluation_episodes": self.audit_evaluation_episodes,
            "exploration_sample_stride": self.exploration_sample_stride,
            "exploration_overhead_limit": self.exploration_overhead_limit,
            "incumbent_candidate_id": self.incumbent_candidate_id,
            "parent_calibration_id": self.parent_calibration_id,
            "candidate_generation_rule": self.candidate_generation_rule,
            "tuning_context_fingerprint": self.tuning_context_fingerprint,
            "max_steps_per_episode": self.max_steps_per_episode,
            "learner": self.learner.to_json(),
            "candidates": [candidate.to_json() for candidate in self.candidates],
        }


def _load_schema() -> dict[str, Any]:
    value = json.loads(CALIBRATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CalibrationConfigError("calibration schema must be an object")
    return value


def _schema_error(error: jsonschema.ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    prefix = f" at {location}" if location else ""
    return f"calibration config schema validation failed{prefix}: {error.message}"


def compute_tuning_context_fingerprint(value: Mapping[str, Any]) -> str:
    """Hash every field that changes what an OI winner means."""

    learner_value = value.get("learner")
    learner = dict(learner_value) if isinstance(learner_value, Mapping) else {}
    tuples_value = learner.get("tuples")
    learner["tuples"] = (
        [list(coordinates) for coordinates in DEFAULT_TUPLES]
        if tuples_value is None
        else [
            [int(cast(int, index)) for index in coordinates]
            for coordinates in cast(Sequence[Sequence[object]], tuples_value)
        ]
    )
    candidate_values = value.get("candidates")
    candidates_by_id = {
        str(item.get("id")): dict(item)
        for item in cast(Sequence[Mapping[str, Any]], candidate_values or ())
        if isinstance(item, Mapping)
    }
    candidates = [
        candidates_by_id[candidate_id]
        for candidate_id, _ in CALIBRATION_CANDIDATES
        if candidate_id in candidates_by_id
    ]
    payload = {
        "learner": learner,
        "max_steps_per_episode": value.get("max_steps_per_episode"),
        "screen_target_episode": value.get("screen_target_episode"),
        "confirm_target_episode": value.get("confirm_target_episode"),
        "screen_evaluation_episodes": value.get("screen_evaluation_episodes"),
        "audit_evaluation_episodes": value.get("audit_evaluation_episodes"),
        "training_seeds": value.get("training_seeds"),
        "selection_evaluation_root_seed": value.get("selection_evaluation_root_seed"),
        "audit_evaluation_root_seed": value.get("audit_evaluation_root_seed"),
        "candidate_generation_rule": value.get("candidate_generation_rule"),
        "candidates": candidates,
    }
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def resolve_calibration_config(value: Mapping[str, Any]) -> CalibrationConfig:
    """Validate and canonicalize the fixed v1 calibration protocol."""

    raw = dict(value)
    if "optimistic_value" in raw or any(
        isinstance(item, dict) and "optimistic_value" in item
        for item in cast(Sequence[object], raw.get("candidates", ()))
    ):
        raise CalibrationConfigError("optimistic_value is ambiguous; use optimistic_total_value")
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(raw)
    except jsonschema.ValidationError as error:
        raise CalibrationConfigError(_schema_error(error)) from error
    expected_fingerprint = compute_tuning_context_fingerprint(raw)
    if raw["tuning_context_fingerprint"] != expected_fingerprint:
        raise CalibrationConfigError(
            "tuning_context_fingerprint does not match the resolved tuning context"
        )

    training_values = cast(list[str], raw["training_seeds"])
    training_seeds = (training_values[0], training_values[1])
    selection_seed = cast(str, raw["selection_evaluation_root_seed"])
    audit_seed = cast(str, raw["audit_evaluation_root_seed"])
    seed_names = {*training_seeds, selection_seed, audit_seed}
    if len(seed_names) != 4:
        raise CalibrationConfigError(
            "training, selection, and audit seed roots must be pairwise distinct"
        )

    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    for item in cast(list[Mapping[str, Any]], raw["candidates"]):
        candidate_id = cast(str, item["id"])
        if candidate_id in candidates_by_id:
            raise CalibrationConfigError(f"duplicate calibration candidate id: {candidate_id}")
        candidates_by_id[candidate_id] = item
    expected = dict(CALIBRATION_CANDIDATES)
    if set(candidates_by_id) != set(expected):
        raise CalibrationConfigError("v1 requires exactly the approved five candidates")

    canonical_candidates: list[CalibrationCandidateConfig] = []
    for candidate_id, expected_total in CALIBRATION_CANDIDATES:
        item = candidates_by_id[candidate_id]
        total = float(item["optimistic_total_value"])
        initialization = cast(str, item["initialization"])
        expected_initialization = "zero" if expected_total == 0.0 else "optimistic"
        if initialization != expected_initialization:
            raise CalibrationConfigError(
                f"{candidate_id} initialization must be {expected_initialization}"
            )
        if not math.isfinite(total) or total != expected_total:
            raise CalibrationConfigError(
                f"{candidate_id} optimistic_total_value must be {expected_total:g}"
            )
        canonical_candidates.append(
            CalibrationCandidateConfig(
                id=candidate_id,
                initialization=cast(Literal["zero", "optimistic"], initialization),
                optimistic_total_value=total,
            )
        )

    learner_value = cast(Mapping[str, Any], raw["learner"])
    tuples_value = learner_value.get("tuples")
    tuples = (
        DEFAULT_TUPLES
        if tuples_value is None
        else tuple(
            tuple(int(index) for index in coordinates)
            for coordinates in cast(list[list[int]], tuples_value)
        )
    )
    alpha = float(learner_value["alpha"])
    gamma = float(learner_value["gamma"])
    if not math.isfinite(alpha) or not math.isfinite(gamma):
        raise CalibrationConfigError("learner alpha and gamma must be finite")

    config = CalibrationConfig(
        experiment_id=cast(str, raw["experiment_id"]),
        output_root=cast(str, raw["output_root"]),
        training_seeds=training_seeds,
        selection_evaluation_root_seed=selection_seed,
        audit_evaluation_root_seed=audit_seed,
        exploration_sample_stride=cast(int, raw["exploration_sample_stride"]),
        incumbent_candidate_id=cast(str, raw["incumbent_candidate_id"]),
        parent_calibration_id=cast(str, raw["parent_calibration_id"]),
        candidate_generation_rule=cast(str, raw["candidate_generation_rule"]),
        tuning_context_fingerprint=cast(str, raw["tuning_context_fingerprint"]),
        max_steps_per_episode=cast(int | None, raw.get("max_steps_per_episode")),
        learner=DiscoveryLearnerConfig(
            alpha=alpha,
            gamma=gamma,
            symmetry=cast(bool, learner_value["symmetry"]),
            value_cardinality=cast(int, learner_value["value_cardinality"]),
            tuples=tuples,
        ),
        candidates=cast(
            tuple[
                CalibrationCandidateConfig,
                CalibrationCandidateConfig,
                CalibrationCandidateConfig,
                CalibrationCandidateConfig,
                CalibrationCandidateConfig,
            ],
            tuple(canonical_candidates),
        ),
    )
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(config.to_json())
    except jsonschema.ValidationError as error:
        raise CalibrationConfigError(_schema_error(error)) from error
    return config


def load_calibration_config(path: str | Path) -> CalibrationConfig:
    with Path(path).open("rb") as handle:
        return resolve_calibration_config(tomllib.load(handle))


def _execution_config(
    config: CalibrationConfig,
    *,
    evaluation_root_seed: str,
    evaluation_episodes: int,
) -> DiscoveryPilotConfig:
    zero = cast(DiscoveryArmConfig, config.candidates[0])
    optimistic = cast(DiscoveryArmConfig, config.candidates[1])
    return DiscoveryPilotConfig(
        experiment_id=config.experiment_id,
        output_root=config.output_root,
        round_robin_training_chunk=config.round_robin_training_chunk,
        training_seeds=config.training_seeds,
        evaluation_root_seed=evaluation_root_seed,
        evaluation_episodes_per_checkpoint=evaluation_episodes,
        diagnostic_score_milestone=1,
        diagnostic_tile_milestone=256,
        learner=config.learner,
        arms=(zero, optimistic),
        max_steps_per_episode=config.max_steps_per_episode,
        shared_wall_seconds=config.shared_wall_seconds,
        finalization_reserve_seconds=config.finalization_reserve_seconds,
        checkpoint_episodes=(0, config.screen_target_episode, config.confirm_target_episode),
        max_training_episodes_per_run=config.confirm_target_episode,
    )


def _append_progress(store: ArtifactStore, event: str, **fields: object) -> None:
    store.append_jsonl(
        "calibration-progress.jsonl",
        {"schema_version": "algorithm-calibration-progress-v1", "event": event, **fields},
    )


def _knowledge_manifest(config: CalibrationConfig) -> KnowledgeManifest:
    return KnowledgeManifest(
        experiment_kind="discovery",
        initialization={
            "source": "optimistic",
            "comparison_candidates": [candidate.id for candidate in config.candidates],
            "optimistic_total_values": [
                candidate.optimistic_total_value for candidate in config.candidates
            ],
        },
    )


def _candidate(config: CalibrationConfig, candidate_id: str) -> CalibrationCandidateConfig:
    for item in config.candidates:
        if item.id == candidate_id:
            return item
    raise ArtifactError(f"unknown calibration candidate: {candidate_id}")


def _new_states(config: CalibrationConfig) -> list[_RunState]:
    execution = _execution_config(
        config,
        evaluation_root_seed=config.selection_evaluation_root_seed,
        evaluation_episodes=config.screen_evaluation_episodes,
    )
    states: list[_RunState] = []
    for training_seed in config.training_seeds:
        for candidate in config.candidates:
            arm = cast(DiscoveryArmConfig, candidate)
            state = _RunState(
                arm=arm,
                training_seed=training_seed,
                agent=_build_agent(execution, arm),
                relative_root=f"runs/{candidate.id}/{training_seed}",
            )
            state.last_snapshot = _initial_snapshot(execution, training_seed)
            states.append(state)
    return states


def _milestone_records(root: Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    path = root / "checkpoints.jsonl"
    if not path.is_file():
        return {}
    records: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in _read_jsonl(path):
        if record.get("kind") != "milestone":
            continue
        candidate_id = record.get("arm_id")
        training_seed = record.get("training_seed")
        episode = record.get("checkpoint_episode")
        if (
            isinstance(candidate_id, str)
            and isinstance(training_seed, str)
            and type(episode) is int
        ):
            records[(candidate_id, training_seed, episode)] = record
    return records


def encode_afterstate_u64(board: Sequence[int]) -> int:
    """Pack a 4x4 exponent board without collisions for exponents 0..15."""

    if len(board) != 16:
        raise ArtifactError("afterstate encoding requires exactly 16 cells")
    encoded = 0
    for index, raw in enumerate(board):
        if type(raw) is not int or raw < 0 or raw > 15:
            raise ArtifactError("afterstate-u64-v1 cannot encode exponents outside 0..15")
        encoded |= raw << (index * 4)
    return encoded


@dataclass(slots=True)
class ExplorationTracker:
    sample_stride: int
    observed_steps: int = 0
    sampled_observations: int = 0
    instrumentation_wall_seconds: float = 0.0
    sampled_afterstates: set[int] | None = None
    action_counts: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.sample_stride <= 0:
            raise ValueError("exploration sample stride must be positive")
        if self.sampled_afterstates is None:
            self.sampled_afterstates = set()
        if self.action_counts is None:
            self.action_counts = Counter()

    def observe(self, result: StepResult) -> None:
        started = time.perf_counter()
        self.observed_steps += 1
        assert self.action_counts is not None
        self.action_counts[result.action.name_lower] += 1
        if self.observed_steps % self.sample_stride == 0:
            self.sampled_observations += 1
            assert self.sampled_afterstates is not None
            self.sampled_afterstates.add(encode_afterstate_u64(result.afterstate))
        self.instrumentation_wall_seconds += time.perf_counter() - started

    def to_json(self) -> dict[str, object]:
        assert self.sampled_afterstates is not None
        assert self.action_counts is not None
        encoded = sorted(self.sampled_afterstates)
        payload = canonical_json(encoded)
        return {
            "schema_version": "exploration-coverage-v1",
            "encoding": "afterstate-u64-v1",
            "sample_stride": self.sample_stride,
            "observed_steps": self.observed_steps,
            "sampled_observations": self.sampled_observations,
            "distinct_sampled_afterstates": len(encoded),
            "first_visit_ratio": (
                len(encoded) / self.sampled_observations if self.sampled_observations else 0.0
            ),
            "action_distribution": {
                action.name_lower: self.action_counts.get(action.name_lower, 0) for action in Action
            },
            "instrumentation_wall_seconds": self.instrumentation_wall_seconds,
            "coverage_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
            "sampled_afterstates": encoded,
        }

    @classmethod
    def from_json(cls, value: Mapping[str, Any]) -> ExplorationTracker:
        if value.get("schema_version") != "exploration-coverage-v1":
            raise ArtifactError("unsupported exploration coverage schema")
        encoded = value.get("sampled_afterstates")
        action_distribution = value.get("action_distribution")
        if not isinstance(encoded, list) or any(type(item) is not int for item in encoded):
            raise ArtifactError("exploration coverage states are malformed")
        if encoded != sorted(set(encoded)):
            raise ArtifactError("exploration coverage states must be sorted and unique")
        if not isinstance(action_distribution, Mapping):
            raise ArtifactError("exploration action distribution is malformed")
        expected_hash = hashlib.sha256(canonical_json(encoded).encode("utf-8")).hexdigest()
        if value.get("coverage_sha256") != expected_hash:
            raise ArtifactError("exploration coverage hash mismatch")
        tracker = cls(
            sample_stride=cast(int, value["sample_stride"]),
            observed_steps=cast(int, value["observed_steps"]),
            sampled_observations=cast(int, value["sampled_observations"]),
            instrumentation_wall_seconds=float(value["instrumentation_wall_seconds"]),
            sampled_afterstates=set(cast(list[int], encoded)),
            action_counts=Counter(
                {
                    str(name): count
                    for name, count in action_distribution.items()
                    if type(count) is int
                }
            ),
        )
        if tracker.to_json()["coverage_sha256"] != expected_hash:
            raise ArtifactError("exploration coverage reconstruction mismatch")
        return tracker


def _coverage_path(state: _RunState) -> str:
    return f"{state.relative_root}/coverage/latest.json"


def _write_coverage(
    store: ArtifactStore,
    state: _RunState,
    tracker: ExplorationTracker,
    *,
    checkpoint_episode: int,
) -> None:
    value = tracker.to_json()
    value.update(
        {
            "candidate_id": state.arm.id,
            "training_seed": state.training_seed,
            "checkpoint_episode": checkpoint_episode,
        }
    )
    store.write_json(_coverage_path(state), value)
    store.append_jsonl(f"{state.relative_root}/coverage/checkpoints.jsonl", value)


def _evaluation_path(state: _RunState, suite: EvaluationSuite, episode: int) -> str:
    return f"{state.relative_root}/evaluation/{suite}/{episode}/episodes.jsonl"


def _evaluation_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("runs/*/*/evaluation/*/*/episodes.jsonl")):
        _path_in_artifact(root, str(path.relative_to(root)))
        records.extend(_read_jsonl(path))
    return records


def _training_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("runs/*/*/training-episodes.jsonl")):
        _path_in_artifact(root, str(path.relative_to(root)))
        records.extend(_read_jsonl(path))
    return records


def _evaluation_keys(root: Path) -> set[tuple[str, str, str, int, int]]:
    keys: set[tuple[str, str, str, int, int]] = set()
    for record in _evaluation_records(root):
        candidate_id = record.get("candidate_id")
        training_seed = record.get("training_seed")
        suite = record.get("suite")
        checkpoint = record.get("checkpoint_episode")
        episode_id = record.get("evaluation_episode_id")
        if (
            isinstance(candidate_id, str)
            and isinstance(training_seed, str)
            and isinstance(suite, str)
            and type(checkpoint) is int
            and type(episode_id) is int
        ):
            keys.add((candidate_id, training_seed, suite, checkpoint, episode_id))
    return keys


def _evaluate_round_robin(
    store: ArtifactStore,
    config: CalibrationConfig,
    states: Sequence[_RunState],
    checkpoints: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    suite: EvaluationSuite,
    checkpoint_episode: int,
    episodes: int,
    root_seed: str,
    clock: Clock,
    deadline: float,
    process_clock: Clock,
    interrupts: _InterruptController,
    completed_keys: set[tuple[str, str, str, int, int]],
    phase_hook: Callable[[str], None] | None,
) -> bool:
    execution = _execution_config(
        config,
        evaluation_root_seed=root_seed,
        evaluation_episodes=episodes,
    )
    contexts: list[tuple[_RunState, FrozenPolicyAgent, str, str, Mapping[str, object]]] = []
    for state in states:
        if all(
            (state.arm.id, state.training_seed, suite, checkpoint_episode, episode_id)
            in completed_keys
            for episode_id in range(episodes)
        ):
            continue
        if interrupts.requested:
            _persist_interrupt_boundaries(store, states, phase="calibration_evaluation_clone")
            raise _DiscoveryInterrupted
        if clock() >= deadline:
            return False
        checkpoint = checkpoints[state.key]
        contexts.append(
            (
                state,
                _checkpoint_clone(store, execution, state, checkpoint),
                state.agent.learner.state_hash(),
                canonical_json(state.agent.counters.to_json()),
                checkpoint,
            )
        )

    for evaluation_episode_id in range(episodes):
        for state, clone, training_hash, training_counters, checkpoint in contexts:
            key = (
                state.arm.id,
                state.training_seed,
                suite,
                checkpoint_episode,
                evaluation_episode_id,
            )
            if key in completed_keys:
                continue
            if interrupts.requested:
                _persist_interrupt_boundaries(store, states, phase="calibration_evaluation")
                raise _DiscoveryInterrupted
            if clock() >= deadline:
                return False
            if phase_hook is not None:
                phase_hook(f"{suite}_evaluation_episode")
            cpu_started = process_clock()
            result = evaluate_frozen(
                clone,
                episodes=1,
                root_seed=root_seed,
                purpose=f"calibration-{suite}-eval",
                environment_id=f"{config.experiment_id}-{suite}-evaluation",
                max_steps=config.max_steps_per_episode,
                episode_ids=(evaluation_episode_id,),
                clock=clock,
                deadline=deadline,
            )
            process_seconds = max(0.0, process_clock() - cpu_started)
            if result["completed_episodes"] != 1:
                store.append_jsonl(
                    f"{state.relative_root}/evaluation/{suite}/{checkpoint_episode}/partials.jsonl",
                    {
                        "schema_version": "algorithm-calibration-evaluation-partial-v1",
                        "candidate_id": state.arm.id,
                        "training_seed": state.training_seed,
                        "suite": suite,
                        "checkpoint_episode": checkpoint_episode,
                        "evaluation_episode_id": evaluation_episode_id,
                        "evaluation_root_seed": root_seed,
                        "frozen_result": result,
                    },
                )
                return False
            episode = cast(list[dict[str, object]], result["episodes"])[0]
            training_hash_after = state.agent.learner.state_hash()
            training_counters_after = canonical_json(state.agent.counters.to_json())
            record: dict[str, object] = {
                "schema_version": "algorithm-calibration-evaluation-episode-v1",
                "candidate_id": state.arm.id,
                "training_seed": state.training_seed,
                "suite": suite,
                "checkpoint_episode": checkpoint_episode,
                "checkpoint_global_env_step": checkpoint["global_env_step"],
                "evaluation_episode_id": evaluation_episode_id,
                "evaluation_root_seed": root_seed,
                "purpose": f"calibration-{suite}-eval",
                "official_score": episode["official_score"],
                "max_tile": episode["max_tile"],
                "steps": episode["steps"],
                "wall_seconds": episode["wall_seconds"],
                "process_cpu_seconds": process_seconds,
                "terminated": episode["terminated"],
                "truncated": episode["truncated"],
                "metrics": result["metrics"],
                "evaluation_counters": result["evaluation_counters"],
                "frozen_state_unchanged": result["state_unchanged"],
                "clone_state_hash_before": result["state_hash_before"],
                "clone_state_hash_after": result["state_hash_after"],
                "clone_table_hash_before": result["table_hash_before"],
                "clone_table_hash_after": result["table_hash_after"],
                "clone_counters_before": result["counters_before"],
                "clone_counters_after": result["counters_after"],
                "training_state_hash_before": training_hash,
                "training_state_hash_after": training_hash_after,
                "training_counters_unchanged": training_counters == training_counters_after,
            }
            store.append_jsonl(_evaluation_path(state, suite, checkpoint_episode), record)
            completed_keys.add(key)
            _append_progress(
                store,
                "evaluation_episode_completed",
                candidate_id=state.arm.id,
                training_seed=state.training_seed,
                suite=suite,
                checkpoint_episode=checkpoint_episode,
                evaluation_episode_id=evaluation_episode_id,
            )
    return True


def _suite_scores(
    records: Sequence[Mapping[str, Any]],
    *,
    suite: EvaluationSuite,
    checkpoint_episode: int,
) -> dict[tuple[str, str], dict[int, float]]:
    result: dict[tuple[str, str], dict[int, float]] = {}
    for record in records:
        if record.get("suite") != suite or record.get("checkpoint_episode") != checkpoint_episode:
            continue
        candidate_id = record.get("candidate_id")
        training_seed = record.get("training_seed")
        evaluation_episode_id = record.get("evaluation_episode_id")
        score = record.get("official_score")
        if (
            isinstance(candidate_id, str)
            and isinstance(training_seed, str)
            and type(evaluation_episode_id) is int
            and type(score) is int
        ):
            key = (candidate_id, training_seed)
            episode_scores = result.setdefault(key, {})
            if evaluation_episode_id in episode_scores:
                raise ArtifactError(f"duplicate evaluation record: {key}/{evaluation_episode_id}")
            episode_scores[evaluation_episode_id] = float(score)
    return result


def _paired_relative_differences(
    config: CalibrationConfig,
    scores: Mapping[tuple[str, str], Mapping[int, float]],
    *,
    candidate_id: str,
    expected_episodes: int,
) -> dict[str, float]:
    differences: dict[str, float] = {}
    for training_seed in config.training_seeds:
        zero = scores.get(("td0_zero", training_seed), {})
        candidate = scores.get((candidate_id, training_seed), {})
        expected_ids = set(range(expected_episodes))
        if set(zero) != expected_ids or set(candidate) != expected_ids:
            raise ArtifactError(f"incomplete paired evaluation for {candidate_id}/{training_seed}")
        zero_mean = statistics.fmean(zero.values())
        candidate_mean = statistics.fmean(candidate.values())
        differences[training_seed] = (
            (candidate_mean - zero_mean) / zero_mean
            if zero_mean > 0.0
            else (0.0 if candidate_mean == 0.0 else math.inf)
        )
    return differences


def derive_screen_decision(
    config: CalibrationConfig,
    evaluation_records: Sequence[Mapping[str, Any]],
    training_records: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Select at most one OI survivor from selection-suite raw records."""

    selection_inputs = [
        dict(record)
        for record in evaluation_records
        if record.get("suite") == "selection"
        and record.get("checkpoint_episode") == config.screen_target_episode
    ]
    scores = _suite_scores(
        selection_inputs,
        suite="selection",
        checkpoint_episode=config.screen_target_episode,
    )
    episode_zero_scores = _suite_scores(
        evaluation_records,
        suite="selection",
        checkpoint_episode=0,
    )
    training_inputs = [
        dict(record)
        for record in training_records
        if record.get("arm_id") in {candidate.id for candidate in config.candidates}
        and record.get("training_seed") in config.training_seeds
        and type(record.get("episode_id")) is int
        and cast(int, record["episode_id"]) < config.screen_target_episode
    ]
    candidates: list[dict[str, Any]] = []
    for candidate in config.candidates[1:]:
        differences = _paired_relative_differences(
            config,
            scores,
            candidate_id=candidate.id,
            expected_episodes=config.screen_evaluation_episodes,
        )
        values = list(differences.values())
        eliminated = all(value <= -0.10 for value in values)
        checkpoint_mean = statistics.fmean(
            score
            for training_seed in config.training_seeds
            for score in scores[(candidate.id, training_seed)].values()
        )
        initial_values = [
            score
            for training_seed in config.training_seeds
            for score in episode_zero_scores.get((candidate.id, training_seed), {}).values()
        ]
        initial_mean = statistics.fmean(initial_values) if initial_values else checkpoint_mean
        candidate_training = [
            record for record in training_inputs if record.get("arm_id") == candidate.id
        ]
        env_steps = sum(
            cast(int, record["steps"])
            for record in candidate_training
            if type(record.get("steps")) is int
        )
        wall_seconds = sum(
            float(cast(float | int, record["wall_seconds"]))
            for record in candidate_training
            if isinstance(record.get("wall_seconds"), (int, float))
            and not isinstance(record.get("wall_seconds"), bool)
        )
        score_gain = checkpoint_mean - initial_mean
        candidates.append(
            {
                "candidate_id": candidate.id,
                "optimistic_total_value": candidate.optimistic_total_value,
                "relative_difference_by_training_seed": differences,
                "worst_seed_relative_difference": min(values),
                "paired_mean_relative_difference": statistics.fmean(values),
                "selection_score_gain_from_episode_zero": score_gain,
                "selection_score_gain_per_env_step": (score_gain / env_steps if env_steps else 0.0),
                "selection_score_gain_per_wall_second": (
                    score_gain / wall_seconds if wall_seconds else 0.0
                ),
                "eliminated": eliminated,
                "elimination_reason": (
                    "both-training-seeds-at-least-10-percent-below-zero" if eliminated else None
                ),
            }
        )
    eligible = [item for item in candidates if item["eliminated"] is False]
    survivor = None
    if eligible:
        survivor = sorted(
            eligible,
            key=lambda item: (
                -cast(float, item["worst_seed_relative_difference"]),
                -cast(float, item["paired_mean_relative_difference"]),
                -cast(float, item["selection_score_gain_per_env_step"]),
                -cast(float, item["selection_score_gain_per_wall_second"]),
                cast(str, item["candidate_id"]),
            ),
        )[0]["candidate_id"]
    return {
        "schema_version": "algorithm-calibration-screen-decision-v1",
        "suite": "selection",
        "checkpoint_episode": config.screen_target_episode,
        "candidate_results": candidates,
        "survivor_candidate_id": survivor,
        "all_oi_eliminated": survivor is None,
        "selection_rule": "worst-seed-then-paired-mean-v1",
        "selection_input_record_count": len(selection_inputs),
        "selection_input_sha256": hashlib.sha256(
            canonical_json(selection_inputs).encode("utf-8")
        ).hexdigest(),
        "selection_training_input_record_count": len(training_inputs),
        "selection_training_input_sha256": hashlib.sha256(
            canonical_json(training_inputs).encode("utf-8")
        ).hexdigest(),
    }


def _final_gate(
    config: CalibrationConfig,
    evaluation_records: Sequence[Mapping[str, Any]],
    survivor_candidate_id: str,
) -> tuple[CalibrationGate, dict[str, float]]:
    scores = _suite_scores(
        evaluation_records,
        suite="audit",
        checkpoint_episode=config.confirm_target_episode,
    )
    differences = _paired_relative_differences(
        config,
        scores,
        candidate_id=survivor_candidate_id,
        expected_episodes=config.audit_evaluation_episodes,
    )
    values = list(differences.values())
    if statistics.fmean(values) >= 0.05 and all(value >= 0.0 for value in values):
        return "oi-candidate-recommended", differences
    if all(value < 0.0 for value in values):
        return "zero-retained", differences
    return "inconclusive", differences


def _open_existing_store(root: Path, config: CalibrationConfig) -> ArtifactStore:
    store = ArtifactStore.__new__(ArtifactStore)
    store.root = root
    store.resolved_config = cast(dict[str, Any], config.to_json())
    store.config_hash = config_hash(store.resolved_config)
    store.repo_root = REPOSITORY_ROOT
    store.started_at = time.time()
    return store


def _latest_stop(root: Path) -> dict[str, Any] | None:
    path = root / "calibration-progress.jsonl"
    if not path.is_file():
        return None
    for record in reversed(_read_jsonl(path)):
        if record.get("event") == "calibration_stopped":
            return record
    return None


def _restore_states(
    store: ArtifactStore,
    config: CalibrationConfig,
) -> tuple[list[_RunState], dict[tuple[str, str], ExplorationTracker]]:
    execution = _execution_config(
        config,
        evaluation_root_seed=config.selection_evaluation_root_seed,
        evaluation_episodes=config.screen_evaluation_episodes,
    )
    states: list[_RunState] = []
    trackers: dict[tuple[str, str], ExplorationTracker] = {}
    for training_seed in config.training_seeds:
        for candidate in config.candidates:
            arm = cast(DiscoveryArmConfig, candidate)
            state = _RunState(
                arm=arm,
                training_seed=training_seed,
                agent=_build_agent(execution, arm),
                relative_root=f"runs/{candidate.id}/{training_seed}",
            )
            pointer_path = store.root / f"{state.relative_root}/resume-checkpoint.json"
            if not pointer_path.is_file():
                state.last_snapshot = _initial_snapshot(execution, training_seed)
            else:
                pointer = _read_json(pointer_path)
                step = pointer.get("checkpoint_step")
                if type(step) is not int:
                    raise ArtifactError("calibration resume checkpoint step is invalid")
                directory = _validated_checkpoint_directory(store, pointer, step=step)
                snapshot = state.agent.restore_checkpoint(
                    directory,
                    step,
                    config_hash=store.config_hash,
                )
                if pointer.get("arm_id") != candidate.id:
                    raise ArtifactError("calibration resume checkpoint candidate mismatch")
                if pointer.get("training_seed") != training_seed:
                    raise ArtifactError("calibration resume checkpoint seed mismatch")
                if pointer.get("learner_state_hash") != state.agent.learner.state_hash():
                    raise ArtifactError("calibration resume learner state hash mismatch")
                if pointer.get("table_hash") != state.agent.learner.table_hash():
                    raise ArtifactError("calibration resume table hash mismatch")
                if canonical_json(pointer.get("counters")) != canonical_json(
                    state.agent.counters.to_json()
                ):
                    raise ArtifactError("calibration resume learner counters mismatch")
                if canonical_json(pointer.get("environment")) != canonical_json(snapshot.to_json()):
                    raise ArtifactError("calibration resume environment mismatch")
                lineage = snapshot.rng.lineage
                if (
                    lineage.get("root_seed") != training_seed
                    or lineage.get("purpose") != "train-env"
                    or lineage.get("environment_id") != f"{config.experiment_id}-training"
                ):
                    raise ArtifactError("calibration resume RNG lineage mismatch")
                completed = pointer.get("completed_training_episodes")
                global_steps = pointer.get("global_env_step")
                if type(completed) is not int or type(global_steps) is not int:
                    raise ArtifactError("calibration resume progress is invalid")
                state.completed_episodes = completed
                state.global_env_steps = global_steps
                state.last_snapshot = snapshot
                state.metrics = _restore_metrics(pointer)
                active_wall = pointer.get("active_wall_seconds", 0.0)
                process_cpu = pointer.get("process_cpu_seconds", 0.0)
                if any(
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or float(item) < 0.0
                    for item in (active_wall, process_cpu)
                ):
                    raise ArtifactError("calibration resume timing metrics are invalid")
                state.active_wall_seconds = float(active_wall)
                state.process_cpu_seconds = float(process_cpu)
                resume_episode = _resume_episode_boundary(pointer, snapshot)
                if resume_episode is not None:
                    state.resume_snapshot = snapshot
                    (
                        state.resume_episode_env_steps_before,
                        state.resume_episode_counters_before,
                        state.resume_episode_wall_seconds,
                        state.resume_episode_process_cpu_seconds,
                    ) = resume_episode
            coverage_path = store.root / _coverage_path(state)
            tracker = (
                ExplorationTracker.from_json(_read_json(coverage_path))
                if coverage_path.is_file()
                else ExplorationTracker(config.exploration_sample_stride)
            )
            if tracker.observed_steps != state.global_env_steps:
                raise ArtifactError("exploration coverage step count does not match run state")
            states.append(state)
            trackers[state.key] = tracker
    return states, trackers


def _train_to_target(
    store: ArtifactStore,
    config: CalibrationConfig,
    states: Sequence[_RunState],
    trackers: Mapping[tuple[str, str], ExplorationTracker],
    *,
    target_episode: int,
    clock: Clock,
    deadline: float,
    process_clock: Clock,
    interrupts: _InterruptController,
    phase_hook: Callable[[str], None] | None,
) -> bool:
    execution = _execution_config(
        config,
        evaluation_root_seed=config.selection_evaluation_root_seed,
        evaluation_episodes=config.screen_evaluation_episodes,
    )
    while any(
        state.completed_episodes < target_episode or state.resume_snapshot is not None
        for state in states
    ):
        for state in states:
            target = min(
                target_episode,
                ((state.completed_episodes // config.round_robin_training_chunk) + 1)
                * config.round_robin_training_chunk,
            )
            tracker = trackers[state.key]
            while state.completed_episodes < target or state.resume_snapshot is not None:
                try:
                    completed = _train_one_episode(
                        store,
                        execution,
                        state,
                        clock=clock,
                        deadline=deadline,
                        process_clock=process_clock,
                        interrupts=interrupts,
                        phase_hook=phase_hook,
                        step_observer=tracker.observe,
                    )
                except _DiscoveryInterrupted:
                    _write_coverage(
                        store,
                        state,
                        tracker,
                        checkpoint_episode=state.completed_episodes,
                    )
                    raise
                if not completed:
                    _write_coverage(
                        store,
                        state,
                        tracker,
                        checkpoint_episode=state.completed_episodes,
                    )
                    return False
            _save_resume_checkpoint(store, state)
            _write_coverage(
                store,
                state,
                tracker,
                checkpoint_episode=state.completed_episodes,
            )
            _append_progress(
                store,
                "training_chunk_completed",
                candidate_id=state.arm.id,
                training_seed=state.training_seed,
                completed_training_episodes=state.completed_episodes,
                target_episode=target_episode,
            )
            if interrupts.requested:
                _persist_interrupt_boundaries(store, states, phase="calibration_training_chunk")
                raise _DiscoveryInterrupted
            if clock() >= deadline:
                return False
    return True


def _ensure_milestones(
    store: ArtifactStore,
    states: Sequence[_RunState],
    *,
    checkpoint_episode: int,
    clock: Clock,
    deadline: float,
) -> dict[tuple[str, str], Mapping[str, object]] | None:
    existing = _milestone_records(store.root)
    result: dict[tuple[str, str], Mapping[str, object]] = {}
    for state in states:
        key = (state.arm.id, state.training_seed, checkpoint_episode)
        if key in existing:
            result[state.key] = existing[key]
            continue
        if state.completed_episodes != checkpoint_episode or state.resume_snapshot is not None:
            raise ArtifactError("calibration milestone reached with incomplete run state")
        if clock() >= deadline:
            return None
        result[state.key] = _save_checkpoint(
            store,
            state,
            checkpoint_episode=checkpoint_episode,
            kind="milestone",
        )
    return result


def _stage_decisions(root: Path) -> list[dict[str, Any]]:
    path = root / "stage-decisions.jsonl"
    return _read_jsonl(path) if path.is_file() else []


def _write_or_validate_screen_decision(
    store: ArtifactStore,
    decision: Mapping[str, Any],
) -> None:
    records = _stage_decisions(store.root)
    if len(records) > 1:
        raise ArtifactError("calibration artifact contains duplicate screen decisions")
    if records:
        if canonical_json(records[0]) != canonical_json(decision):
            raise ArtifactError("persisted screen decision does not match raw records")
        return
    store.append_jsonl("stage-decisions.jsonl", decision)


def _coverage_summary(root: Path, config: CalibrationConfig) -> dict[str, Any]:
    runs: dict[str, Any] = {}
    instrumentation = 0.0
    observed_training_wall = 0.0
    for training_seed in config.training_seeds:
        for candidate in config.candidates:
            run_key = f"{candidate.id}/{training_seed}"
            path = root / f"runs/{run_key}/coverage/latest.json"
            if not path.is_file():
                continue
            coverage = _read_json(path)
            tracker = ExplorationTracker.from_json(coverage)
            value = tracker.to_json()
            value.pop("sampled_afterstates")
            runs[run_key] = value
            instrumentation += tracker.instrumentation_wall_seconds
            training_path = root / f"runs/{run_key}/training-episodes.jsonl"
            if training_path.is_file():
                for record in _read_jsonl(training_path):
                    wall = record.get("wall_seconds", 0.0)
                    if isinstance(wall, (int, float)) and not isinstance(wall, bool):
                        observed_training_wall += float(wall)
    share = instrumentation / observed_training_wall if observed_training_wall else 0.0
    return {
        "encoding": "afterstate-u64-v1",
        "runs": runs,
        "instrumentation_wall_seconds": instrumentation,
        "observed_training_wall_seconds": observed_training_wall,
        "instrumentation_share": share,
        "exploration_metric_status": (
            "too-expensive" if share > config.exploration_overhead_limit else "usable"
        ),
        "selection_uses_exploration_metric": False,
    }


def _next_round_suggestion(
    config: CalibrationConfig,
    *,
    gate: CalibrationGate,
    survivor_candidate_id: str | None,
) -> dict[str, Any]:
    if gate == "oi-candidate-recommended" and survivor_candidate_id is not None:
        incumbent = _candidate(config, survivor_candidate_id).optimistic_total_value
        values = sorted({0.0, incumbent / 3.0, incumbent, incumbent * 3.0})
        recommendation = "confirm-with-more-training-seeds"
    elif gate == "zero-retained":
        incumbent = 0.0
        values = [0.0]
        recommendation = "retain-zero-and-calibrate-another-algorithm-axis"
    else:
        incumbent = (
            _candidate(config, survivor_candidate_id).optimistic_total_value
            if survivor_candidate_id is not None
            else 0.0
        )
        values = (
            sorted({0.0, incumbent / 3.0, incumbent, incumbent * 3.0}) if incumbent > 0.0 else [0.0]
        )
        recommendation = "collect-more-independent-evidence-before-promotion"
    return {
        "automatic_launch": False,
        "recommendation": recommendation,
        "incumbent_optimistic_total_value": incumbent,
        "candidate_generation_rule": "zero-plus-incumbent-log-neighbors-v1",
        "suggested_optimistic_total_values": values,
    }


def recompute_calibration_summary(
    artifact_directory: str | Path,
    *,
    contract_errors: Sequence[str] = (),
    stopped_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(artifact_directory)
    config = resolve_calibration_config(_read_json(root / "resolved-config.json"))
    evaluations = _evaluation_records(root)
    milestones = _milestone_records(root)
    stop = dict(stopped_record) if stopped_record is not None else (_latest_stop(root) or {})
    errors = list(contract_errors)
    if not errors:
        recorded_errors = stop.get("contract_errors", [])
        if isinstance(recorded_errors, list) and all(
            isinstance(item, str) for item in recorded_errors
        ):
            errors.extend(cast(list[str], recorded_errors))

    expected_screen_keys = {
        (candidate.id, seed, episode)
        for candidate in config.candidates
        for seed in config.training_seeds
        for episode in (0, config.screen_target_episode)
    }
    existing_milestones = set(milestones)
    screen_checkpoints_complete = expected_screen_keys <= existing_milestones
    screen_scores = _suite_scores(
        evaluations,
        suite="selection",
        checkpoint_episode=config.screen_target_episode,
    )
    expected_eval_ids = set(range(config.screen_evaluation_episodes))
    screen_evaluation_complete = all(
        set(screen_scores.get((candidate.id, seed), {})) == expected_eval_ids
        for candidate in config.candidates
        for seed in config.training_seeds
    )

    screen_decision: dict[str, Any] | None = None
    survivor: str | None = None
    if screen_checkpoints_complete and screen_evaluation_complete:
        screen_decision = derive_screen_decision(
            config,
            evaluations,
            _training_records(root),
        )
        survivor_value = screen_decision["survivor_candidate_id"]
        survivor = survivor_value if isinstance(survivor_value, str) else None

    gate: CalibrationGate
    audit_differences: dict[str, float] | None = None
    if errors:
        gate = "contract-failed"
    elif not screen_checkpoints_complete or not screen_evaluation_complete:
        gate = "performance-blocked"
    elif survivor is None:
        gate = "zero-retained"
    else:
        selected_ids = {"td0_zero", survivor}
        audit_scores = _suite_scores(
            evaluations,
            suite="audit",
            checkpoint_episode=config.confirm_target_episode,
        )
        expected_audit_ids = set(range(config.audit_evaluation_episodes))
        audit_complete = all(
            set(audit_scores.get((candidate_id, seed), {})) == expected_audit_ids
            for candidate_id in selected_ids
            for seed in config.training_seeds
        )
        confirm_checkpoints_complete = all(
            (candidate_id, seed, config.confirm_target_episode) in milestones
            for candidate_id in selected_ids
            for seed in config.training_seeds
        )
        if not audit_complete or not confirm_checkpoints_complete:
            gate = "inconclusive"
        else:
            gate, audit_differences = _final_gate(config, evaluations, survivor)

    coverage = _coverage_summary(root, config)
    return {
        "schema_version": "algorithm-calibration-summary-v1",
        "experiment_id": config.experiment_id,
        "config_hash": config_hash(config.to_json()),
        "gate": gate,
        "stop_reason": stop.get("stop_reason", "unknown"),
        "consumed_wall_seconds": stop.get("consumed_wall_seconds"),
        "process_cpu_seconds": stop.get("process_cpu_seconds"),
        "shared_wall_seconds": config.shared_wall_seconds,
        "screen_complete": screen_checkpoints_complete and screen_evaluation_complete,
        "screen_decision": screen_decision,
        "survivor_candidate_id": survivor,
        "audit_relative_difference_by_training_seed": audit_differences,
        "exploration": coverage,
        "lineage": {
            "parent_calibration_id": config.parent_calibration_id,
            "incumbent_candidate_id": config.incumbent_candidate_id,
            "candidate_generation_rule": config.candidate_generation_rule,
            "tuning_context_fingerprint": config.tuning_context_fingerprint,
        },
        "next_round_suggestion": _next_round_suggestion(
            config,
            gate=gate,
            survivor_candidate_id=survivor,
        ),
        "evidence_boundary": {
            "training_seed_count": len(config.training_seeds),
            "statistical_significance_claimed": False,
            "named_strategy_detector_run": False,
            "selection_suite_used_for_promotion": False,
            "audit_suite_used_for_final_gate": True,
        },
        "raw_record_counts": {
            "evaluation_episodes": len(evaluations),
            "checkpoints": len(_read_jsonl(root / "checkpoints.jsonl"))
            if (root / "checkpoints.jsonl").is_file()
            else 0,
            "screen_decisions": len(_stage_decisions(root)),
        },
        "contract_errors": errors,
    }


def run_algorithm_calibration(
    config: str | Path | Mapping[str, Any] | CalibrationConfig,
    *,
    artifact_directory: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume: bool = False,
    clock: Clock = time.monotonic,
    process_clock: Clock = time.process_time,
    phase_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or explicitly resume the fixed two-stage v1 calibration matrix."""

    if isinstance(config, (str, Path)):
        resolved = load_calibration_config(config)
    elif isinstance(config, CalibrationConfig):
        resolved = resolve_calibration_config(config.to_json())
    else:
        resolved = resolve_calibration_config(config)
    if resume and resume_from is None:
        if artifact_directory is None:
            raise ArtifactError("resume=True requires artifact_directory or resume_from")
        resume_from = artifact_directory
    if (
        resume_from is not None
        and artifact_directory is not None
        and Path(resume_from).resolve() != Path(artifact_directory).resolve()
    ):
        raise ArtifactError("resume_from and artifact_directory must identify the same root")
    root = (
        Path(resume_from)
        if resume_from is not None
        else (
            Path(artifact_directory)
            if artifact_directory is not None
            else Path(resolved.output_root) / resolved.experiment_id
        )
    )

    prior_consumed = 0.0
    prior_process = 0.0
    if resume_from is not None:
        preflight = verify_calibration_artifact(root)
        if preflight.get("valid") is not True:
            raise ArtifactError(
                "resume artifact failed read-only verification: "
                + "; ".join(str(item) for item in preflight.get("errors", []))
            )
        stored = resolve_calibration_config(_read_json(root / "resolved-config.json"))
        if canonical_json(stored.to_json()) != canonical_json(resolved.to_json()):
            raise ArtifactError("resume config does not match resolved-config.json")
        latest = _latest_stop(root)
        if latest is None or latest.get("stop_reason") not in {
            "interrupted",
            "budget_exhausted",
        }:
            raise ArtifactError("resume requires an interrupted or budget-exhausted artifact")
        consumed = latest.get("consumed_wall_seconds")
        process = latest.get("process_cpu_seconds", 0.0)
        if (
            isinstance(consumed, bool)
            or not isinstance(consumed, (int, float))
            or isinstance(process, bool)
            or not isinstance(process, (int, float))
        ):
            raise ArtifactError("resume budget accounting is malformed")
        prior_consumed = float(consumed)
        prior_process = float(process)
        if prior_consumed < 0.0 or prior_consumed > resolved.shared_wall_seconds:
            raise ArtifactError("resume consumed wall time is outside the fixed budget")
        store = _open_existing_store(root, resolved)
        states, trackers = _restore_states(store, resolved)
        _append_progress(
            store,
            "calibration_resumed",
            prior_consumed_wall_seconds=prior_consumed,
            remaining_wall_seconds=resolved.shared_wall_seconds - prior_consumed,
        )
    else:
        store = ArtifactStore(root, resolved.to_json(), repo_root=REPOSITORY_ROOT)
        store.initialize(
            knowledge_manifest=_knowledge_manifest(resolved),
            seed=canonical_json(
                {
                    "training": list(resolved.training_seeds),
                    "selection": resolved.selection_evaluation_root_seed,
                    "audit": resolved.audit_evaluation_root_seed,
                }
            ),
            budget={
                "shared_wall_seconds": resolved.shared_wall_seconds,
                "finalization_reserve_seconds": resolved.finalization_reserve_seconds,
                "screen_wall_seconds": resolved.screen_wall_seconds,
                "screen_target_episode": resolved.screen_target_episode,
                "confirm_target_episode": resolved.confirm_target_episode,
            },
        )
        states = _new_states(resolved)
        trackers = {
            state.key: ExplorationTracker(resolved.exploration_sample_stride) for state in states
        }
        for state in states:
            _write_run_manifest(store, state)
            _write_coverage(store, state, trackers[state.key], checkpoint_episode=0)
        _append_progress(
            store,
            "calibration_started",
            shared_wall_seconds=resolved.shared_wall_seconds,
            screen_wall_seconds=resolved.screen_wall_seconds,
            candidate_count=len(resolved.candidates),
            run_count=len(states),
        )

    started = clock()
    process_started = process_clock()
    remaining = max(0.0, resolved.shared_wall_seconds - prior_consumed)
    hard_deadline = started + remaining
    finalization_reserve = min(resolved.finalization_reserve_seconds, remaining)
    data_deadline = hard_deadline - finalization_reserve
    remaining_screen = max(0.0, resolved.screen_wall_seconds - prior_consumed)
    screen_deadline = min(data_deadline, started + remaining_screen)
    completed_keys = _evaluation_keys(root)
    stop_reason = "completed"
    contract_errors: list[str] = []
    survivor: str | None = None
    interrupts = _InterruptController()
    interrupts.install()
    try:
        checkpoint_zero = _ensure_milestones(
            store,
            states,
            checkpoint_episode=0,
            clock=clock,
            deadline=screen_deadline,
        )
        if checkpoint_zero is None or not _evaluate_round_robin(
            store,
            resolved,
            states,
            checkpoint_zero,
            suite="selection",
            checkpoint_episode=0,
            episodes=resolved.screen_evaluation_episodes,
            root_seed=resolved.selection_evaluation_root_seed,
            clock=clock,
            deadline=screen_deadline,
            process_clock=process_clock,
            interrupts=interrupts,
            completed_keys=completed_keys,
            phase_hook=phase_hook,
        ):
            stop_reason = "budget_exhausted"
        if stop_reason == "completed" and not _train_to_target(
            store,
            resolved,
            states,
            trackers,
            target_episode=resolved.screen_target_episode,
            clock=clock,
            deadline=screen_deadline,
            process_clock=process_clock,
            interrupts=interrupts,
            phase_hook=phase_hook,
        ):
            stop_reason = "budget_exhausted"
        checkpoint_screen = None
        if stop_reason == "completed":
            checkpoint_screen = _ensure_milestones(
                store,
                states,
                checkpoint_episode=resolved.screen_target_episode,
                clock=clock,
                deadline=screen_deadline,
            )
            if checkpoint_screen is None or not _evaluate_round_robin(
                store,
                resolved,
                states,
                checkpoint_screen,
                suite="selection",
                checkpoint_episode=resolved.screen_target_episode,
                episodes=resolved.screen_evaluation_episodes,
                root_seed=resolved.selection_evaluation_root_seed,
                clock=clock,
                deadline=screen_deadline,
                process_clock=process_clock,
                interrupts=interrupts,
                completed_keys=completed_keys,
                phase_hook=phase_hook,
            ):
                stop_reason = "budget_exhausted"

        decisions = _stage_decisions(root)
        if stop_reason == "completed":
            decision = derive_screen_decision(
                resolved,
                _evaluation_records(root),
                _training_records(root),
            )
            _write_or_validate_screen_decision(store, decision)
            raw_survivor = decision["survivor_candidate_id"]
            survivor = raw_survivor if isinstance(raw_survivor, str) else None
        elif decisions:
            raw_survivor = decisions[0].get("survivor_candidate_id")
            survivor = raw_survivor if isinstance(raw_survivor, str) else None

        if stop_reason == "completed" and survivor is None:
            _append_progress(store, "all_oi_eliminated", early_stop=True)
        elif stop_reason == "completed" and survivor is not None:
            selected_states = [state for state in states if state.arm.id in {"td0_zero", survivor}]
            if not _train_to_target(
                store,
                resolved,
                selected_states,
                trackers,
                target_episode=resolved.confirm_target_episode,
                clock=clock,
                deadline=data_deadline,
                process_clock=process_clock,
                interrupts=interrupts,
                phase_hook=phase_hook,
            ):
                stop_reason = "budget_exhausted"
            checkpoint_confirm = None
            if stop_reason == "completed":
                checkpoint_confirm = _ensure_milestones(
                    store,
                    selected_states,
                    checkpoint_episode=resolved.confirm_target_episode,
                    clock=clock,
                    deadline=data_deadline,
                )
                if checkpoint_confirm is None or not _evaluate_round_robin(
                    store,
                    resolved,
                    selected_states,
                    checkpoint_confirm,
                    suite="audit",
                    checkpoint_episode=resolved.confirm_target_episode,
                    episodes=resolved.audit_evaluation_episodes,
                    root_seed=resolved.audit_evaluation_root_seed,
                    clock=clock,
                    deadline=data_deadline,
                    process_clock=process_clock,
                    interrupts=interrupts,
                    completed_keys=completed_keys,
                    phase_hook=phase_hook,
                ):
                    stop_reason = "budget_exhausted"
    except _DiscoveryInterrupted:
        stop_reason = "interrupted"
    except KeyboardInterrupt as error:
        stop_reason = "contract_failed"
        contract_errors.append("unsafe KeyboardInterrupt bypassed cooperative SIGINT handling")
        store.failure(error)
    except Exception as error:
        stop_reason = "contract_failed"
        contract_errors.append(f"{type(error).__name__}: {error}")
        store.failure(error)
    finally:
        interrupts.restore()

    stopped_at = clock()
    measured_segment = max(0.0, stopped_at - started)
    process_seconds = prior_process + max(0.0, process_clock() - process_started)
    finalization_started = clock()
    provisional = {
        "schema_version": "algorithm-calibration-progress-v1",
        "event": "calibration_stopped",
        "stop_reason": stop_reason,
        "consumed_wall_seconds": prior_consumed + measured_segment,
        "measured_segment_wall_seconds": measured_segment,
        "measured_finalization_wall_seconds": 0.0,
        "process_cpu_seconds": process_seconds,
        "shared_wall_seconds": resolved.shared_wall_seconds,
    }
    try:
        summary = recompute_calibration_summary(
            root,
            contract_errors=contract_errors,
            stopped_record=provisional,
        )
        store.write_json("calibration-summary.json", summary)
        store.finalize(stop_reason=stop_reason, summary=summary)
    except Exception as error:
        stop_reason = "contract_failed"
        contract_errors.append(f"{type(error).__name__}: {error}")
        store.failure(error)

    finalization_seconds = max(0.0, clock() - finalization_started)
    consumed_total = prior_consumed + measured_segment + finalization_seconds
    if consumed_total > resolved.shared_wall_seconds and stop_reason != "contract_failed":
        stop_reason = "contract_failed"
        contract_errors.append("shared 600-second wall budget was exceeded during finalization")
    terminal = {
        "stop_reason": stop_reason,
        "consumed_wall_seconds": consumed_total,
        "measured_segment_wall_seconds": measured_segment,
        "measured_finalization_wall_seconds": finalization_seconds,
        "process_cpu_seconds": process_seconds,
        "shared_wall_seconds": resolved.shared_wall_seconds,
        "deadline_reached": consumed_total >= resolved.shared_wall_seconds,
        "survivor_candidate_id": survivor,
        "contract_errors": contract_errors,
    }
    _append_progress(store, "calibration_stopped", **terminal)
    summary = recompute_calibration_summary(root, contract_errors=contract_errors)
    store.write_json("calibration-summary.json", summary)
    store.finalize(stop_reason=stop_reason, summary=summary)
    verification = verify_calibration_artifact(root)
    store.write_json("verification.json", verification)
    if verification.get("valid") is not True and stop_reason != "contract_failed":
        stop_reason = "contract_failed"
        contract_errors.append(
            "post-run independent verifier rejected the finalized artifact: "
            + "; ".join(str(item) for item in verification.get("errors", []))
        )
        terminal.update(
            {
                "stop_reason": stop_reason,
                "contract_errors": contract_errors,
            }
        )
        _append_progress(store, "calibration_stopped", **terminal)
        summary = recompute_calibration_summary(root)
        store.write_json("calibration-summary.json", summary)
        store.finalize(stop_reason=stop_reason, summary=summary)
        store.write_json("verification.json", verify_calibration_artifact(root))
    return summary


def verify_calibration_artifact(artifact_directory: str | Path) -> dict[str, Any]:
    """Recompute the calibration decision and fail closed on contract drift."""

    root = Path(artifact_directory)
    errors: list[str] = []
    try:
        if not root.is_dir():
            raise ArtifactError(f"calibration artifact directory does not exist: {root}")
        for path in root.rglob("*"):
            if path.is_symlink():
                raise ArtifactError(f"calibration artifact contains a symlink: {path}")
        required = (
            "resolved-config.json",
            "run-manifest.json",
            "knowledge-manifest.json",
            "calibration-progress.jsonl",
            "calibration-summary.json",
        )
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise ArtifactError(
                "calibration artifact is missing required files: " + ", ".join(missing)
            )
        config = resolve_calibration_config(_read_json(root / "resolved-config.json"))
        manifest = _read_json(root / "run-manifest.json")
        expected_config_hash = config_hash(config.to_json())
        if manifest.get("config_hash") != expected_config_hash:
            raise ArtifactError("run manifest config hash mismatch")
        KnowledgeManifest.from_json(_read_json(root / "knowledge-manifest.json"))

        evaluations = _evaluation_records(root)
        seen: set[tuple[str, str, str, int, int]] = set()
        candidate_ids = {candidate.id for candidate in config.candidates}
        for record in evaluations:
            candidate_id = record.get("candidate_id")
            training_seed = record.get("training_seed")
            suite = record.get("suite")
            checkpoint_episode = record.get("checkpoint_episode")
            episode_id = record.get("evaluation_episode_id")
            if (
                candidate_id not in candidate_ids
                or training_seed not in config.training_seeds
                or suite not in {"selection", "audit"}
                or type(checkpoint_episode) is not int
                or type(episode_id) is not int
            ):
                raise ArtifactError("calibration evaluation identity is malformed")
            key = (
                cast(str, candidate_id),
                cast(str, training_seed),
                cast(str, suite),
                checkpoint_episode,
                episode_id,
            )
            if key in seen:
                raise ArtifactError(f"duplicate calibration evaluation record: {key}")
            seen.add(key)
            expected_seed = (
                config.selection_evaluation_root_seed
                if suite == "selection"
                else config.audit_evaluation_root_seed
            )
            if record.get("evaluation_root_seed") != expected_seed:
                raise ArtifactError("calibration evaluation seed suite mismatch")
            if suite == "selection" and checkpoint_episode not in {
                0,
                config.screen_target_episode,
            }:
                raise ArtifactError("selection suite was used outside the screen stage")
            if suite == "audit" and checkpoint_episode != config.confirm_target_episode:
                raise ArtifactError("audit suite was used outside the final gate")
            if (
                record.get("frozen_state_unchanged") is not True
                or record.get("clone_state_hash_before") != record.get("clone_state_hash_after")
                or record.get("clone_table_hash_before") != record.get("clone_table_hash_after")
                or record.get("training_state_hash_before")
                != record.get("training_state_hash_after")
                or record.get("training_counters_unchanged") is not True
            ):
                raise ArtifactError("frozen evaluation state changed")

        execution = _execution_config(
            config,
            evaluation_root_seed=config.selection_evaluation_root_seed,
            evaluation_episodes=config.screen_evaluation_episodes,
        )
        for milestone_key, record in _milestone_records(root).items():
            candidate_id, training_seed, _ = milestone_key
            candidate = _candidate(config, candidate_id)
            step = record.get("global_env_step")
            if type(step) is not int:
                raise ArtifactError("milestone checkpoint step is invalid")
            directory = _validated_checkpoint_directory(
                _open_existing_store(root, config),
                record,
                step=step,
            )
            agent = _build_agent(execution, cast(DiscoveryArmConfig, candidate))
            snapshot: EngineSnapshot = agent.restore_checkpoint(
                directory,
                step,
                config_hash=expected_config_hash,
            )
            if record.get("learner_state_hash") != agent.learner.state_hash():
                raise ArtifactError("milestone learner state hash mismatch")
            if record.get("table_hash") != agent.learner.table_hash():
                raise ArtifactError("milestone table hash mismatch")
            lineage = snapshot.rng.lineage
            if (
                lineage.get("root_seed") != training_seed
                or lineage.get("purpose") != "train-env"
                or lineage.get("environment_id") != f"{config.experiment_id}-training"
            ):
                raise ArtifactError("milestone RNG lineage mismatch")

        coverage = _coverage_summary(root, config)
        if len(cast(Mapping[str, Any], coverage["runs"])) != (
            len(config.candidates) * len(config.training_seeds)
        ):
            raise ArtifactError("calibration coverage is missing a run")

        decisions = _stage_decisions(root)
        summary_screen_complete = recompute_calibration_summary(root)["screen_complete"]
        if summary_screen_complete:
            if len(decisions) != 1:
                raise ArtifactError("complete screen requires exactly one stage decision")
            derived = derive_screen_decision(config, evaluations, _training_records(root))
            if canonical_json(decisions[0]) != canonical_json(derived):
                raise ArtifactError("screen decision does not match selection raw records")
        elif len(decisions) > 1:
            raise ArtifactError("partial calibration contains duplicate stage decisions")

        stored_summary = _read_json(root / "calibration-summary.json")
        recomputed = recompute_calibration_summary(root)
        if canonical_json(stored_summary) != canonical_json(recomputed):
            raise ArtifactError("calibration summary does not match raw-derived reducer")
        if manifest.get("stop_reason") != recomputed.get("stop_reason"):
            raise ArtifactError("run manifest stop reason does not match summary")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    return {
        "schema_version": "algorithm-calibration-verification-v1",
        "valid": not errors,
        "gate": (
            _read_json(root / "calibration-summary.json").get("gate")
            if not errors and (root / "calibration-summary.json").is_file()
            else "contract-failed"
        ),
        "errors": errors,
        "artifact_directory": str(root),
    }
