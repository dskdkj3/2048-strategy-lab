"""Versioned, fair, and independently verifiable Discovery pilot runner."""

from __future__ import annotations

import json
import math
import signal
import statistics
import threading
import time
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import jsonschema  # type: ignore[import-untyped]

from strategy2048.engine.oracle import EngineSnapshot, OracleEnv
from strategy2048.experiments.artifacts import (
    ArtifactError,
    ArtifactStore,
    KnowledgeManifest,
    canonical_json,
    config_hash,
)
from strategy2048.experiments.evaluation import FrozenPolicyAgent, evaluate_frozen
from strategy2048.experiments.metrics import Metrics
from strategy2048.learning.td import DEFAULT_TUPLES, TD1PAgent, TDLearner, TupleValueFunction
from strategy2048.rules.core import max_tile_value

DISCOVERY_SCHEMA_VERSION = "discovery-pilot-v1"
DISCOVERY_SHARED_WALL_SECONDS = 900
DISCOVERY_FINALIZATION_RESERVE_SECONDS = 10.0
DISCOVERY_FINAL_WRITE_RESERVE_SECONDS = 1.0
DISCOVERY_CHECKPOINT_EPISODES = (0, 50, 200)
DISCOVERY_MAX_TRAINING_EPISODES = 200
DISCOVERY_DIAGNOSTIC_SCORE_MILESTONE = 5000
DISCOVERY_DIAGNOSTIC_TILE_MILESTONE = 256
DISCOVERY_ARM_IDS = ("td0_zero", "td0_optimistic")
DISCOVERY_GATES = (
    "pipeline-valid-signal-visible",
    "pipeline-valid-inconclusive",
    "performance-blocked",
    "contract-failed",
)
REPOSITORY_ROOT = Path(__file__).parents[3]
DISCOVERY_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/discovery-pilot.v1.schema.json"

ArmId = Literal["td0_zero", "td0_optimistic"]
Initialization = Literal["zero", "optimistic"]
DiscoveryGate = Literal[
    "pipeline-valid-signal-visible",
    "pipeline-valid-inconclusive",
    "performance-blocked",
    "contract-failed",
]
Clock = Callable[[], float]


class DiscoveryConfigError(ValueError):
    """The versioned Discovery config is malformed or semantically unsafe."""


class _DiscoveryInterrupted(Exception):
    """Internal control flow after a durable operator interruption."""


@dataclass(slots=True)
class _InterruptController:
    requested: bool = False
    _installed: bool = False
    _previous_handler: Any = None

    def install(self) -> None:
        if threading.current_thread() is not threading.main_thread():
            return
        self._previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._request)
        self._installed = True

    def restore(self) -> None:
        if self._installed:
            signal.signal(signal.SIGINT, self._previous_handler)
            self._installed = False

    def _request(self, _signum: int, _frame: object) -> None:
        self.requested = True


@dataclass(frozen=True, slots=True)
class DiscoveryArmConfig:
    id: ArmId
    initialization: Initialization
    optimistic_total_value: float

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "initialization": self.initialization,
            "optimistic_total_value": self.optimistic_total_value,
        }


@dataclass(frozen=True, slots=True)
class DiscoveryLearnerConfig:
    alpha: float
    gamma: float
    symmetry: bool
    value_cardinality: int
    tuples: tuple[tuple[int, ...], ...] = DEFAULT_TUPLES

    def to_json(self) -> dict[str, object]:
        return {
            "alpha": self.alpha,
            "gamma": self.gamma,
            "symmetry": self.symmetry,
            "value_cardinality": self.value_cardinality,
            "tuples": [list(coordinates) for coordinates in self.tuples],
        }


@dataclass(frozen=True, slots=True)
class DiscoveryPilotConfig:
    experiment_id: str
    output_root: str
    round_robin_training_chunk: int
    training_seeds: tuple[str, str]
    evaluation_root_seed: str
    evaluation_episodes_per_checkpoint: int
    diagnostic_score_milestone: int
    diagnostic_tile_milestone: int
    learner: DiscoveryLearnerConfig
    arms: tuple[DiscoveryArmConfig, DiscoveryArmConfig]
    max_steps_per_episode: int | None = None
    schema_version: str = DISCOVERY_SCHEMA_VERSION
    shared_wall_seconds: int = DISCOVERY_SHARED_WALL_SECONDS
    finalization_reserve_seconds: float = DISCOVERY_FINALIZATION_RESERVE_SECONDS
    checkpoint_episodes: tuple[int, int, int] = DISCOVERY_CHECKPOINT_EPISODES
    max_training_episodes_per_run: int = DISCOVERY_MAX_TRAINING_EPISODES

    def to_json(self) -> dict[str, object]:
        value: dict[str, object] = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "output_root": self.output_root,
            "shared_wall_seconds": self.shared_wall_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "round_robin_training_chunk": self.round_robin_training_chunk,
            "training_seeds": list(self.training_seeds),
            "evaluation_root_seed": self.evaluation_root_seed,
            "checkpoint_episodes": list(self.checkpoint_episodes),
            "evaluation_episodes_per_checkpoint": self.evaluation_episodes_per_checkpoint,
            "diagnostic_score_milestone": self.diagnostic_score_milestone,
            "diagnostic_tile_milestone": self.diagnostic_tile_milestone,
            "max_training_episodes_per_run": self.max_training_episodes_per_run,
            "max_steps_per_episode": self.max_steps_per_episode,
            "learner": self.learner.to_json(),
            "arms": [arm.to_json() for arm in self.arms],
        }
        return value


@dataclass(slots=True)
class _RunState:
    arm: DiscoveryArmConfig
    training_seed: str
    agent: TD1PAgent
    relative_root: str
    metrics: Metrics = field(default_factory=Metrics)
    completed_episodes: int = 0
    global_env_steps: int = 0
    process_cpu_seconds: float = 0.0
    active_wall_seconds: float = 0.0
    last_snapshot: EngineSnapshot | None = None
    resume_snapshot: EngineSnapshot | None = None
    resume_episode_env_steps_before: int | None = None
    resume_episode_counters_before: dict[str, object] | None = None
    resume_episode_wall_seconds: float = 0.0
    resume_episode_process_cpu_seconds: float = 0.0

    @property
    def key(self) -> tuple[str, str]:
        return self.arm.id, self.training_seed


def _load_schema() -> dict[str, Any]:
    value = json.loads(DISCOVERY_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DiscoveryConfigError("Discovery schema must be an object")
    return value


def _schema_error(error: jsonschema.ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    prefix = f" at {location}" if location else ""
    return f"Discovery config schema validation failed{prefix}: {error.message}"


def resolve_discovery_config(value: Mapping[str, Any]) -> DiscoveryPilotConfig:
    """Validate and canonicalize the only supported Discovery protocol revision."""

    raw = dict(value)
    if "optimistic_value" in raw or any(
        isinstance(item, dict) and "optimistic_value" in item
        for item in cast(Sequence[object], raw.get("arms", ()))
    ):
        raise DiscoveryConfigError(
            "optimistic_value is ambiguous; migrate to optimistic_total_value"
        )
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(raw)
    except jsonschema.ValidationError as error:
        raise DiscoveryConfigError(_schema_error(error)) from error

    training_seeds_value = cast(list[str], raw["training_seeds"])
    training_seeds = (training_seeds_value[0], training_seeds_value[1])
    evaluation_root_seed = cast(str, raw["evaluation_root_seed"])
    if evaluation_root_seed in training_seeds:
        raise DiscoveryConfigError("evaluation_root_seed must be separate from training seeds")

    arms_by_id: dict[str, Mapping[str, Any]] = {}
    for arm_value in cast(list[Mapping[str, Any]], raw["arms"]):
        arm_id = cast(str, arm_value["id"])
        if arm_id in arms_by_id:
            raise DiscoveryConfigError(f"duplicate Discovery arm id: {arm_id}")
        arms_by_id[arm_id] = arm_value
    if set(arms_by_id) != set(DISCOVERY_ARM_IDS):
        raise DiscoveryConfigError("v1 requires exactly td0_zero and td0_optimistic arms")

    canonical_arms: list[DiscoveryArmConfig] = []
    for arm_id, expected_initialization in zip(
        DISCOVERY_ARM_IDS, ("zero", "optimistic"), strict=True
    ):
        arm_value = arms_by_id[arm_id]
        initialization = cast(str, arm_value["initialization"])
        if initialization != expected_initialization:
            raise DiscoveryConfigError(f"{arm_id} initialization must be {expected_initialization}")
        total = float(arm_value.get("optimistic_total_value", 0.0))
        if not math.isfinite(total):
            raise DiscoveryConfigError("optimistic_total_value must be finite")
        if arm_id == "td0_zero" and total != 0.0:
            raise DiscoveryConfigError("td0_zero optimistic_total_value must be exactly zero")
        if arm_id == "td0_optimistic" and total <= 0.0:
            raise DiscoveryConfigError("td0_optimistic optimistic_total_value must be positive")
        canonical_arms.append(
            DiscoveryArmConfig(
                id=cast(ArmId, arm_id),
                initialization=cast(Initialization, initialization),
                optimistic_total_value=total,
            )
        )

    learner_value = cast(Mapping[str, Any], raw["learner"])
    tuples_value = learner_value.get("tuples")
    tuples = (
        DEFAULT_TUPLES
        if tuples_value is None
        else tuple(
            tuple(int(index) for index in item) for item in cast(list[list[int]], tuples_value)
        )
    )
    if any(len(item) != len(tuples[0]) for item in tuples):
        raise DiscoveryConfigError("all learner tuples must have the same length")

    alpha = float(learner_value["alpha"])
    gamma = float(learner_value["gamma"])
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise DiscoveryConfigError("learner.alpha must be finite and in (0, 1]")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise DiscoveryConfigError("learner.gamma must be finite and in [0, 1]")

    config = DiscoveryPilotConfig(
        experiment_id=cast(str, raw["experiment_id"]),
        output_root=cast(str, raw["output_root"]),
        round_robin_training_chunk=cast(int, raw["round_robin_training_chunk"]),
        training_seeds=training_seeds,
        evaluation_root_seed=evaluation_root_seed,
        evaluation_episodes_per_checkpoint=cast(int, raw["evaluation_episodes_per_checkpoint"]),
        diagnostic_score_milestone=cast(int, raw["diagnostic_score_milestone"]),
        diagnostic_tile_milestone=cast(int, raw["diagnostic_tile_milestone"]),
        max_steps_per_episode=cast(int | None, raw.get("max_steps_per_episode")),
        learner=DiscoveryLearnerConfig(
            alpha=alpha,
            gamma=gamma,
            symmetry=cast(bool, learner_value["symmetry"]),
            value_cardinality=cast(int, learner_value["value_cardinality"]),
            tuples=tuples,
        ),
        arms=cast(tuple[DiscoveryArmConfig, DiscoveryArmConfig], tuple(canonical_arms)),
    )
    # The canonical form is itself schema-valid.  This catches resolver/schema drift.
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(config.to_json())
    except jsonschema.ValidationError as error:
        raise DiscoveryConfigError(_schema_error(error)) from error
    return config


def load_discovery_config(path: str | Path) -> DiscoveryPilotConfig:
    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    return resolve_discovery_config(value)


def _build_agent(config: DiscoveryPilotConfig, arm: DiscoveryArmConfig) -> TD1PAgent:
    value_function = TupleValueFunction(
        tuples=config.learner.tuples,
        value_cardinality=config.learner.value_cardinality,
        symmetry=config.learner.symmetry,
        optimistic_total_value=arm.optimistic_total_value,
    )
    learner = TDLearner(
        value_function=value_function,
        alpha=config.learner.alpha,
        gamma=config.learner.gamma,
        optimistic_total_value=arm.optimistic_total_value,
    )
    return TD1PAgent(learner=learner)


def _overall_manifest(config: DiscoveryPilotConfig) -> KnowledgeManifest:
    optimistic = next(arm for arm in config.arms if arm.id == "td0_optimistic")
    active_feature_count = len(config.learner.tuples) * (8 if config.learner.symmetry else 1)
    return KnowledgeManifest(
        initialization={
            "source": "optimistic",
            "comparison_arms": list(DISCOVERY_ARM_IDS),
            "optimistic_total_value": optimistic.optimistic_total_value,
            "active_feature_count": active_feature_count,
            "initial_feature_value": optimistic.optimistic_total_value / active_feature_count,
        }
    )


def _deadline_reached(clock: Clock, deadline: float) -> bool:
    return clock() >= deadline


def _append_progress(store: ArtifactStore, event: str, **fields: object) -> None:
    store.append_jsonl(
        "progress.jsonl",
        {"schema_version": "discovery-progress-v1", "event": event, **fields},
    )


def _counter_delta(before: Mapping[str, object], after: Mapping[str, object]) -> dict[str, int]:
    result: dict[str, int] = {}
    for name in ("action_value_calls", "tuple_lookups", "updates", "tuple_updates"):
        before_value = before.get(name, 0)
        after_value = after.get(name, before_value)
        if type(before_value) is int and type(after_value) is int:
            result[name] = after_value - before_value
    return result


def _initial_snapshot(config: DiscoveryPilotConfig, training_seed: str) -> EngineSnapshot:
    env = OracleEnv(
        root_seed=training_seed,
        environment_id=f"{config.experiment_id}-training",
        max_steps=config.max_steps_per_episode,
    )
    env.reset(episode_id=0, purpose="train-env")
    return env.snapshot()


def _open_existing_store(
    root: Path,
    resolved_config: Mapping[str, Any],
) -> ArtifactStore:
    """Re-open one immutable artifact directory for an explicit resume.

    ``ArtifactStore`` intentionally rejects non-empty roots for fresh runs.
    Resume is the one operation that must append to an existing source of
    truth, so it reuses the same store methods after validating the directory
    and config hash at the discovery boundary.
    """

    if not root.is_dir():
        raise ArtifactError(f"resume artifact directory does not exist: {root}")
    store = ArtifactStore.__new__(ArtifactStore)
    store.root = root
    store.resolved_config = dict(resolved_config)
    store.config_hash = config_hash(store.resolved_config)
    store.repo_root = REPOSITORY_ROOT
    store.started_at = time.time()
    return store


def _path_in_artifact(root: Path, relative: str) -> Path:
    """Resolve an artifact-relative path without allowing traversal."""

    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise ArtifactError(f"artifact path must be relative: {relative}")
    lexical = root
    for part in relative_path.parts:
        if part in {"", "."}:
            continue
        lexical /= part
        if lexical.is_symlink():
            raise ArtifactError(f"artifact path contains a symlink: {relative}")
    candidate = lexical.resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise ArtifactError(f"artifact path escapes resume root: {relative}") from error
    return candidate


def _assert_artifact_path(root: Path, path: Path) -> None:
    """Reject paths whose lexical components leave or link outside ``root``."""

    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ArtifactError(f"artifact path is outside resume root: {path}") from error
    _path_in_artifact(root, str(relative))


def _latest_stop_record(progress_records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for record in reversed(progress_records):
        if record.get("event") == "pilot_stopped":
            return record
    raise ArtifactError("resume artifact has no pilot_stopped progress record")


def _resume_metadata(
    root: Path,
    resolved: DiscoveryPilotConfig,
) -> tuple[ArtifactStore, float, float, list[dict[str, Any]]]:
    """Validate immutable resume inputs and return prior budget accounting."""

    preflight = verify_discovery_artifact(root)
    if preflight.get("valid") is not True:
        errors = preflight.get("errors", [])
        detail = "; ".join(str(error) for error in cast(Sequence[object], errors))
        raise ArtifactError(f"resume artifact failed read-only preflight verification: {detail}")
    store = _open_existing_store(root, resolved.to_json())
    resolved_path = root / "resolved-config.json"
    manifest_path = root / "run-manifest.json"
    progress_path = root / "progress.jsonl"
    if not resolved_path.is_file() or not manifest_path.is_file() or not progress_path.is_file():
        raise ArtifactError("resume artifact is missing its durable control files")
    stored_config = _read_json(resolved_path)
    if canonical_json(stored_config) != canonical_json(resolved.to_json()):
        raise ArtifactError("resume config does not match resolved-config.json")
    manifest = _read_json(manifest_path)
    if manifest.get("config_hash") != store.config_hash:
        raise ArtifactError("resume run manifest config hash mismatch")
    if manifest.get("stop_reason") not in {"budget_exhausted", "interrupted"}:
        raise ArtifactError(
            "resume requires a prior budget-exhausted or interrupted artifact; "
            "completed or failed runs are immutable"
        )
    progress_records = _read_jsonl(progress_path)
    stopped = _latest_stop_record(progress_records)
    if stopped.get("stop_reason") not in {"budget_exhausted", "interrupted"}:
        raise ArtifactError(
            "resume requires the latest pilot stop reason to be budget_exhausted or interrupted"
        )
    consumed = stopped.get("consumed_wall_seconds")
    process_cpu = stopped.get("process_cpu_seconds", 0.0)
    if isinstance(consumed, bool) or not isinstance(consumed, (int, float)):
        raise ArtifactError("resume consumed wall time is not numeric")
    if isinstance(process_cpu, bool) or not isinstance(process_cpu, (int, float)):
        raise ArtifactError("resume process CPU time is not numeric")
    consumed_float = float(consumed)
    process_float = float(process_cpu)
    if not math.isfinite(consumed_float) or consumed_float < 0:
        raise ArtifactError("resume consumed wall time is invalid")
    recorded_budget = stopped.get("shared_wall_seconds", resolved.shared_wall_seconds)
    if type(recorded_budget) is not int or recorded_budget != resolved.shared_wall_seconds:
        raise ArtifactError("resume shared wall budget does not match the resolved config")
    if (
        stopped.get("deadline_reached") is not None
        and type(stopped["deadline_reached"]) is not bool
    ):
        raise ArtifactError("resume deadline state is invalid")
    if stopped.get("deadline_reached") is not None and stopped["deadline_reached"] != (
        consumed_float >= resolved.shared_wall_seconds
    ):
        raise ArtifactError("resume deadline state does not match consumed wall time")
    if not math.isfinite(process_float) or process_float < 0:
        raise ArtifactError("resume process CPU time is invalid")
    return store, consumed_float, process_float, progress_records


def _validated_checkpoint_directory(
    store: ArtifactStore,
    checkpoint: Mapping[str, object],
    *,
    step: int,
) -> Path:
    directory_value = checkpoint.get("checkpoint_directory")
    array_value = checkpoint.get("array_path")
    metadata_value = checkpoint.get("metadata_path")
    if not all(isinstance(value, str) for value in (directory_value, array_value, metadata_value)):
        raise ArtifactError("checkpoint index is missing its contained path triplet")
    directory = _path_in_artifact(store.root, cast(str, directory_value))
    array_path = _path_in_artifact(store.root, cast(str, array_value))
    metadata_path = _path_in_artifact(store.root, cast(str, metadata_value))
    if array_path != directory / f"{step}.npz":
        raise ArtifactError("checkpoint array path does not match checkpoint step")
    if metadata_path != directory / f"{step}.json":
        raise ArtifactError("checkpoint metadata path does not match checkpoint step")
    if not array_path.is_file() or not metadata_path.is_file():
        raise ArtifactError("checkpoint pair is missing from the artifact root")
    return directory


def _restore_metrics(record: Mapping[str, Any] | None) -> Metrics:
    if record is None:
        return Metrics()
    payload = record.get("metrics")
    if not isinstance(payload, Mapping):
        raise ArtifactError("resume metrics record is malformed")
    counters_value = payload.get("counters", {})
    wall_value = payload.get("wall_seconds", {})
    if not isinstance(counters_value, Mapping) or not isinstance(wall_value, Mapping):
        raise ArtifactError("resume metrics counters or wall_seconds are malformed")
    counters: dict[str, int] = {}
    for name, value in counters_value.items():
        if type(value) is not int or value < 0:
            raise ArtifactError(f"resume metric counter is invalid: {name}")
        counters[str(name)] = value
    wall_seconds: dict[str, float] = {}
    for name, value in wall_value.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ArtifactError(f"resume metric timer is invalid: {name}")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric < 0:
            raise ArtifactError(f"resume metric timer is invalid: {name}")
        wall_seconds[str(name)] = numeric
    return Metrics(counters=counters, wall_seconds=wall_seconds)


def _validate_metrics_against_checkpoint(
    metrics_record: Mapping[str, Any],
    checkpoint_record: Mapping[str, Any],
    *,
    require_durable_fields: bool = False,
) -> None:
    payload = metrics_record.get("metrics")
    learner_counters = checkpoint_record.get("counters")
    completed = checkpoint_record.get("completed_training_episodes")
    global_steps = checkpoint_record.get("global_env_step")
    if not isinstance(payload, Mapping) or not isinstance(learner_counters, Mapping):
        raise ArtifactError("durable metrics or learner counters are malformed")
    metric_counters = payload.get("counters")
    wall_seconds = payload.get("wall_seconds")
    if not isinstance(metric_counters, Mapping) or not isinstance(wall_seconds, Mapping):
        raise ArtifactError("durable metric counters or timers are malformed")
    if type(completed) is not int or type(global_steps) is not int:
        raise ArtifactError("durable metrics progress is malformed")
    if metric_counters.get("games", 0) != completed:
        raise ArtifactError("durable metrics games do not match completed episodes")
    if metric_counters.get("env_steps", 0) != global_steps:
        raise ArtifactError("durable metrics env_steps do not match checkpoint progress")
    for name in ("action_value_calls", "tuple_lookups", "updates", "tuple_updates"):
        if metric_counters.get(name, 0) != learner_counters.get(name, 0):
            raise ArtifactError(f"durable metric counter does not match learner: {name}")
    process_cpu_seconds = metrics_record.get("process_cpu_seconds", 0.0)
    if isinstance(process_cpu_seconds, bool) or not isinstance(process_cpu_seconds, (int, float)):
        raise ArtifactError("durable process CPU metric is invalid")
    if not math.isfinite(float(process_cpu_seconds)) or float(process_cpu_seconds) < 0.0:
        raise ArtifactError("durable process CPU metric is invalid")
    active_wall_seconds = metrics_record.get("active_wall_seconds")
    if require_durable_fields and active_wall_seconds is None:
        raise ArtifactError("resume checkpoint is missing active wall time")
    if active_wall_seconds is not None:
        if isinstance(active_wall_seconds, bool) or not isinstance(
            active_wall_seconds, (int, float)
        ):
            raise ArtifactError("durable active wall metric is invalid")
        numeric_active = float(active_wall_seconds)
        raw_end_to_end = wall_seconds.get("end_to_end")
        if (
            not math.isfinite(numeric_active)
            or numeric_active < 0.0
            or isinstance(raw_end_to_end, bool)
            or not isinstance(raw_end_to_end, (int, float))
            or not math.isclose(float(raw_end_to_end), numeric_active, abs_tol=1e-9)
        ):
            raise ArtifactError("durable end-to-end wall metric is inconsistent")


def _resume_episode_boundary(
    record: Mapping[str, Any], snapshot: EngineSnapshot
) -> tuple[int, dict[str, object], float, float] | None:
    resume_in_episode = record.get("resume_in_episode")
    payload = record.get("resume_episode")
    if type(resume_in_episode) is not bool:
        raise ArtifactError("resume checkpoint boundary flag is invalid")
    if not resume_in_episode:
        if payload is not None:
            raise ArtifactError("boundary checkpoint must not contain episode accumulators")
        return None
    if not isinstance(payload, Mapping):
        raise ArtifactError("in-episode resume checkpoint is missing episode accumulators")
    completed = record.get("completed_training_episodes")
    global_steps = record.get("global_env_step")
    env_steps_before = payload.get("env_steps_before")
    counters_before = payload.get("counters_before")
    wall_seconds = payload.get("wall_seconds")
    process_cpu_seconds = payload.get("process_cpu_seconds")
    if (
        type(completed) is not int
        or type(global_steps) is not int
        or type(env_steps_before) is not int
        or env_steps_before < 0
        or env_steps_before > global_steps
    ):
        raise ArtifactError("resume episode step accumulator is invalid")
    if snapshot.episode_id != completed or snapshot.terminated or snapshot.truncated:
        raise ArtifactError("resume checkpoint in-episode boundary is inconsistent")
    current_counters = record.get("counters")
    if not isinstance(counters_before, Mapping) or not isinstance(current_counters, Mapping):
        raise ArtifactError("resume episode counter accumulator is invalid")
    if set(counters_before) != set(current_counters):
        raise ArtifactError("resume episode counter fields do not match learner counters")
    for name in ("action_value_calls", "tuple_lookups", "updates", "tuple_updates"):
        before_value = counters_before.get(name)
        current_value = current_counters.get(name)
        if (
            type(before_value) is not int
            or type(current_value) is not int
            or before_value < 0
            or before_value > current_value
        ):
            raise ArtifactError(f"resume episode counter accumulator is invalid: {name}")
    before_error = counters_before.get("td_error_abs_sum")
    current_error = current_counters.get("td_error_abs_sum")
    if (
        isinstance(before_error, bool)
        or not isinstance(before_error, (int, float))
        or isinstance(current_error, bool)
        or not isinstance(current_error, (int, float))
        or not math.isfinite(float(before_error))
        or float(before_error) < 0.0
        or float(before_error) > float(current_error)
    ):
        raise ArtifactError("resume episode TD-error accumulator is invalid")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in (wall_seconds, process_cpu_seconds)
    ):
        raise ArtifactError("resume episode timing accumulator is invalid")
    return (
        env_steps_before,
        dict(counters_before),
        float(cast(float | int, wall_seconds)),
        float(cast(float | int, process_cpu_seconds)),
    )


def _restore_run_states(
    store: ArtifactStore,
    config: DiscoveryPilotConfig,
    progress_records: Sequence[Mapping[str, Any]],
) -> list[_RunState]:
    """Restore the latest verified learner/environment pair for every run."""

    del progress_records
    checkpoint_records = _read_jsonl(store.root / "checkpoints.jsonl")
    latest_checkpoint: dict[tuple[str, str], Mapping[str, Any]] = {}
    for record in checkpoint_records:
        arm_id = record.get("arm_id")
        training_seed = record.get("training_seed")
        if not isinstance(arm_id, str) or not isinstance(training_seed, str):
            raise ArtifactError("resume checkpoint is missing arm_id or training_seed")
        if arm_id not in DISCOVERY_ARM_IDS or training_seed not in config.training_seeds:
            raise ArtifactError("resume checkpoint belongs to an unknown run")
        latest_checkpoint[(arm_id, training_seed)] = record

    metrics_by_run: dict[tuple[str, str], Mapping[str, Any]] = {}
    for path in sorted((store.root / "runs").glob("*/*/metrics.jsonl")):
        _assert_artifact_path(store.root, path)
        records = _read_jsonl(path)
        if records:
            last = records[-1]
            arm_id = last.get("arm_id")
            training_seed = last.get("training_seed")
            if isinstance(arm_id, str) and isinstance(training_seed, str):
                metrics_by_run[(arm_id, training_seed)] = last

    states: list[_RunState] = []
    for training_seed in config.training_seeds:
        for arm in config.arms:
            key = (arm.id, training_seed)
            state = _RunState(
                arm=arm,
                training_seed=training_seed,
                agent=_build_agent(config, arm),
                relative_root=f"runs/{arm.id}/{training_seed}",
                metrics=_restore_metrics(metrics_by_run.get(key)),
            )
            resume_pointer_path = (
                store.root / f"runs/{arm.id}/{training_seed}/resume-checkpoint.json"
            )
            has_resume_pointer = resume_pointer_path.is_file()
            checkpoint = (
                _read_json(resume_pointer_path)
                if has_resume_pointer
                else latest_checkpoint.get(key)
            )
            if checkpoint is None:
                state.last_snapshot = _initial_snapshot(config, training_seed)
                _write_run_manifest(store, state)
                states.append(state)
                continue
            directory_value = checkpoint.get("checkpoint_directory")
            step_value = checkpoint.get("checkpoint_step", checkpoint.get("global_env_step"))
            if not isinstance(directory_value, str) or type(step_value) is not int:
                raise ArtifactError("resume checkpoint index has invalid path or step")
            directory = _validated_checkpoint_directory(store, checkpoint, step=step_value)
            snapshot = state.agent.restore_checkpoint(
                directory,
                step_value,
                config_hash=store.config_hash,
            )
            if checkpoint.get("config_hash") != store.config_hash:
                raise ArtifactError("resume checkpoint config hash mismatch")
            if checkpoint.get("learner_state_hash") != state.agent.learner.state_hash():
                raise ArtifactError("resume checkpoint learner state hash mismatch")
            if checkpoint.get("table_hash") != state.agent.learner.table_hash():
                raise ArtifactError("resume checkpoint table hash mismatch")
            if canonical_json(checkpoint.get("counters")) != canonical_json(
                state.agent.counters.to_json()
            ):
                raise ArtifactError("resume checkpoint counters mismatch")
            if canonical_json(checkpoint.get("environment")) != canonical_json(snapshot.to_json()):
                raise ArtifactError("resume checkpoint environment mismatch")
            if has_resume_pointer:
                _validate_metrics_against_checkpoint(
                    checkpoint,
                    checkpoint,
                    require_durable_fields=True,
                )
            lineage = snapshot.rng.lineage
            if (
                lineage.get("root_seed") != training_seed
                or lineage.get("purpose") != "train-env"
                or lineage.get("environment_id") != f"{config.experiment_id}-training"
            ):
                raise ArtifactError("resume checkpoint RNG lineage mismatch")
            completed = checkpoint.get("completed_training_episodes")
            global_steps = checkpoint.get("global_env_step")
            if type(completed) is not int or completed < 0 or type(global_steps) is not int:
                raise ArtifactError("resume checkpoint progress is invalid")
            state.completed_episodes = completed
            state.global_env_steps = global_steps
            state.last_snapshot = snapshot
            resume_episode = (
                _resume_episode_boundary(checkpoint, snapshot) if has_resume_pointer else None
            )
            if resume_episode is not None:
                state.resume_snapshot = snapshot
                (
                    state.resume_episode_env_steps_before,
                    state.resume_episode_counters_before,
                    state.resume_episode_wall_seconds,
                    state.resume_episode_process_cpu_seconds,
                ) = resume_episode
            metrics_record = checkpoint if has_resume_pointer else metrics_by_run.get(key)
            state.metrics = _restore_metrics(metrics_record)
            raw_active_seconds = (
                checkpoint.get("active_wall_seconds")
                if has_resume_pointer
                else state.metrics.wall_seconds.get("end_to_end", 0.0)
            )
            if isinstance(raw_active_seconds, bool) or not isinstance(
                raw_active_seconds, (int, float)
            ):
                raise ArtifactError("resume active wall metric is invalid")
            state.active_wall_seconds = float(raw_active_seconds)
            raw_process_seconds = (
                checkpoint.get("process_cpu_seconds", 0.0)
                if has_resume_pointer
                else (
                    metrics_record.get("process_cpu_seconds", 0.0)
                    if metrics_record is not None
                    else 0.0
                )
            )
            if isinstance(raw_process_seconds, bool) or not isinstance(
                raw_process_seconds, (int, float)
            ):
                raise ArtifactError("resume process CPU metric is invalid")
            state.process_cpu_seconds = float(raw_process_seconds)
            _write_run_manifest(store, state)
            states.append(state)
    return states


def _write_run_manifest(store: ArtifactStore, state: _RunState) -> None:
    manifest = state.agent.knowledge_manifest()
    manifest.validate()
    store.write_json(f"{state.relative_root}/knowledge-manifest.json", manifest.to_json())


def _state_metrics_snapshot(state: _RunState) -> dict[str, object]:
    metrics = state.metrics.snapshot()
    wall_seconds = cast(dict[str, float], metrics["wall_seconds"])
    wall_seconds["end_to_end"] = state.active_wall_seconds
    counters = cast(Mapping[str, int], metrics["counters"])
    rates = cast(dict[str, float], metrics["rates"])
    rates["end_to_end_env_steps_per_second"] = (
        counters.get("env_steps", 0) / state.active_wall_seconds
        if state.active_wall_seconds
        else 0.0
    )
    rates["games_per_hour"] = (
        counters.get("games", 0) * 3600.0 / state.active_wall_seconds
        if state.active_wall_seconds
        else 0.0
    )
    return metrics


def _write_training_metrics(
    store: ArtifactStore,
    state: _RunState,
    *,
    checkpoint_episode: int,
    kind: Literal["milestone", "partial"],
) -> None:
    store.append_jsonl(
        f"{state.relative_root}/metrics.jsonl",
        {
            "schema_version": "discovery-training-metrics-v1",
            "arm_id": state.arm.id,
            "training_seed": state.training_seed,
            "checkpoint_episode": checkpoint_episode,
            "kind": kind,
            "completed_training_episodes": state.completed_episodes,
            "global_env_step": state.global_env_steps,
            "process_cpu_seconds": state.process_cpu_seconds,
            "metrics": _state_metrics_snapshot(state),
        },
    )


def _save_resume_checkpoint(store: ArtifactStore, state: _RunState) -> dict[str, object]:
    """Persist the latest chunk boundary without changing the milestone index.

    Milestone checkpoints remain the public experiment records.  This small
    per-run pointer is the durability boundary that lets a later explicit
    resume recover episodes completed before another run hit the deadline.
    """

    if state.last_snapshot is None:
        raise ArtifactError("run state has no environment snapshot for resume checkpointing")
    relative_directory = f"{state.relative_root}/resume-checkpoint"
    checkpoint_directory = store.root / relative_directory
    active_started = time.perf_counter()
    with state.metrics.timer("checkpoint"):
        array_path, metadata_path = state.agent.save_checkpoint(
            checkpoint_directory,
            0,
            config_hash=store.config_hash,
            environment_snapshot=state.last_snapshot,
        )
    state.active_wall_seconds += time.perf_counter() - active_started
    resume_episode: dict[str, object] | None = None
    if state.resume_snapshot is not None:
        if (
            state.resume_episode_env_steps_before is None
            or state.resume_episode_counters_before is None
        ):
            raise ArtifactError("in-episode resume checkpoint is missing episode accumulators")
        resume_episode = {
            "env_steps_before": state.resume_episode_env_steps_before,
            "counters_before": state.resume_episode_counters_before,
            "wall_seconds": state.resume_episode_wall_seconds,
            "process_cpu_seconds": state.resume_episode_process_cpu_seconds,
        }
    record: dict[str, object] = {
        "schema_version": "discovery-resume-checkpoint-v1",
        "kind": "resume",
        "arm_id": state.arm.id,
        "training_seed": state.training_seed,
        "completed_training_episodes": state.completed_episodes,
        "global_env_step": state.global_env_steps,
        "resume_in_episode": state.resume_snapshot is not None,
        "resume_episode": resume_episode,
        "checkpoint_step": 0,
        "array_path": str(array_path.relative_to(store.root)),
        "metadata_path": str(metadata_path.relative_to(store.root)),
        "checkpoint_directory": relative_directory,
        "config_hash": store.config_hash,
        "learner_state_hash": state.agent.learner.state_hash(),
        "table_hash": state.agent.learner.table_hash(),
        "counters": state.agent.counters.to_json(),
        "metrics": _state_metrics_snapshot(state),
        "active_wall_seconds": state.active_wall_seconds,
        "process_cpu_seconds": state.process_cpu_seconds,
        "environment": state.last_snapshot.to_json(),
    }
    store.write_json(f"{state.relative_root}/resume-checkpoint.json", record)
    return record


def _save_checkpoint(
    store: ArtifactStore,
    state: _RunState,
    *,
    checkpoint_episode: int,
    kind: Literal["milestone", "partial"],
) -> dict[str, object]:
    if state.last_snapshot is None:
        raise ArtifactError("run state has no environment snapshot for checkpointing")
    relative_directory = (
        f"{state.relative_root}/checkpoints/episode-{checkpoint_episode}"
        if kind == "milestone"
        else (
            f"{state.relative_root}/partial-checkpoints/"
            f"episode-{state.completed_episodes}-step-{state.global_env_steps}"
        )
    )
    checkpoint_directory = store.root / relative_directory
    active_started = time.perf_counter()
    with state.metrics.timer("checkpoint"):
        array_path, metadata_path = state.agent.save_checkpoint(
            checkpoint_directory,
            state.global_env_steps,
            config_hash=store.config_hash,
            environment_snapshot=state.last_snapshot,
        )
    record: dict[str, object] = {
        "schema_version": "discovery-checkpoint-v1",
        "kind": kind,
        "arm_id": state.arm.id,
        "training_seed": state.training_seed,
        "checkpoint_episode": checkpoint_episode,
        "completed_training_episodes": state.completed_episodes,
        "global_env_step": state.global_env_steps,
        "array_path": str(array_path.relative_to(store.root)),
        "metadata_path": str(metadata_path.relative_to(store.root)),
        "checkpoint_directory": relative_directory,
        "config_hash": store.config_hash,
        "learner_state_hash": state.agent.learner.state_hash(),
        "table_hash": state.agent.learner.table_hash(),
        "counters": state.agent.counters.to_json(),
        "environment": state.last_snapshot.to_json(),
    }
    store.append_jsonl("checkpoints.jsonl", record)
    _append_progress(
        store,
        "checkpoint_saved",
        arm_id=state.arm.id,
        training_seed=state.training_seed,
        checkpoint_episode=checkpoint_episode,
        kind=kind,
        global_env_step=state.global_env_steps,
    )
    state.active_wall_seconds += time.perf_counter() - active_started
    with state.metrics.timer("artifact_logging"):
        _write_training_metrics(
            store,
            state,
            checkpoint_episode=checkpoint_episode,
            kind=kind,
        )
    return record


def _record_training_partial(
    store: ArtifactStore,
    state: _RunState,
    *,
    episode_id: int,
    snapshot: EngineSnapshot,
    stop_reason: Literal["budget_exhausted", "interrupted"] = "budget_exhausted",
) -> None:
    state.last_snapshot = snapshot
    state.resume_snapshot = (
        snapshot
        if state.resume_episode_env_steps_before is not None
        and snapshot.episode_id == episode_id
        and not snapshot.terminated
        and not snapshot.truncated
        else None
    )
    checkpoint = _save_checkpoint(
        store,
        state,
        checkpoint_episode=state.completed_episodes,
        kind="partial",
    )
    _save_resume_checkpoint(store, state)
    record = {
        "schema_version": "discovery-training-partial-v1",
        "arm_id": state.arm.id,
        "training_seed": state.training_seed,
        "episode_id": episode_id,
        "step_id": snapshot.step_id,
        "global_env_step": state.global_env_steps,
        "stop_reason": stop_reason,
        "learner_state_hash": state.agent.learner.state_hash(),
        "environment": snapshot.to_json(),
        "checkpoint": checkpoint,
    }
    store.append_jsonl(f"{state.relative_root}/training-partials.jsonl", record)
    _append_progress(
        store,
        stop_reason,
        phase="training",
        arm_id=state.arm.id,
        training_seed=state.training_seed,
        episode_id=episode_id,
        step_id=snapshot.step_id,
    )


def _persist_interrupt_boundaries(
    store: ArtifactStore,
    states: Sequence[_RunState],
    *,
    phase: str,
) -> None:
    """Durably preserve all boundary states when interruption is outside training."""

    for state in states:
        if state.last_snapshot is not None:
            _save_resume_checkpoint(store, state)
    _append_progress(store, "interrupted", phase=phase)


def _train_one_episode(
    store: ArtifactStore,
    config: DiscoveryPilotConfig,
    state: _RunState,
    *,
    clock: Clock,
    deadline: float,
    process_clock: Clock,
    interrupts: _InterruptController,
    phase_hook: Callable[[str], None] | None,
) -> bool:
    episode_id = state.completed_episodes
    env: OracleEnv | None = None
    active_started: float | None = None
    cpu_started: float | None = None
    try:
        if interrupts.requested:
            if state.last_snapshot is None:
                raise ArtifactError("interrupted run has no boundary snapshot")
            _save_resume_checkpoint(store, state)
            _append_progress(
                store,
                "interrupted",
                phase="training_boundary",
                arm_id=state.arm.id,
                training_seed=state.training_seed,
                completed_training_episodes=state.completed_episodes,
            )
            raise _DiscoveryInterrupted
        if _deadline_reached(clock, deadline):
            if state.last_snapshot is None:
                raise ArtifactError("run state has no boundary snapshot at deadline")
            _record_training_partial(
                store,
                state,
                episode_id=episode_id,
                snapshot=state.last_snapshot,
            )
            return False
        env = OracleEnv(
            root_seed=state.training_seed,
            environment_id=f"{config.experiment_id}-training",
            max_steps=config.max_steps_per_episode,
        )
        if state.resume_snapshot is None:
            observation = env.reset(episode_id=episode_id, purpose="train-env")
            state.resume_episode_env_steps_before = state.global_env_steps
            state.resume_episode_counters_before = state.agent.counters.to_json()
            state.resume_episode_wall_seconds = 0.0
            state.resume_episode_process_cpu_seconds = 0.0
        else:
            env.restore(state.resume_snapshot)
            if env.episode_id != episode_id:
                raise ArtifactError("resume environment episode does not match scheduler progress")
            observation = env.observation()
            if (
                state.resume_episode_env_steps_before is None
                or state.resume_episode_counters_before is None
            ):
                raise ArtifactError("resumed episode is missing its durable accumulators")
        active_started = time.perf_counter()
        episode_started = clock()
        cpu_started = process_clock()
        assert state.resume_episode_counters_before is not None
        assert state.resume_episode_env_steps_before is not None
        counters_before = state.resume_episode_counters_before
        env_steps_before = state.resume_episode_env_steps_before
        while not observation.terminated and not observation.truncated:
            if _deadline_reached(clock, deadline):
                segment_cpu_seconds = max(0.0, process_clock() - cpu_started)
                state.process_cpu_seconds += segment_cpu_seconds
                state.resume_episode_process_cpu_seconds += segment_cpu_seconds
                state.resume_episode_wall_seconds += max(0.0, clock() - episode_started)
                state.active_wall_seconds += time.perf_counter() - active_started
                _record_training_partial(
                    store, state, episode_id=episode_id, snapshot=env.snapshot()
                )
                return False
            step_counters_before = state.agent.counters.to_json()
            with state.metrics.timer("action_selection"):
                action = state.agent.learner.choose_action(
                    observation,
                    feature_timer=lambda: state.metrics.timer("feature_value_lookup"),
                )
            if phase_hook is not None:
                phase_hook("env_step")
            with state.metrics.timer("rules"):
                result = env.step(action)
            state.metrics.increment("env_steps")
            state.global_env_steps += 1
            if phase_hook is not None:
                phase_hook("observe")
            with state.metrics.timer("learning"), state.metrics.timer("td_update"):
                state.agent.learner.observe(
                    result,
                    result.observation,
                    feature_timer=lambda: state.metrics.timer("feature_value_lookup"),
                )
            step_counters_after = state.agent.counters.to_json()
            for name, amount in _counter_delta(step_counters_before, step_counters_after).items():
                state.metrics.increment(name, amount)
            observation = result.observation
    except KeyboardInterrupt as error:
        raise ArtifactError(
            "unsafe KeyboardInterrupt bypassed the cooperative SIGINT boundary"
        ) from error

    assert env is not None
    assert active_started is not None
    assert cpu_started is not None
    state.completed_episodes += 1
    state.resume_snapshot = None
    state.metrics.increment("games")
    state.last_snapshot = env.snapshot()
    segment_cpu_seconds = max(0.0, process_clock() - cpu_started)
    episode_cpu_seconds = state.resume_episode_process_cpu_seconds + segment_cpu_seconds
    state.process_cpu_seconds += segment_cpu_seconds
    episode_wall_seconds = state.resume_episode_wall_seconds + max(0.0, clock() - episode_started)
    counters_after = state.agent.counters.to_json()
    deltas = _counter_delta(counters_before, counters_after)
    record: dict[str, object] = {
        "schema_version": "discovery-training-episode-v1",
        "arm_id": state.arm.id,
        "training_seed": state.training_seed,
        "episode_id": episode_id,
        "official_score": observation.score,
        "max_tile": max_tile_value(observation.board),
        "steps": state.global_env_steps - env_steps_before,
        "terminated": observation.terminated,
        "truncated": observation.truncated,
        "wall_seconds": episode_wall_seconds,
        "process_cpu_seconds": episode_cpu_seconds,
        "global_env_step": state.global_env_steps,
        "counter_delta": deltas,
        "counters": counters_after,
        "learner_state_hash": state.agent.learner.state_hash(),
        "environment_rng_lineage": dict(state.last_snapshot.rng.lineage),
    }
    state.resume_episode_env_steps_before = None
    state.resume_episode_counters_before = None
    state.resume_episode_wall_seconds = 0.0
    state.resume_episode_process_cpu_seconds = 0.0
    if phase_hook is not None:
        phase_hook("training_jsonl")
    with state.metrics.timer("artifact_logging"):
        store.append_jsonl(f"{state.relative_root}/training-episodes.jsonl", record)
    if phase_hook is not None:
        phase_hook("training_progress")
    _append_progress(
        store,
        "training_episode_completed",
        arm_id=state.arm.id,
        training_seed=state.training_seed,
        episode_id=episode_id,
        completed_training_episodes=state.completed_episodes,
        global_env_step=state.global_env_steps,
    )
    state.active_wall_seconds += time.perf_counter() - active_started
    if interrupts.requested:
        _save_resume_checkpoint(store, state)
        _append_progress(
            store,
            "interrupted",
            phase="training_boundary",
            arm_id=state.arm.id,
            training_seed=state.training_seed,
            completed_training_episodes=state.completed_episodes,
            global_env_step=state.global_env_steps,
        )
        raise _DiscoveryInterrupted
    return True


def _checkpoint_clone(
    store: ArtifactStore,
    config: DiscoveryPilotConfig,
    state: _RunState,
    checkpoint: Mapping[str, object],
) -> FrozenPolicyAgent:
    step = checkpoint.get("global_env_step")
    if type(step) is not int or step < 0:
        raise ArtifactError("milestone checkpoint step is invalid")
    directory = _validated_checkpoint_directory(store, checkpoint, step=step)
    clone = _build_agent(config, state.arm)
    clone.restore_checkpoint(
        directory,
        step,
        config_hash=store.config_hash,
    )
    return FrozenPolicyAgent(clone.learner)


def _evaluate_checkpoint_round_robin(
    store: ArtifactStore,
    config: DiscoveryPilotConfig,
    states: Sequence[_RunState],
    checkpoints: Mapping[tuple[str, str], Mapping[str, object]],
    *,
    checkpoint_episode: int,
    clock: Clock,
    deadline: float,
    process_clock: Clock,
    interrupts: _InterruptController,
    phase_hook: Callable[[str], None] | None,
    existing_evaluation_keys: set[tuple[str, str, int, int]] | None = None,
) -> bool:
    completed_keys = existing_evaluation_keys if existing_evaluation_keys is not None else set()
    contexts: list[tuple[_RunState, FrozenPolicyAgent, str, str, Mapping[str, object]]] = []
    for state in states:
        if all(
            (state.arm.id, state.training_seed, checkpoint_episode, evaluation_episode_id)
            in completed_keys
            for evaluation_episode_id in range(config.evaluation_episodes_per_checkpoint)
        ):
            continue
        if interrupts.requested:
            _persist_interrupt_boundaries(store, states, phase="evaluation_clone_boundary")
            raise _DiscoveryInterrupted
        if _deadline_reached(clock, deadline):
            _append_progress(
                store,
                "budget_exhausted",
                phase="evaluation_clone",
                checkpoint_episode=checkpoint_episode,
                arm_id=state.arm.id,
                training_seed=state.training_seed,
            )
            return False
        checkpoint = checkpoints[state.key]
        training_hash = state.agent.learner.state_hash()
        training_counters = canonical_json(state.agent.counters.to_json())
        clone = _checkpoint_clone(store, config, state, checkpoint)
        contexts.append((state, clone, training_hash, training_counters, checkpoint))

    for evaluation_episode_id in range(config.evaluation_episodes_per_checkpoint):
        for state, clone, training_hash, training_counters, checkpoint in contexts:
            evaluation_key = (
                state.arm.id,
                state.training_seed,
                checkpoint_episode,
                evaluation_episode_id,
            )
            if evaluation_key in completed_keys:
                continue
            if _deadline_reached(clock, deadline):
                _append_progress(
                    store,
                    "budget_exhausted",
                    phase="evaluation",
                    checkpoint_episode=checkpoint_episode,
                    evaluation_episode_id=evaluation_episode_id,
                    arm_id=state.arm.id,
                    training_seed=state.training_seed,
                )
                return False
            if phase_hook is not None:
                phase_hook("evaluation_episode")
            evaluation_cpu_started = process_clock()
            result = evaluate_frozen(
                clone,
                episodes=1,
                root_seed=config.evaluation_root_seed,
                purpose="discovery-eval",
                environment_id=f"{config.experiment_id}-evaluation",
                max_steps=config.max_steps_per_episode,
                episode_ids=(evaluation_episode_id,),
                clock=clock,
                deadline=deadline,
            )
            evaluation_cpu_seconds = max(0.0, process_clock() - evaluation_cpu_started)
            training_hash_after = state.agent.learner.state_hash()
            training_counters_after = canonical_json(state.agent.counters.to_json())
            if result["completed_episodes"] != 1:
                partial_record = {
                    "schema_version": "discovery-evaluation-partial-v1",
                    "arm_id": state.arm.id,
                    "training_seed": state.training_seed,
                    "checkpoint_episode": checkpoint_episode,
                    "evaluation_episode_id": evaluation_episode_id,
                    "evaluation_root_seed": config.evaluation_root_seed,
                    "purpose": "discovery-eval",
                    "stop_reason": "budget_exhausted",
                    "frozen_result": result,
                }
                if phase_hook is not None:
                    phase_hook("evaluation_partial_jsonl")
                store.append_jsonl(
                    f"{state.relative_root}/evaluation/{checkpoint_episode}/partials.jsonl",
                    partial_record,
                )
                if phase_hook is not None:
                    phase_hook("evaluation_partial_progress")
                _append_progress(
                    store,
                    "budget_exhausted",
                    phase="evaluation",
                    checkpoint_episode=checkpoint_episode,
                    evaluation_episode_id=evaluation_episode_id,
                    arm_id=state.arm.id,
                    training_seed=state.training_seed,
                )
                return False
            episode = cast(list[dict[str, object]], result["episodes"])[0]
            record: dict[str, object] = {
                "schema_version": "discovery-evaluation-episode-v1",
                "arm_id": state.arm.id,
                "training_seed": state.training_seed,
                "checkpoint_episode": checkpoint_episode,
                "checkpoint_global_env_step": checkpoint["global_env_step"],
                "evaluation_episode_id": evaluation_episode_id,
                "evaluation_root_seed": config.evaluation_root_seed,
                "purpose": "discovery-eval",
                "official_score": episode["official_score"],
                "max_tile": episode["max_tile"],
                "steps": episode["steps"],
                "wall_seconds": episode["wall_seconds"],
                "process_cpu_seconds": evaluation_cpu_seconds,
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
            if phase_hook is not None:
                phase_hook("evaluation_jsonl")
            with state.metrics.timer("artifact_logging"):
                store.append_jsonl(
                    f"{state.relative_root}/evaluation/{checkpoint_episode}/episodes.jsonl",
                    record,
                )
            evaluation_metrics = result.get("metrics")
            if isinstance(evaluation_metrics, Mapping):
                evaluation_wall = evaluation_metrics.get("wall_seconds")
                if isinstance(evaluation_wall, Mapping):
                    raw_evaluation_seconds = evaluation_wall.get("evaluation", 0.0)
                    if isinstance(raw_evaluation_seconds, (int, float)) and not isinstance(
                        raw_evaluation_seconds, bool
                    ):
                        state.metrics.add_wall("evaluation", float(raw_evaluation_seconds))
            completed_keys.add(evaluation_key)
            if phase_hook is not None:
                phase_hook("evaluation_progress")
            _append_progress(
                store,
                "evaluation_episode_completed",
                arm_id=state.arm.id,
                training_seed=state.training_seed,
                checkpoint_episode=checkpoint_episode,
                evaluation_episode_id=evaluation_episode_id,
            )
            if interrupts.requested:
                _persist_interrupt_boundaries(store, states, phase="evaluation_boundary")
                raise _DiscoveryInterrupted
    return True


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ArtifactError(f"JSON artifact path is a symlink: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    if path.is_symlink():
        raise ArtifactError(f"JSONL artifact path is a symlink: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ArtifactError(f"JSONL record must be an object: {path}:{line_number}")
        records.append(value)
    return records


def _aggregate_strength(records: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    scores = [float(record["official_score"]) for record in records]
    tiles = [int(record["max_tile"]) for record in records]
    return {
        "count": len(records),
        "official_score_mean": statistics.fmean(scores) if scores else 0.0,
        "official_score_range": [min(scores), max(scores)] if scores else [0.0, 0.0],
        "max_tile": max(tiles, default=0),
        "env_steps": sum(int(record["steps"]) for record in records),
        "wall_seconds": sum(float(record["wall_seconds"]) for record in records),
        "process_cpu_seconds": sum(
            float(record.get("process_cpu_seconds", 0.0)) for record in records
        ),
        "tile_reach_rate": {
            str(tile): sum(value >= tile for value in tiles) / len(tiles) if tiles else 0.0
            for tile in (128, 256, 512, 1024, 2048, 4096)
        },
    }


def _direction(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def _training_cost_at_checkpoint(
    training_records: Sequence[Mapping[str, Any]], checkpoint_episode: int
) -> dict[str, object]:
    prefix = [
        record
        for record in training_records
        if type(record.get("episode_id")) is int
        and cast(int, record["episode_id"]) < checkpoint_episode
    ]
    return {
        "training_episodes": len(prefix),
        "training_env_steps": sum(int(record["steps"]) for record in prefix),
        "training_wall_seconds": sum(float(record["wall_seconds"]) for record in prefix),
    }


def _diagnostic_milestone_efficiency(
    config: DiscoveryPilotConfig,
    training_records: Sequence[Mapping[str, Any]],
    evaluation_records: Sequence[Mapping[str, Any]],
    *,
    completed_artifact: bool,
) -> dict[str, object]:
    def attainment(kind: Literal["score", "tile"], target: int) -> dict[str, object]:
        observed_checkpoints: list[int] = []
        for checkpoint_episode in config.checkpoint_episodes:
            records = [
                record
                for record in evaluation_records
                if record.get("checkpoint_episode") == checkpoint_episode
            ]
            if not records:
                continue
            observed_checkpoints.append(checkpoint_episode)
            observed_value = (
                statistics.fmean(float(record["official_score"]) for record in records)
                if kind == "score"
                else max(int(record["max_tile"]) for record in records)
            )
            if observed_value >= target:
                return {
                    "target": target,
                    "status": "attained",
                    "checkpoint_episode": checkpoint_episode,
                    "observed_value": observed_value,
                    "evaluation_count": len(records),
                    **_training_cost_at_checkpoint(training_records, checkpoint_episode),
                }
        return {
            "target": target,
            "status": "not-attained" if completed_artifact else "inconclusive",
            "checkpoint_episode": None,
            "observed_value": None,
            "evaluation_count": 0,
            "observed_checkpoints": observed_checkpoints,
            "training_episodes": None,
            "training_env_steps": None,
            "training_wall_seconds": None,
        }

    return {
        "score": attainment("score", config.diagnostic_score_milestone),
        "tile": attainment("tile", config.diagnostic_tile_milestone),
    }


def _derive_next_step_decision(
    gate: DiscoveryGate,
    runs: Mapping[str, object],
) -> dict[str, object]:
    def phase_seconds(wall_seconds: Mapping[object, object], name: str) -> float:
        value = wall_seconds.get(name, 0.0)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return 0.0
        numeric = float(value)
        return numeric if math.isfinite(numeric) and numeric > 0.0 else 0.0

    end_to_end = 0.0
    learner_hot_path = 0.0
    rules = 0.0
    durability = 0.0
    for run in runs.values():
        if not isinstance(run, Mapping):
            continue
        metrics = run.get("training_metrics")
        if not isinstance(metrics, Mapping):
            continue
        wall = metrics.get("wall_seconds")
        if not isinstance(wall, Mapping):
            continue

        end_to_end += phase_seconds(wall, "end_to_end")
        learner_hot_path += phase_seconds(wall, "action_selection") + phase_seconds(
            wall, "learning"
        )
        rules += phase_seconds(wall, "rules")
        durability += phase_seconds(wall, "checkpoint") + phase_seconds(wall, "artifact_logging")

    hot_share = min(1.0, learner_hot_path / end_to_end) if end_to_end else 0.0
    rules_share = min(1.0, rules / end_to_end) if end_to_end else 0.0
    durability_share = min(1.0, durability / end_to_end) if end_to_end else 0.0
    two_x_speedup = 1.0 / ((1.0 - hot_share) + hot_share / 2.0) if end_to_end else 1.0
    two_x_gain = two_x_speedup - 1.0
    evidence_sufficient = end_to_end > 0.0

    if gate == "contract-failed" or not evidence_sufficient:
        decision = "stop-route"
        rationale = "contract-or-profile-evidence-insufficient"
    elif gate == "performance-blocked" and hot_share >= 0.5 and two_x_gain >= 0.3:
        decision = "native-core-child"
        rationale = "learner-hot-path-blocks-minimum-matrix-and-clears-amdahl-gate"
    elif gate == "performance-blocked" and (hot_share >= 0.3 or durability_share >= 0.25):
        decision = "python-optimization"
        rationale = "bounded-python-or-runner-optimization-before-more-training"
    elif gate in {"pipeline-valid-signal-visible", "pipeline-valid-inconclusive"}:
        decision = "continue-algorithm"
        rationale = "pilot-throughput-sufficient-so-learning-question-dominates"
    else:
        decision = "stop-route"
        rationale = "pilot-does-not-support-a-cost-effective-next-step"

    return {
        "decision": decision,
        "rationale": rationale,
        "evidence_sufficient": evidence_sufficient,
        "training_end_to_end_wall_seconds": end_to_end,
        "learner_hot_path_wall_seconds": learner_hot_path,
        "learner_hot_path_share": hot_share,
        "rules_share": rules_share,
        "durability_share": durability_share,
        "assumed_hot_path_speedup": 2.0,
        "estimated_overall_speedup": two_x_speedup,
        "estimated_overall_gain": two_x_gain,
        "native_core_gain_threshold": 0.3,
        "rules_only_rewrite_recommended": False,
    }


def classify_discovery_result(
    *,
    contract_errors: Sequence[str],
    minimum_comparable: bool,
    consistent_signal: bool,
) -> DiscoveryGate:
    """Apply the reviewed fail-closed precedence for the four public gates."""

    if contract_errors:
        return "contract-failed"
    if not minimum_comparable:
        return "performance-blocked"
    if consistent_signal:
        return "pipeline-valid-signal-visible"
    return "pipeline-valid-inconclusive"


def recompute_discovery_summary(
    artifact_directory: str | Path,
    *,
    contract_errors: Sequence[str] = (),
    stopped_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the diagnostic summary exclusively from durable raw records."""

    root = Path(artifact_directory)
    config = resolve_discovery_config(_read_json(root / "resolved-config.json"))
    training_records: list[dict[str, Any]] = []
    training_metrics_records: list[dict[str, Any]] = []
    evaluation_records: list[dict[str, Any]] = []
    for path in sorted((root / "runs").glob("*/*/training-episodes.jsonl")):
        _assert_artifact_path(root, path)
        training_records.extend(_read_jsonl(path))
    for path in sorted((root / "runs").glob("*/*/metrics.jsonl")):
        _assert_artifact_path(root, path)
        training_metrics_records.extend(_read_jsonl(path))
    for path in sorted((root / "runs").glob("*/*/evaluation/*/episodes.jsonl")):
        _assert_artifact_path(root, path)
        evaluation_records.extend(_read_jsonl(path))
    checkpoint_records = _read_jsonl(root / "checkpoints.jsonl")
    progress_records = _read_jsonl(root / "progress.jsonl")
    stopped = (
        dict(stopped_record)
        if stopped_record is not None
        else next(
            (
                record
                for record in reversed(progress_records)
                if record.get("event") == "pilot_stopped"
            ),
            {},
        )
    )
    completed_artifact = stopped.get("stop_reason") == "completed"

    runs: dict[str, object] = {}
    for training_seed in config.training_seeds:
        for arm in config.arms:
            key = f"{arm.id}/{training_seed}"
            run_training = [
                record
                for record in training_records
                if record.get("arm_id") == arm.id and record.get("training_seed") == training_seed
            ]
            run_evaluation = [
                record
                for record in evaluation_records
                if record.get("arm_id") == arm.id and record.get("training_seed") == training_seed
            ]
            run_metrics = [
                record
                for record in training_metrics_records
                if record.get("arm_id") == arm.id and record.get("training_seed") == training_seed
            ]
            evaluation_by_checkpoint = {
                str(checkpoint): _aggregate_strength(
                    [
                        record
                        for record in run_evaluation
                        if record.get("checkpoint_episode") == checkpoint
                    ]
                )
                for checkpoint in config.checkpoint_episodes
            }
            milestone_efficiency = _diagnostic_milestone_efficiency(
                config,
                run_training,
                run_evaluation,
                completed_artifact=completed_artifact,
            )
            runs[key] = {
                "arm_id": arm.id,
                "training_seed": training_seed,
                "training_episodes": len(run_training),
                "training_env_steps": sum(int(record["steps"]) for record in run_training),
                "training_updates": sum(
                    cast(int, cast(Mapping[str, object], record["counter_delta"]).get("updates", 0))
                    for record in run_training
                ),
                "training_wall_seconds": sum(
                    float(record["wall_seconds"]) for record in run_training
                ),
                "training_process_cpu_seconds": sum(
                    float(record["process_cpu_seconds"]) for record in run_training
                ),
                "training_metrics": run_metrics[-1]["metrics"] if run_metrics else None,
                "checkpoints": sorted(
                    int(record["checkpoint_episode"])
                    for record in checkpoint_records
                    if record.get("kind") == "milestone"
                    and record.get("arm_id") == arm.id
                    and record.get("training_seed") == training_seed
                ),
                "evaluation": evaluation_by_checkpoint,
                "milestone_efficiency": milestone_efficiency,
            }

    paired_differences: list[dict[str, object]] = []
    for training_seed in config.training_seeds:
        for checkpoint_episode in config.checkpoint_episodes:
            zero_by_episode = {
                int(record["evaluation_episode_id"]): record
                for record in evaluation_records
                if record.get("arm_id") == "td0_zero"
                and record.get("training_seed") == training_seed
                and record.get("checkpoint_episode") == checkpoint_episode
            }
            optimistic_by_episode = {
                int(record["evaluation_episode_id"]): record
                for record in evaluation_records
                if record.get("arm_id") == "td0_optimistic"
                and record.get("training_seed") == training_seed
                and record.get("checkpoint_episode") == checkpoint_episode
            }
            paired_ids = sorted(set(zero_by_episode) & set(optimistic_by_episode))
            differences = [
                float(optimistic_by_episode[episode_id]["official_score"])
                - float(zero_by_episode[episode_id]["official_score"])
                for episode_id in paired_ids
            ]
            paired_differences.append(
                {
                    "training_seed": training_seed,
                    "checkpoint_episode": checkpoint_episode,
                    "paired_episode_ids": paired_ids,
                    "count": len(differences),
                    "optimistic_minus_zero_score_mean": (
                        statistics.fmean(differences) if differences else 0.0
                    ),
                    "range": [min(differences), max(differences)] if differences else [0.0, 0.0],
                }
            )

    learning_directions: list[dict[str, object]] = []
    for arm in config.arms:
        for training_seed in config.training_seeds:
            seed_records = [
                record
                for record in evaluation_records
                if record.get("arm_id") == arm.id and record.get("training_seed") == training_seed
            ]
            available = sorted(
                {
                    int(record["checkpoint_episode"])
                    for record in seed_records
                    if int(record["checkpoint_episode"]) > 0
                }
            )
            latest = available[-1] if available else None
            delta = 0.0
            paired_count = 0
            if latest is not None:
                baseline = {
                    int(record["evaluation_episode_id"]): record
                    for record in seed_records
                    if record.get("checkpoint_episode") == 0
                }
                trained = {
                    int(record["evaluation_episode_id"]): record
                    for record in seed_records
                    if record.get("checkpoint_episode") == latest
                }
                paired_ids = sorted(set(baseline) & set(trained))
                paired_count = len(paired_ids)
                deltas = [
                    float(trained[episode_id]["official_score"])
                    - float(baseline[episode_id]["official_score"])
                    for episode_id in paired_ids
                ]
                delta = statistics.fmean(deltas) if deltas else 0.0
            learning_directions.append(
                {
                    "arm_id": arm.id,
                    "training_seed": training_seed,
                    "latest_checkpoint_episode": latest,
                    "paired_count": paired_count,
                    "paired_score_delta_mean": delta,
                    "direction": _direction(delta),
                }
            )

    minimum_comparable = all(
        bool(
            {
                int(record["evaluation_episode_id"])
                for record in evaluation_records
                if record.get("arm_id") == "td0_zero"
                and record.get("training_seed") == training_seed
                and record.get("checkpoint_episode") == checkpoint
            }
            & {
                int(record["evaluation_episode_id"])
                for record in evaluation_records
                if record.get("arm_id") == "td0_optimistic"
                and record.get("training_seed") == training_seed
                and record.get("checkpoint_episode") == checkpoint
            }
        )
        for training_seed in config.training_seeds
        for checkpoint in (0, 50)
    )
    consistent_signal = any(
        len(directions) == len(config.training_seeds)
        and directions[0] != 0
        and len(set(directions)) == 1
        for arm in config.arms
        if (
            directions := [
                cast(int, record["direction"])
                for record in learning_directions
                if record["arm_id"] == arm.id and cast(int, record["paired_count"]) > 0
            ]
        )
    )
    gate = classify_discovery_result(
        contract_errors=contract_errors,
        minimum_comparable=minimum_comparable,
        consistent_signal=consistent_signal,
    )
    next_step_decision = _derive_next_step_decision(gate, runs)
    return {
        "schema_version": "discovery-pilot-summary-v1",
        "protocol_revision": DISCOVERY_SCHEMA_VERSION,
        "diagnostic_only": True,
        "statistical_significance_claimed": False,
        "strategy_discovery_claimed": False,
        "training_seed_count": len(config.training_seeds),
        "gate": gate,
        "stop_reason": stopped.get("stop_reason", "contract_failed"),
        "shared_wall_seconds": config.shared_wall_seconds,
        "consumed_wall_seconds": float(stopped.get("consumed_wall_seconds", 0.0)),
        "measured_segment_wall_seconds": float(stopped.get("measured_segment_wall_seconds", 0.0)),
        "measured_finalization_wall_seconds": float(
            stopped.get("measured_finalization_wall_seconds", 0.0)
        ),
        "charged_final_write_reserve_seconds": float(
            stopped.get("charged_final_write_reserve_seconds", 0.0)
        ),
        "hard_deadline_overrun_seconds": float(stopped.get("hard_deadline_overrun_seconds", 0.0)),
        "planned_finalization_reserve_seconds": float(
            stopped.get(
                "planned_finalization_reserve_seconds",
                config.finalization_reserve_seconds,
            )
        ),
        "budget_accounting": stopped.get(
            "budget_accounting",
            "measured-work-plus-finalization-plus-final-write-reserve",
        ),
        "process_cpu_seconds": float(stopped.get("process_cpu_seconds", 0.0)),
        "minimum_comparable": minimum_comparable,
        "consistent_signal": consistent_signal,
        "contract_errors": list(contract_errors),
        "raw_record_counts": {
            "training_episodes": len(training_records),
            "training_metrics": len(training_metrics_records),
            "evaluation_episodes": len(evaluation_records),
            "checkpoints": len(checkpoint_records),
            "progress": len(progress_records),
        },
        "runs": runs,
        "paired_optimistic_minus_zero": paired_differences,
        "learning_directions": learning_directions,
        "next_step_decision": next_step_decision,
    }


def _checkpoint_parity_key(learner_config: Mapping[str, Any]) -> str:
    value = json.loads(canonical_json(learner_config))
    for name in (
        "optimistic_initialization",
        "optimistic_total_value",
        "initial_feature_value",
    ):
        value.pop(name, None)
    value_function = value.get("value_function")
    if isinstance(value_function, dict):
        for name in (
            "initial_value",
            "optimistic_total_value",
            "initial_feature_value",
        ):
            value_function.pop(name, None)
    return canonical_json(value)


def _milestone_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    result: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for record in records:
        if record.get("kind") != "milestone":
            continue
        arm_id = record.get("arm_id")
        training_seed = record.get("training_seed")
        checkpoint_episode = record.get("checkpoint_episode")
        if (
            isinstance(arm_id, str)
            and isinstance(training_seed, str)
            and type(checkpoint_episode) is int
        ):
            result[(arm_id, training_seed, checkpoint_episode)] = record
    return result


def _evaluation_keys(root: Path) -> set[tuple[str, str, int, int]]:
    keys: set[tuple[str, str, int, int]] = set()
    for path in sorted((root / "runs").glob("*/*/evaluation/*/episodes.jsonl")):
        _assert_artifact_path(root, path)
        for record in _read_jsonl(path):
            arm_id = record.get("arm_id")
            training_seed = record.get("training_seed")
            checkpoint_episode = record.get("checkpoint_episode")
            evaluation_episode_id = record.get("evaluation_episode_id")
            if (
                isinstance(arm_id, str)
                and isinstance(training_seed, str)
                and type(checkpoint_episode) is int
                and type(evaluation_episode_id) is int
            ):
                keys.add((arm_id, training_seed, checkpoint_episode, evaluation_episode_id))
    return keys


def _scheduler_cursor(
    config: DiscoveryPilotConfig,
    states: Sequence[_RunState],
    milestone_records: Mapping[tuple[str, str, int], Mapping[str, Any]],
    completed_evaluation: set[tuple[str, str, int, int]],
) -> dict[str, object]:
    """Return the next durable scheduler position for progress/resume audits."""

    for checkpoint_episode in config.checkpoint_episodes:
        if checkpoint_episode > 0:
            for state in states:
                if state.resume_snapshot is not None:
                    return {
                        "phase": "training",
                        "checkpoint_episode": checkpoint_episode,
                        "arm_id": state.arm.id,
                        "training_seed": state.training_seed,
                        "completed_training_episodes": state.completed_episodes,
                        "global_env_step": state.global_env_steps,
                    }
            incomplete_states = [
                state for state in states if state.completed_episodes < checkpoint_episode
            ]
            if incomplete_states:
                minimum_completed = min(state.completed_episodes for state in incomplete_states)
                current_chunk_target = min(
                    checkpoint_episode,
                    (minimum_completed // config.round_robin_training_chunk + 1)
                    * config.round_robin_training_chunk,
                )
                for state in states:
                    if state.completed_episodes >= current_chunk_target:
                        continue
                    return {
                        "phase": "training",
                        "checkpoint_episode": checkpoint_episode,
                        "arm_id": state.arm.id,
                        "training_seed": state.training_seed,
                        "completed_training_episodes": state.completed_episodes,
                        "global_env_step": state.global_env_steps,
                    }
        for state in states:
            if (state.arm.id, state.training_seed, checkpoint_episode) not in milestone_records:
                return {
                    "phase": "checkpoint",
                    "checkpoint_episode": checkpoint_episode,
                    "arm_id": state.arm.id,
                    "training_seed": state.training_seed,
                }
        for evaluation_episode_id in range(config.evaluation_episodes_per_checkpoint):
            for state in states:
                key = (state.arm.id, state.training_seed, checkpoint_episode, evaluation_episode_id)
                if key not in completed_evaluation:
                    return {
                        "phase": "evaluation",
                        "checkpoint_episode": checkpoint_episode,
                        "evaluation_episode_id": evaluation_episode_id,
                        "arm_id": state.arm.id,
                        "training_seed": state.training_seed,
                    }
    return {"phase": "complete"}


def verify_discovery_artifact(artifact_directory: str | Path) -> dict[str, Any]:
    """Read and recompute a Discovery artifact without writing or resuming it."""

    root = Path(artifact_directory)
    started = time.perf_counter()
    errors: list[str] = []
    config: DiscoveryPilotConfig | None = None
    try:
        config = resolve_discovery_config(_read_json(root / "resolved-config.json"))
    except Exception as error:  # The verifier reports all contract failures in-band.
        errors.append(f"resolved config: {error}")

    if config is not None:
        resolved = config.to_json()
        expected_config_hash = config_hash(resolved)
        required_control_files = (
            "resolved-config.json",
            "run-manifest.json",
            "knowledge-manifest.json",
            "progress.jsonl",
            "checkpoints.jsonl",
            "pilot-summary.json",
            "summary.json",
        )
        for name in required_control_files:
            path = root / name
            try:
                _assert_artifact_path(root, path)
            except ArtifactError as error:
                errors.append(f"invalid required artifact path {name}: {error}")
                continue
            if not path.is_file():
                errors.append(f"missing required artifact control file: {name}")
        run_manifest_stop_reason: str | None = None
        try:
            run_manifest = _read_json(root / "run-manifest.json")
            raw_stop_reason = run_manifest.get("stop_reason")
            if isinstance(raw_stop_reason, str):
                run_manifest_stop_reason = raw_stop_reason
            if run_manifest.get("config_hash") != expected_config_hash:
                errors.append("run manifest config hash mismatch")
            if run_manifest.get("stop_reason") not in {
                "completed",
                "budget_exhausted",
                "interrupted",
                "contract_failed",
            }:
                errors.append("run manifest stop reason is invalid")
        except Exception as error:
            errors.append(f"run manifest: {error}")

        manifest_paths = [root / "knowledge-manifest.json"]
        expected_run_manifests = [
            root / "runs" / arm.id / training_seed / "knowledge-manifest.json"
            for training_seed in config.training_seeds
            for arm in config.arms
        ]
        for manifest_path in expected_run_manifests:
            try:
                _assert_artifact_path(root, manifest_path)
            except ArtifactError as error:
                errors.append(f"invalid knowledge manifest path: {error}")
                continue
            if not manifest_path.is_file():
                errors.append(f"missing knowledge manifest: {manifest_path.relative_to(root)}")
        for manifest_path in sorted((root / "runs").glob("*/*/knowledge-manifest.json")):
            try:
                _assert_artifact_path(root, manifest_path)
            except ArtifactError as error:
                errors.append(f"invalid knowledge manifest path: {error}")
                continue
            manifest_paths.append(manifest_path)
        for manifest_path in manifest_paths:
            try:
                if manifest_path.is_symlink():
                    raise ArtifactError("knowledge manifest path is a symlink")
                raw_manifest = _read_json(manifest_path)
                manifest = KnowledgeManifest.from_json(raw_manifest)
                if manifest.experiment_kind != "discovery":
                    raise ArtifactError(
                        "Discovery artifact knowledge manifest must use experiment_kind=discovery"
                    )
                if manifest_path == root / "knowledge-manifest.json":
                    expected_manifest = _overall_manifest(config).to_json()
                else:
                    parts = manifest_path.relative_to(root).parts
                    if len(parts) != 4 or parts[-1] != "knowledge-manifest.json":
                        expected_manifest = None
                    else:
                        arm_id, training_seed = parts[1], parts[2]
                        if training_seed not in config.training_seeds:
                            expected_manifest = None
                        else:
                            arm = next((item for item in config.arms if item.id == arm_id), None)
                            expected_manifest = (
                                None
                                if arm is None
                                else _build_agent(config, arm).knowledge_manifest().to_json()
                            )
                if expected_manifest is not None and canonical_json(raw_manifest) != canonical_json(
                    expected_manifest
                ):
                    errors.append(
                        "knowledge manifest does not match resolved Discovery config: "
                        f"{manifest_path.relative_to(root)}"
                    )
            except Exception as error:
                errors.append(f"knowledge manifest {manifest_path.relative_to(root)}: {error}")

        checkpoint_records: list[dict[str, Any]] = []
        try:
            checkpoint_records = _read_jsonl(root / "checkpoints.jsonl")
        except Exception as error:
            errors.append(f"checkpoint index: {error}")
        parity: dict[tuple[str, int], dict[str, str]] = {}
        seen_milestones: list[int] = []
        milestone_keys: list[tuple[str, str, int]] = []
        seen_milestone_keys: set[tuple[str, str, int]] = set()
        try:
            latest_stop = _latest_stop_record(_read_jsonl(root / "progress.jsonl"))
            latest_cursor = latest_stop.get("scheduler_cursor")
            partial_artifact = latest_stop.get("stop_reason") in {
                "budget_exhausted",
                "interrupted",
            } or (
                latest_stop.get("stop_reason") == "contract_failed"
                and isinstance(latest_cursor, Mapping)
                and latest_cursor.get("phase") != "complete"
            )
        except Exception:
            partial_artifact = False
        for record in checkpoint_records:
            try:
                arm_id_value = record.get("arm_id")
                training_seed_value = record.get("training_seed")
                checkpoint_episode_value = record.get("checkpoint_episode")
                kind = record.get("kind")
                if not isinstance(arm_id_value, str) or arm_id_value not in DISCOVERY_ARM_IDS:
                    raise ArtifactError("checkpoint arm_id is unknown")
                if (
                    not isinstance(training_seed_value, str)
                    or training_seed_value not in config.training_seeds
                ):
                    raise ArtifactError("checkpoint training_seed is unknown")
                if type(checkpoint_episode_value) is not int or checkpoint_episode_value < 0:
                    raise ArtifactError("checkpoint episode is invalid")
                if kind not in {"milestone", "partial"}:
                    raise ArtifactError("checkpoint kind is invalid")
                arm_id = arm_id_value
                training_seed = training_seed_value
                checkpoint_episode = checkpoint_episode_value
                arm = next(item for item in config.arms if item.id == arm_id)
                agent = _build_agent(config, arm)
                step_value = record.get("global_env_step")
                if type(step_value) is not int or step_value < 0:
                    raise ArtifactError("checkpoint global_env_step is invalid")
                step = step_value
                directory = _path_in_artifact(root, cast(str, record["checkpoint_directory"]))
                array_path = _path_in_artifact(root, cast(str, record["array_path"]))
                metadata_path = _path_in_artifact(root, cast(str, record["metadata_path"]))
                if array_path != directory / f"{step}.npz":
                    raise ArtifactError("checkpoint array path does not match checkpoint step")
                if metadata_path != directory / f"{step}.json":
                    raise ArtifactError("checkpoint metadata path does not match checkpoint step")
                agent.restore_checkpoint(directory, step, config_hash=expected_config_hash)
                if agent.learner.state_hash() != record.get("learner_state_hash"):
                    errors.append(
                        f"checkpoint state hash mismatch: {arm_id}/{training_seed}/{step}"
                    )
                if agent.learner.table_hash() != record.get("table_hash"):
                    errors.append(
                        f"checkpoint table hash mismatch: {arm_id}/{training_seed}/{step}"
                    )
                if canonical_json(agent.counters.to_json()) != canonical_json(
                    record.get("counters")
                ):
                    errors.append(f"checkpoint counters mismatch: {arm_id}/{training_seed}/{step}")
                metadata = _read_json(_path_in_artifact(root, cast(str, record["metadata_path"])))
                if record.get("config_hash") != expected_config_hash:
                    errors.append(
                        f"checkpoint index config hash mismatch: {arm_id}/{training_seed}"
                    )
                if canonical_json(record.get("environment")) != canonical_json(
                    metadata.get("environment")
                ):
                    errors.append(
                        f"checkpoint environment mismatch: {arm_id}/{training_seed}/{step}"
                    )
                environment = cast(Mapping[str, Any], metadata["environment"])
                rng = cast(Mapping[str, Any], environment["rng"])
                lineage = cast(Mapping[str, Any], rng["lineage"])
                if lineage.get("root_seed") != training_seed:
                    errors.append(
                        f"checkpoint training seed lineage mismatch: {arm_id}/{training_seed}"
                    )
                if lineage.get("purpose") != "train-env":
                    errors.append(f"checkpoint training purpose mismatch: {arm_id}/{training_seed}")
                if lineage.get("environment_id") != f"{config.experiment_id}-training":
                    errors.append(
                        f"checkpoint training environment mismatch: {arm_id}/{training_seed}"
                    )
                parity.setdefault((training_seed, checkpoint_episode), {})[arm_id] = (
                    _checkpoint_parity_key(cast(Mapping[str, Any], metadata["learner_config"]))
                )
                if kind == "milestone":
                    milestone_key = (arm_id, training_seed, checkpoint_episode)
                    if milestone_key in seen_milestone_keys:
                        errors.append(f"duplicate milestone checkpoint: {milestone_key}")
                    else:
                        seen_milestone_keys.add(milestone_key)
                        milestone_keys.append(milestone_key)
                        seen_milestones.append(checkpoint_episode)
            except Exception as error:
                errors.append(f"checkpoint verification: {error}")
        for key, arms in parity.items():
            if set(arms) == set(DISCOVERY_ARM_IDS) and len(set(arms.values())) != 1:
                errors.append(f"zero/OI checkpoint configs differ beyond initialization: {key}")
        for pointer_path in sorted((root / "runs").glob("*/*/resume-checkpoint.json")):
            try:
                _assert_artifact_path(root, pointer_path)
                pointer = _read_json(pointer_path)
                pointer_arm_id = pointer.get("arm_id")
                pointer_training_seed = pointer.get("training_seed")
                arm = next(item for item in config.arms if item.id == pointer_arm_id)
                if pointer_training_seed not in config.training_seeds:
                    raise ArtifactError("resume checkpoint training seed is unknown")
                directory_value = pointer.get("checkpoint_directory")
                checkpoint_step = pointer.get("checkpoint_step", 0)
                if not isinstance(directory_value, str) or type(checkpoint_step) is not int:
                    raise ArtifactError("resume checkpoint pointer has invalid path or step")
                directory = _path_in_artifact(root, directory_value)
                array_value = pointer.get("array_path")
                metadata_value = pointer.get("metadata_path")
                if not isinstance(array_value, str) or not isinstance(metadata_value, str):
                    raise ArtifactError(
                        "resume checkpoint pointer is missing checkpoint pair paths"
                    )
                if _path_in_artifact(root, array_value) != directory / f"{checkpoint_step}.npz":
                    raise ArtifactError(
                        "resume checkpoint array path does not match checkpoint step"
                    )
                if _path_in_artifact(root, metadata_value) != directory / f"{checkpoint_step}.json":
                    raise ArtifactError(
                        "resume checkpoint metadata path does not match checkpoint step"
                    )
                agent = _build_agent(config, arm)
                snapshot = agent.restore_checkpoint(
                    directory,
                    checkpoint_step,
                    config_hash=expected_config_hash,
                )
                if pointer.get("config_hash") != expected_config_hash:
                    errors.append(f"resume checkpoint config hash mismatch: {pointer_path}")
                if pointer.get("learner_state_hash") != agent.learner.state_hash():
                    errors.append(f"resume checkpoint state hash mismatch: {pointer_path}")
                if pointer.get("table_hash") != agent.learner.table_hash():
                    errors.append(f"resume checkpoint table hash mismatch: {pointer_path}")
                if canonical_json(pointer.get("counters")) != canonical_json(
                    agent.counters.to_json()
                ):
                    errors.append(f"resume checkpoint counters mismatch: {pointer_path}")
                if canonical_json(pointer.get("environment")) != canonical_json(snapshot.to_json()):
                    errors.append(f"resume checkpoint environment mismatch: {pointer_path}")
                _validate_metrics_against_checkpoint(
                    pointer,
                    pointer,
                    require_durable_fields=True,
                )
                _resume_episode_boundary(pointer, snapshot)
            except Exception as error:
                errors.append(
                    f"resume checkpoint verification {pointer_path.relative_to(root)}: {error}"
                )
        if seen_milestones != sorted(seen_milestones):
            errors.append("milestone checkpoint order is not monotonic")
        if not partial_artifact:
            expected_milestone_keys = {
                (arm.id, training_seed, checkpoint)
                for checkpoint in config.checkpoint_episodes
                for training_seed in config.training_seeds
                for arm in config.arms
            }
            actual_milestone_keys = set(milestone_keys)
            if actual_milestone_keys != expected_milestone_keys:
                missing = sorted(expected_milestone_keys - actual_milestone_keys)
                extra = sorted(actual_milestone_keys - expected_milestone_keys)
                for checkpoint in config.checkpoint_episodes:
                    expected_for_checkpoint = {
                        key for key in expected_milestone_keys if key[2] == checkpoint
                    }
                    actual_for_checkpoint = {
                        key for key in actual_milestone_keys if key[2] == checkpoint
                    }
                    if actual_for_checkpoint != expected_for_checkpoint:
                        errors.append(
                            f"checkpoint {checkpoint} was not saved for the complete matrix"
                        )
                errors.append(
                    "completed artifact checkpoint matrix mismatch: "
                    f"missing={missing}, extra={extra}"
                )

        training_record_keys: list[tuple[str, str, int]] = []
        seen_training_record_keys: set[tuple[str, str, int]] = set()
        metrics_record_keys: list[tuple[str, str, int, str]] = []
        seen_metrics_record_keys: set[tuple[str, str, int, str]] = set()
        metrics_records_by_key: dict[tuple[str, str, int, str], Mapping[str, Any]] = {}
        for path in sorted((root / "runs").glob("*/*/training-episodes.jsonl")):
            try:
                relative = path.relative_to(root)
                _path_in_artifact(root, str(relative))
                parts = relative.parts
                if len(parts) != 4 or parts[0] != "runs" or parts[3] != "training-episodes.jsonl":
                    raise ArtifactError("training records path is outside a run")
                owner = (parts[1], parts[2])
                if owner[0] not in DISCOVERY_ARM_IDS or owner[1] not in config.training_seeds:
                    raise ArtifactError("training records path names an unknown run")
                for record in _read_jsonl(path):
                    training_record_arm_id = record.get("arm_id")
                    training_record_seed = record.get("training_seed")
                    training_episode_id = record.get("episode_id")
                    if (
                        not isinstance(training_record_arm_id, str)
                        or not isinstance(training_record_seed, str)
                        or type(training_episode_id) is not int
                        or (training_record_arm_id, training_record_seed) != owner
                    ):
                        raise ArtifactError("training record does not match its run path")
                    if training_episode_id < 0:
                        raise ArtifactError("training record episode_id is invalid")
                    training_record_key = (
                        training_record_arm_id,
                        training_record_seed,
                        training_episode_id,
                    )
                    if training_record_key in seen_training_record_keys:
                        errors.append(f"duplicate training record: {training_record_key}")
                    else:
                        seen_training_record_keys.add(training_record_key)
                        training_record_keys.append(training_record_key)
            except Exception as error:
                errors.append(f"training records {path.relative_to(root)}: {error}")
        for path in sorted((root / "runs").glob("*/*/metrics.jsonl")):
            try:
                relative = path.relative_to(root)
                _path_in_artifact(root, str(relative))
                parts = relative.parts
                if len(parts) != 4 or parts[0] != "runs" or parts[3] != "metrics.jsonl":
                    raise ArtifactError("metrics path is outside a run")
                owner = (parts[1], parts[2])
                if owner[0] not in DISCOVERY_ARM_IDS or owner[1] not in config.training_seeds:
                    raise ArtifactError("metrics path names an unknown run")
                for record in _read_jsonl(path):
                    metrics_record_arm_id = record.get("arm_id")
                    metrics_record_seed = record.get("training_seed")
                    metrics_checkpoint_episode = record.get("checkpoint_episode")
                    metrics_kind = record.get("kind")
                    if (
                        not isinstance(metrics_record_arm_id, str)
                        or not isinstance(metrics_record_seed, str)
                        or (metrics_record_arm_id, metrics_record_seed) != owner
                    ):
                        raise ArtifactError("metrics record does not match its run path")
                    if (
                        type(metrics_checkpoint_episode) is not int
                        or metrics_checkpoint_episode < 0
                    ):
                        raise ArtifactError("metrics checkpoint_episode is invalid")
                    if metrics_kind not in {"milestone", "partial"}:
                        raise ArtifactError("metrics kind is invalid")
                    metrics_record_key = (
                        metrics_record_arm_id,
                        metrics_record_seed,
                        metrics_checkpoint_episode,
                        cast(str, metrics_kind),
                    )
                    if metrics_record_key in seen_metrics_record_keys:
                        errors.append(f"duplicate metrics record: {metrics_record_key}")
                    else:
                        seen_metrics_record_keys.add(metrics_record_key)
                        metrics_record_keys.append(metrics_record_key)
                        metrics_records_by_key[metrics_record_key] = record
            except Exception as error:
                errors.append(f"metrics {path.relative_to(root)}: {error}")
        checkpoint_metric_keys = {
            (
                record.get("arm_id"),
                record.get("training_seed"),
                record.get("checkpoint_episode"),
                record.get("kind"),
            )
            for record in checkpoint_records
            if isinstance(record.get("arm_id"), str)
            and isinstance(record.get("training_seed"), str)
            and type(record.get("checkpoint_episode")) is int
            and record.get("kind") in {"milestone", "partial"}
        }
        if checkpoint_metric_keys != set(metrics_record_keys):
            errors.append(
                "checkpoint and training metrics records do not match: "
                f"missing={sorted(checkpoint_metric_keys - set(metrics_record_keys))}, "
                f"extra={sorted(set(metrics_record_keys) - checkpoint_metric_keys)}"
            )
        checkpoints_by_metric_key = {
            (
                cast(str, record["arm_id"]),
                cast(str, record["training_seed"]),
                cast(int, record["checkpoint_episode"]),
                cast(str, record["kind"]),
            ): record
            for record in checkpoint_records
            if isinstance(record.get("arm_id"), str)
            and isinstance(record.get("training_seed"), str)
            and type(record.get("checkpoint_episode")) is int
            and record.get("kind") in {"milestone", "partial"}
        }
        for metrics_key, metrics_record in metrics_records_by_key.items():
            checkpoint_record = checkpoints_by_metric_key.get(metrics_key)
            if checkpoint_record is None:
                continue
            try:
                _validate_metrics_against_checkpoint(metrics_record, checkpoint_record)
            except ArtifactError as error:
                errors.append(f"training metrics do not match checkpoint {metrics_key}: {error}")

        evaluation_records: list[dict[str, Any]] = []
        for path in sorted((root / "runs").glob("*/*/evaluation/*/episodes.jsonl")):
            try:
                _assert_artifact_path(root, path)
                evaluation_records.extend(_read_jsonl(path))
            except Exception as error:
                errors.append(f"evaluation records {path.relative_to(root)}: {error}")
        seen_eval_keys: set[tuple[str, str, int, int]] = set()
        milestone_index = _milestone_records(checkpoint_records)
        for record in evaluation_records:
            eval_arm_id = record.get("arm_id")
            eval_training_seed = record.get("training_seed")
            eval_checkpoint_episode = record.get("checkpoint_episode")
            eval_episode_id = record.get("evaluation_episode_id")
            if (
                not isinstance(eval_arm_id, str)
                or eval_arm_id not in DISCOVERY_ARM_IDS
                or not isinstance(eval_training_seed, str)
                or eval_training_seed not in config.training_seeds
                or type(eval_checkpoint_episode) is not int
                or eval_checkpoint_episode not in config.checkpoint_episodes
                or type(eval_episode_id) is not int
                or not 0 <= eval_episode_id < config.evaluation_episodes_per_checkpoint
            ):
                errors.append("evaluation record has an invalid run/checkpoint/episode key")
                continue
            eval_key = (
                eval_arm_id,
                eval_training_seed,
                eval_checkpoint_episode,
                eval_episode_id,
            )
            if eval_key in seen_eval_keys:
                errors.append(f"duplicate evaluation record: {eval_key}")
            seen_eval_keys.add(eval_key)
            milestone_checkpoint = milestone_index.get(
                (eval_arm_id, eval_training_seed, eval_checkpoint_episode)
            )
            if milestone_checkpoint is None:
                errors.append(f"evaluation references missing milestone checkpoint: {eval_key}")
            else:
                checkpoint_step = record.get("checkpoint_global_env_step")
                if type(checkpoint_step) is not int or checkpoint_step != milestone_checkpoint.get(
                    "global_env_step"
                ):
                    errors.append(f"evaluation checkpoint step mismatch: {eval_key}")
            if record.get("evaluation_root_seed") != config.evaluation_root_seed:
                errors.append(f"evaluation seed mismatch: {eval_key}")
            if record.get("evaluation_root_seed") in config.training_seeds:
                errors.append(f"training/evaluation seed collision: {eval_key}")
            if record.get("purpose") != "discovery-eval":
                errors.append(f"evaluation purpose mismatch: {eval_key}")
            if record.get("frozen_state_unchanged") is not True:
                errors.append(f"frozen evaluation mutated state: {eval_key}")
            if record.get("clone_state_hash_before") != record.get("clone_state_hash_after"):
                errors.append(f"frozen state hash changed: {eval_key}")
            if record.get("clone_table_hash_before") != record.get("clone_table_hash_after"):
                errors.append(f"frozen table hash changed: {eval_key}")
            if canonical_json(record.get("clone_counters_before")) != canonical_json(
                record.get("clone_counters_after")
            ):
                errors.append(f"frozen counters changed: {eval_key}")
            if (
                record.get("training_state_hash_before") != record.get("training_state_hash_after")
                or record.get("training_counters_unchanged") is not True
            ):
                errors.append(f"training state changed during evaluation: {eval_key}")

        try:
            progress_records = _read_jsonl(root / "progress.jsonl")
            started_records = [
                record for record in progress_records if record.get("event") == "pilot_started"
            ]
            stopped_records = [
                record for record in progress_records if record.get("event") == "pilot_stopped"
            ]
            if len(started_records) != 1 or not stopped_records:
                errors.append("progress must contain one pilot start and at least one stop record")
            elif stopped_records:
                stopped = stopped_records[-1]
                if run_manifest_stop_reason is not None and run_manifest_stop_reason != stopped.get(
                    "stop_reason"
                ):
                    errors.append("run manifest stop reason does not match progress stop reason")
                raw_elapsed = stopped.get("consumed_wall_seconds")
                if isinstance(raw_elapsed, bool) or not isinstance(raw_elapsed, (int, float)):
                    errors.append("pilot stop consumed wall time is not numeric")
                    elapsed = 0.0
                else:
                    elapsed = float(raw_elapsed)
                    if not math.isfinite(elapsed) or elapsed < 0.0:
                        errors.append("pilot stop consumed wall time is invalid")
                raw_process_cpu = stopped.get("process_cpu_seconds", 0.0)
                if isinstance(raw_process_cpu, bool) or not isinstance(
                    raw_process_cpu, (int, float)
                ):
                    errors.append("pilot stop process CPU time is not numeric")
                elif not math.isfinite(float(raw_process_cpu)) or float(raw_process_cpu) < 0.0:
                    errors.append("pilot stop process CPU time is invalid")
                stop_reason = stopped.get("stop_reason")
                if stop_reason not in {
                    "completed",
                    "budget_exhausted",
                    "interrupted",
                    "contract_failed",
                }:
                    errors.append("pilot stop reason is invalid")
                if stop_reason == "completed" and elapsed > config.shared_wall_seconds:
                    errors.append("completed pilot exceeded the shared wall deadline")
                raw_deadline_reached = stopped.get("deadline_reached")
                if type(raw_deadline_reached) is not bool:
                    errors.append("pilot stop deadline_reached must be a boolean")
                elif raw_deadline_reached != (elapsed >= config.shared_wall_seconds):
                    errors.append("pilot stop deadline state does not match consumed wall time")
                raw_budget = stopped.get("shared_wall_seconds")
                if type(raw_budget) is not int or raw_budget != config.shared_wall_seconds:
                    errors.append("pilot stop shared wall budget does not match config")
                raw_prior = stopped.get("prior_consumed_wall_seconds")
                raw_segment = stopped.get("measured_segment_wall_seconds")
                raw_finalization = stopped.get("measured_finalization_wall_seconds")
                raw_final_write_reserve = stopped.get("charged_final_write_reserve_seconds")
                raw_overrun = stopped.get("hard_deadline_overrun_seconds")
                raw_planned_reserve = stopped.get("planned_finalization_reserve_seconds")
                accounting_values = (
                    raw_prior,
                    raw_segment,
                    raw_finalization,
                    raw_final_write_reserve,
                    raw_overrun,
                    raw_planned_reserve,
                )
                if any(
                    isinstance(value, bool) or not isinstance(value, (int, float))
                    for value in accounting_values
                ):
                    errors.append("pilot stop budget accounting fields must be numeric")
                else:
                    prior = float(cast(float | int, raw_prior))
                    segment = float(cast(float | int, raw_segment))
                    finalization = float(cast(float | int, raw_finalization))
                    final_write_reserve = float(cast(float | int, raw_final_write_reserve))
                    overrun = float(cast(float | int, raw_overrun))
                    planned_reserve = float(cast(float | int, raw_planned_reserve))
                    if any(
                        not math.isfinite(value) or value < 0.0
                        for value in (
                            prior,
                            segment,
                            finalization,
                            final_write_reserve,
                            overrun,
                            planned_reserve,
                        )
                    ):
                        errors.append("pilot stop budget accounting fields are invalid")
                    if planned_reserve > config.finalization_reserve_seconds:
                        errors.append("pilot stop planned finalization reserve exceeds config")
                    if final_write_reserve > DISCOVERY_FINAL_WRITE_RESERVE_SECONDS:
                        errors.append("pilot stop final-write reserve exceeds contract")
                    expected_elapsed = prior + segment + finalization + final_write_reserve
                    if not math.isclose(elapsed, expected_elapsed, abs_tol=1e-9):
                        errors.append("pilot stop consumed wall time does not match accounting")
                    expected_overrun = max(0.0, expected_elapsed - config.shared_wall_seconds)
                    if not math.isclose(overrun, expected_overrun, abs_tol=1e-9):
                        errors.append("pilot stop finalization overrun does not match accounting")
                    if stop_reason != "contract_failed" and elapsed > config.shared_wall_seconds:
                        errors.append("pilot stop exceeded the shared wall deadline")
                if (
                    stopped.get("budget_accounting")
                    != "measured-work-plus-finalization-plus-final-write-reserve"
                ):
                    errors.append("pilot stop budget accounting method is invalid")
                cursor = stopped.get("scheduler_cursor")
                if not isinstance(cursor, Mapping):
                    errors.append("pilot stop scheduler cursor is missing or malformed")
                else:
                    phase = cursor.get("phase")
                    if phase not in {"training", "checkpoint", "evaluation", "complete"}:
                        errors.append("pilot stop scheduler cursor phase is invalid")
                    elif phase == "training":
                        if cursor.get("checkpoint_episode") not in config.checkpoint_episodes[1:]:
                            errors.append("training scheduler cursor checkpoint is invalid")
                        if cursor.get("arm_id") not in DISCOVERY_ARM_IDS:
                            errors.append("training scheduler cursor arm is invalid")
                        if cursor.get("training_seed") not in config.training_seeds:
                            errors.append("training scheduler cursor seed is invalid")
                    elif phase == "checkpoint":
                        if cursor.get("checkpoint_episode") not in config.checkpoint_episodes:
                            errors.append("checkpoint scheduler cursor checkpoint is invalid")
                        if cursor.get("arm_id") not in DISCOVERY_ARM_IDS:
                            errors.append("checkpoint scheduler cursor arm is invalid")
                        if cursor.get("training_seed") not in config.training_seeds:
                            errors.append("checkpoint scheduler cursor seed is invalid")
                    elif phase == "evaluation":
                        if cursor.get("checkpoint_episode") not in config.checkpoint_episodes:
                            errors.append("evaluation scheduler cursor checkpoint is invalid")
                        if not (
                            type(cursor.get("evaluation_episode_id")) is int
                            and 0
                            <= cursor["evaluation_episode_id"]
                            < config.evaluation_episodes_per_checkpoint
                        ):
                            errors.append("evaluation scheduler cursor episode is invalid")
                        if cursor.get("arm_id") not in DISCOVERY_ARM_IDS:
                            errors.append("evaluation scheduler cursor arm is invalid")
                        if cursor.get("training_seed") not in config.training_seeds:
                            errors.append("evaluation scheduler cursor seed is invalid")

            run_order = [
                (arm.id, training_seed)
                for training_seed in config.training_seeds
                for arm in config.arms
            ]
            expected_checkpoints = [
                (checkpoint, arm_id, training_seed)
                for checkpoint in config.checkpoint_episodes
                for arm_id, training_seed in run_order
            ]
            actual_checkpoints = [
                (
                    cast(int, record["checkpoint_episode"]),
                    cast(str, record["arm_id"]),
                    cast(str, record["training_seed"]),
                )
                for record in progress_records
                if record.get("event") == "checkpoint_saved" and record.get("kind") == "milestone"
            ]
            if actual_checkpoints != expected_checkpoints[: len(actual_checkpoints)]:
                errors.append("milestone checkpoints are not a fair matrix-order prefix")

            expected_evaluation = [
                (checkpoint, episode_id, arm_id, training_seed)
                for checkpoint in config.checkpoint_episodes
                for episode_id in range(config.evaluation_episodes_per_checkpoint)
                for arm_id, training_seed in run_order
            ]
            actual_evaluation = [
                (
                    cast(int, record["checkpoint_episode"]),
                    cast(int, record["evaluation_episode_id"]),
                    cast(str, record["arm_id"]),
                    cast(str, record["training_seed"]),
                )
                for record in progress_records
                if record.get("event") == "evaluation_episode_completed"
            ]
            if actual_evaluation != expected_evaluation[: len(actual_evaluation)]:
                errors.append("evaluation records are not an episode-first round-robin prefix")
            progress_evaluation_keys = {
                (arm_id, training_seed, checkpoint, episode_id)
                for checkpoint, episode_id, arm_id, training_seed in actual_evaluation
            }
            if progress_evaluation_keys != seen_eval_keys:
                errors.append("evaluation records do not match completed evaluation progress")
            if stopped_records and stopped_records[-1].get("stop_reason") == "completed":
                expected_complete_evaluation = {
                    (arm_id, training_seed, checkpoint, episode_id)
                    for checkpoint in config.checkpoint_episodes
                    for episode_id in range(config.evaluation_episodes_per_checkpoint)
                    for arm_id, training_seed in run_order
                }
                if seen_eval_keys != expected_complete_evaluation:
                    errors.append("completed pilot is missing evaluation matrix records")

            expected_training: list[tuple[str, str, int]] = []
            completed = {key: 0 for key in run_order}
            for milestone in config.checkpoint_episodes[1:]:
                while any(value < milestone for value in completed.values()):
                    for run_key in run_order:
                        target = min(
                            milestone,
                            completed[run_key] + config.round_robin_training_chunk,
                        )
                        while completed[run_key] < target:
                            expected_training.append((*run_key, completed[run_key]))
                            completed[run_key] += 1
            actual_training = [
                (
                    cast(str, record["arm_id"]),
                    cast(str, record["training_seed"]),
                    cast(int, record["episode_id"]),
                )
                for record in progress_records
                if record.get("event") == "training_episode_completed"
            ]
            if actual_training != expected_training[: len(actual_training)]:
                errors.append("training records are not a chunked round-robin prefix")
            if set(training_record_keys) != set(actual_training):
                errors.append("training JSONL records do not match completed training progress")
            if (
                stopped_records
                and stopped_records[-1].get("stop_reason") == "completed"
                and (actual_training != expected_training)
            ):
                errors.append("completed pilot is missing training episode records")
        except Exception as error:
            errors.append(f"progress schedule: {error}")

        try:
            stored_summary = _read_json(root / "pilot-summary.json")
            declared_errors = cast(list[str], stored_summary.get("contract_errors", []))
            expected_summary = recompute_discovery_summary(root, contract_errors=declared_errors)
            if canonical_json(stored_summary) != canonical_json(expected_summary):
                errors.append("pilot summary is not recomputable from raw records")
            stored_generic_summary = _read_json(root / "summary.json")
            expected_generic_summary = {"schema_version": "summary-v1", **expected_summary}
            if canonical_json(stored_generic_summary) != canonical_json(expected_generic_summary):
                errors.append("summary.json is not synchronized with pilot-summary.json")
        except Exception as error:
            stored_summary = {}
            errors.append(f"pilot summary: {error}")
    else:
        stored_summary = {}

    return {
        "schema_version": "discovery-verification-v1",
        "valid": not errors,
        "gate": stored_summary.get("gate", "contract-failed") if not errors else "contract-failed",
        "errors": errors,
        "verification_wall_seconds": time.perf_counter() - started,
        "read_only": True,
    }


def run_discovery_pilot(
    config: DiscoveryPilotConfig | Mapping[str, Any] | str | Path,
    *,
    artifact_directory: str | Path | None = None,
    resume_from: str | Path | None = None,
    resume: bool = False,
    clock: Clock = time.monotonic,
    process_clock: Clock = time.process_time,
    phase_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Run or explicitly resume the four-run v1 matrix.

    A resume is never implicit: callers must pass ``resume_from`` (or
    ``resume=True`` together with ``artifact_directory``).  The existing
    artifact's recorded wall time is subtracted from the fixed 900-second
    budget before any new training or evaluation can happen.
    """

    if isinstance(config, (str, Path)):
        resolved = load_discovery_config(config)
    elif isinstance(config, DiscoveryPilotConfig):
        resolved = resolve_discovery_config(config.to_json())
    else:
        resolved = resolve_discovery_config(config)
    resolved_json = resolved.to_json()
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
    prior_consumed_wall_seconds = 0.0
    prior_process_cpu_seconds = 0.0
    remaining_wall_seconds: float | None = None
    if resume_from is not None:
        store, prior_consumed_wall_seconds, prior_process_cpu_seconds, progress_records = (
            _resume_metadata(root, resolved)
        )
        started = clock()
        process_started = process_clock()
        remaining_wall_seconds = max(
            0.0, resolved.shared_wall_seconds - prior_consumed_wall_seconds
        )
        hard_deadline = started + remaining_wall_seconds
        reserved_finalization_seconds = min(
            resolved.finalization_reserve_seconds, remaining_wall_seconds
        )
        deadline = hard_deadline - reserved_finalization_seconds
        states = _restore_run_states(store, resolved, progress_records)
        checkpoint_records = _read_jsonl(root / "checkpoints.jsonl")
        milestone_records = _milestone_records(checkpoint_records)
        completed_evaluation = _evaluation_keys(root)
        derived_resume_cursor = _scheduler_cursor(
            resolved, states, milestone_records, completed_evaluation
        )
        saved_resume_cursor = _latest_stop_record(progress_records).get("scheduler_cursor")
        if saved_resume_cursor is not None and canonical_json(
            saved_resume_cursor
        ) != canonical_json(derived_resume_cursor):
            raise ArtifactError("resume scheduler cursor does not match durable run progress")
        _append_progress(
            store,
            "pilot_resumed",
            prior_consumed_wall_seconds=prior_consumed_wall_seconds,
            remaining_wall_seconds=remaining_wall_seconds,
            scheduler_cursor=derived_resume_cursor,
        )
    else:
        started = clock()
        process_started = process_clock()
        hard_deadline = started + resolved.shared_wall_seconds
        reserved_finalization_seconds = resolved.finalization_reserve_seconds
        deadline = hard_deadline - reserved_finalization_seconds
        store = ArtifactStore(root, resolved_json, repo_root=REPOSITORY_ROOT)
        store.initialize(
            knowledge_manifest=_overall_manifest(resolved),
            seed=canonical_json(
                {
                    "training": list(resolved.training_seeds),
                    "evaluation": resolved.evaluation_root_seed,
                }
            ),
            budget={
                "shared_wall_seconds": resolved.shared_wall_seconds,
                "finalization_reserve_seconds": resolved.finalization_reserve_seconds,
                "max_training_episodes_per_run": resolved.max_training_episodes_per_run,
                "evaluation_episodes_per_checkpoint": resolved.evaluation_episodes_per_checkpoint,
            },
        )
        _append_progress(
            store,
            "pilot_started",
            shared_wall_seconds=resolved.shared_wall_seconds,
            finalization_reserve_seconds=resolved.finalization_reserve_seconds,
            checkpoint_episodes=list(resolved.checkpoint_episodes),
            run_count=len(resolved.arms) * len(resolved.training_seeds),
        )
        states = []
        for training_seed in resolved.training_seeds:
            for arm in resolved.arms:
                state = _RunState(
                    arm=arm,
                    training_seed=training_seed,
                    agent=_build_agent(resolved, arm),
                    relative_root=f"runs/{arm.id}/{training_seed}",
                )
                state.last_snapshot = _initial_snapshot(resolved, training_seed)
                _write_run_manifest(store, state)
                states.append(state)

    checkpoint_records = _read_jsonl(root / "checkpoints.jsonl")
    milestone_records = _milestone_records(checkpoint_records)
    completed_evaluation = _evaluation_keys(root)
    resume_cursor = (
        _scheduler_cursor(resolved, states, milestone_records, completed_evaluation)
        if resume_from is not None
        else None
    )
    stop_reason = (
        "budget_exhausted"
        if resume_from is not None
        and remaining_wall_seconds is not None
        and remaining_wall_seconds <= reserved_finalization_seconds
        else "completed"
    )
    contract_errors: list[str] = []
    interrupts = _InterruptController()
    interrupts.install()
    try:
        for checkpoint_episode in resolved.checkpoint_episodes:
            if checkpoint_episode > 0:
                if (
                    resume_cursor is not None
                    and resume_cursor.get("phase") == "training"
                    and resume_cursor.get("checkpoint_episode") == checkpoint_episode
                ):
                    cursor_key = (
                        resume_cursor.get("arm_id"),
                        resume_cursor.get("training_seed"),
                    )
                    cursor_index = next(
                        (index for index, state in enumerate(states) if state.key == cursor_key),
                        None,
                    )
                    if cursor_index is None:
                        raise ArtifactError("resume scheduler cursor names an unknown run")
                    for state in states[cursor_index:]:
                        target = min(
                            checkpoint_episode,
                            ((state.completed_episodes // resolved.round_robin_training_chunk) + 1)
                            * resolved.round_robin_training_chunk,
                        )
                        while (
                            state.completed_episodes < target or state.resume_snapshot is not None
                        ):
                            if not _train_one_episode(
                                store,
                                resolved,
                                state,
                                clock=clock,
                                deadline=deadline,
                                process_clock=process_clock,
                                interrupts=interrupts,
                                phase_hook=phase_hook,
                            ):
                                stop_reason = "budget_exhausted"
                                break
                        if stop_reason != "completed":
                            break
                        if state.completed_episodes == target:
                            _save_resume_checkpoint(store, state)
                            if interrupts.requested:
                                _persist_interrupt_boundaries(
                                    store, states, phase="resume_checkpoint_boundary"
                                )
                                raise _DiscoveryInterrupted
                            if _deadline_reached(clock, hard_deadline):
                                stop_reason = "budget_exhausted"
                                _append_progress(
                                    store,
                                    "budget_exhausted",
                                    phase="resume_checkpoint",
                                    arm_id=state.arm.id,
                                    training_seed=state.training_seed,
                                )
                                break
                    resume_cursor = None
                if stop_reason != "completed":
                    break
                while any(
                    state.completed_episodes < checkpoint_episode
                    or state.resume_snapshot is not None
                    for state in states
                ):
                    for state in states:
                        target = min(
                            checkpoint_episode,
                            ((state.completed_episodes // resolved.round_robin_training_chunk) + 1)
                            * resolved.round_robin_training_chunk,
                        )
                        while (
                            state.completed_episodes < target or state.resume_snapshot is not None
                        ):
                            if not _train_one_episode(
                                store,
                                resolved,
                                state,
                                clock=clock,
                                deadline=deadline,
                                process_clock=process_clock,
                                interrupts=interrupts,
                                phase_hook=phase_hook,
                            ):
                                stop_reason = "budget_exhausted"
                                break
                        if stop_reason == "completed" and state.completed_episodes == target:
                            _save_resume_checkpoint(store, state)
                            if interrupts.requested:
                                _persist_interrupt_boundaries(
                                    store, states, phase="resume_checkpoint_boundary"
                                )
                                raise _DiscoveryInterrupted
                            if _deadline_reached(clock, hard_deadline):
                                stop_reason = "budget_exhausted"
                                _append_progress(
                                    store,
                                    "budget_exhausted",
                                    phase="resume_checkpoint",
                                    arm_id=state.arm.id,
                                    training_seed=state.training_seed,
                                )
                        if stop_reason != "completed":
                            break
                    if stop_reason != "completed":
                        break
            if stop_reason != "completed":
                break
            checkpoints: dict[tuple[str, str], Mapping[str, object]] = {}
            for state in states:
                existing = milestone_records.get(
                    (state.arm.id, state.training_seed, checkpoint_episode)
                )
                if existing is not None:
                    checkpoints[state.key] = existing
                    continue
                if (
                    state.completed_episodes != checkpoint_episode
                    or state.resume_snapshot is not None
                ):
                    raise ArtifactError("scheduler reached a checkpoint with incomplete run state")
                if _deadline_reached(clock, hard_deadline):
                    stop_reason = "budget_exhausted"
                    _append_progress(
                        store,
                        "budget_exhausted",
                        phase="checkpoint",
                        checkpoint_episode=checkpoint_episode,
                        arm_id=state.arm.id,
                        training_seed=state.training_seed,
                        scheduler_cursor=_scheduler_cursor(
                            resolved, states, milestone_records, completed_evaluation
                        ),
                    )
                    break
                if phase_hook is not None:
                    phase_hook("checkpoint")
                record = _save_checkpoint(
                    store,
                    state,
                    checkpoint_episode=checkpoint_episode,
                    kind="milestone",
                )
                milestone_records[(state.arm.id, state.training_seed, checkpoint_episode)] = record
                checkpoints[state.key] = record
                if interrupts.requested:
                    _persist_interrupt_boundaries(store, states, phase="checkpoint_boundary")
                    raise _DiscoveryInterrupted
                if _deadline_reached(clock, hard_deadline):
                    stop_reason = "budget_exhausted"
                    _append_progress(
                        store,
                        "budget_exhausted",
                        phase="checkpoint_durability",
                        checkpoint_episode=checkpoint_episode,
                        arm_id=state.arm.id,
                        training_seed=state.training_seed,
                    )
                    break
            if stop_reason != "completed":
                break
            if not _evaluate_checkpoint_round_robin(
                store,
                resolved,
                states,
                checkpoints,
                checkpoint_episode=checkpoint_episode,
                clock=clock,
                deadline=deadline,
                process_clock=process_clock,
                interrupts=interrupts,
                phase_hook=phase_hook,
                existing_evaluation_keys=completed_evaluation,
            ):
                stop_reason = "budget_exhausted"
                break
            if interrupts.requested:
                _persist_interrupt_boundaries(store, states, phase="evaluation_boundary")
                raise _DiscoveryInterrupted
    except _DiscoveryInterrupted:
        stop_reason = "interrupted"
    except KeyboardInterrupt as error:
        stop_reason = "contract_failed"
        contract_errors.append("unsafe KeyboardInterrupt bypassed the cooperative SIGINT boundary")
        store.failure(error)
    except Exception as error:
        stop_reason = "contract_failed"
        contract_errors.append(f"{type(error).__name__}: {error}")
        store.failure(error)

    try:
        stopped_at = clock()
        if stop_reason == "completed" and stopped_at >= deadline:
            stop_reason = "budget_exhausted"
            _append_progress(store, "budget_exhausted", phase="finalization")
        measured_segment_wall_seconds = max(0.0, stopped_at - started)
        process_cpu_seconds = prior_process_cpu_seconds + max(
            0.0, process_clock() - process_started
        )
        final_scheduler_cursor = _scheduler_cursor(
            resolved,
            states,
            _milestone_records(_read_jsonl(root / "checkpoints.jsonl")),
            _evaluation_keys(root),
        )
        budget_accounting = "measured-work-plus-finalization-plus-final-write-reserve"

        # The data deadline leaves room for durable finalization.  We measure
        # the expensive summary/finalize pass, then conservatively charge one
        # small reserve for the terminal progress record and its mirrored
        # atomic JSON writes.  Nothing is clipped to the configured budget.
        finalization_started = clock()
        provisional_stop: dict[str, object] = {
            "schema_version": "discovery-progress-v1",
            "event": "pilot_stopped",
            "stop_reason": stop_reason,
            "consumed_wall_seconds": (prior_consumed_wall_seconds + measured_segment_wall_seconds),
            "measured_segment_wall_seconds": measured_segment_wall_seconds,
            "measured_finalization_wall_seconds": 0.0,
            "charged_final_write_reserve_seconds": 0.0,
            "hard_deadline_overrun_seconds": 0.0,
            "prior_consumed_wall_seconds": prior_consumed_wall_seconds,
            "planned_finalization_reserve_seconds": reserved_finalization_seconds,
            "budget_accounting": budget_accounting,
            "process_cpu_seconds": process_cpu_seconds,
            "shared_wall_seconds": resolved.shared_wall_seconds,
            "deadline_reached": (
                prior_consumed_wall_seconds + measured_segment_wall_seconds
                >= resolved.shared_wall_seconds
            ),
            "scheduler_cursor": final_scheduler_cursor,
        }
        try:
            if phase_hook is not None:
                phase_hook("finalization")
            provisional_summary = recompute_discovery_summary(
                store.root,
                contract_errors=contract_errors,
                stopped_record=provisional_stop,
            )
            if phase_hook is not None:
                phase_hook("finalization_summary")
            store.write_json("pilot-summary.json", provisional_summary)
            store.finalize(stop_reason=stop_reason, summary=provisional_summary)
        except KeyboardInterrupt as error:
            stop_reason = "contract_failed"
            contract_errors.append(
                "unsafe KeyboardInterrupt bypassed the cooperative SIGINT boundary"
            )
            store.failure(error)
        except Exception as error:
            stop_reason = "contract_failed"
            contract_errors.append(f"{type(error).__name__}: {error}")
            store.failure(error)

        measured_finalization_wall_seconds = max(0.0, clock() - finalization_started)
        charged_final_write_reserve_seconds = min(
            DISCOVERY_FINAL_WRITE_RESERVE_SECONDS,
            reserved_finalization_seconds,
        )
        consumed = (
            prior_consumed_wall_seconds
            + measured_segment_wall_seconds
            + measured_finalization_wall_seconds
            + charged_final_write_reserve_seconds
        )
        hard_deadline_overrun_seconds = max(0.0, consumed - float(resolved.shared_wall_seconds))
        if hard_deadline_overrun_seconds > 0.0 and stop_reason != "contract_failed":
            stop_reason = "contract_failed"
            contract_errors.append(
                "shared wall deadline overrun during finalization: "
                f"{hard_deadline_overrun_seconds:.6f}s"
            )
        elif interrupts.requested and stop_reason != "contract_failed":
            stop_reason = "interrupted"

        terminal_stop: dict[str, object] = {
            "schema_version": "discovery-progress-v1",
            "event": "pilot_stopped",
            "stop_reason": stop_reason,
            "consumed_wall_seconds": consumed,
            "measured_segment_wall_seconds": measured_segment_wall_seconds,
            "measured_finalization_wall_seconds": measured_finalization_wall_seconds,
            "charged_final_write_reserve_seconds": charged_final_write_reserve_seconds,
            "hard_deadline_overrun_seconds": hard_deadline_overrun_seconds,
            "prior_consumed_wall_seconds": prior_consumed_wall_seconds,
            "planned_finalization_reserve_seconds": reserved_finalization_seconds,
            "budget_accounting": budget_accounting,
            "process_cpu_seconds": process_cpu_seconds,
            "shared_wall_seconds": resolved.shared_wall_seconds,
            "deadline_reached": consumed >= resolved.shared_wall_seconds,
            "scheduler_cursor": final_scheduler_cursor,
        }

        def write_terminal_artifacts(errors: Sequence[str]) -> dict[str, Any]:
            _append_progress(
                store,
                "pilot_stopped",
                **{
                    key: value
                    for key, value in terminal_stop.items()
                    if key not in {"schema_version", "event"}
                },
            )
            terminal_summary = recompute_discovery_summary(store.root, contract_errors=errors)
            store.write_json("pilot-summary.json", terminal_summary)
            store.finalize(
                stop_reason=cast(str, terminal_stop["stop_reason"]),
                summary=terminal_summary,
            )
            return terminal_summary

        final_write_started = clock()
        try:
            if phase_hook is not None:
                phase_hook("finalization_progress")
            # The cooperative handler may run while the terminal progress
            # record is being prepared.  Preserve that request in the record
            # that is about to become the durable stop boundary; otherwise a
            # SIGINT arriving after the measured finalization phase would be
            # silently reported as ``completed``.
            if interrupts.requested and stop_reason != "contract_failed":
                stop_reason = "interrupted"
                terminal_stop["stop_reason"] = stop_reason
            if interrupts.requested and stop_reason == "interrupted":
                _append_progress(store, "interrupted", phase="finalization_boundary")
            summary = write_terminal_artifacts(contract_errors)
        except KeyboardInterrupt as error:
            stop_reason = "contract_failed"
            contract_errors.append(
                "unsafe KeyboardInterrupt bypassed the cooperative SIGINT boundary"
            )
            store.failure(error)
            terminal_stop["stop_reason"] = stop_reason
            summary = write_terminal_artifacts(contract_errors)

        final_write_finished = clock()
        observed_final_write_wall_seconds = max(0.0, final_write_finished - final_write_started)
        observed_finish_consumed = (
            prior_consumed_wall_seconds
            + measured_segment_wall_seconds
            + measured_finalization_wall_seconds
            + observed_final_write_wall_seconds
        )
        if (
            observed_final_write_wall_seconds > charged_final_write_reserve_seconds
            or final_write_finished > hard_deadline
        ):
            stop_reason = "contract_failed"
            contract_errors.append(
                "terminal artifact writes exceeded their charged reserve or hard deadline"
            )
            terminal_stop.update(
                {
                    "stop_reason": stop_reason,
                    "consumed_wall_seconds": observed_finish_consumed,
                    "measured_finalization_wall_seconds": (
                        measured_finalization_wall_seconds + observed_final_write_wall_seconds
                    ),
                    "charged_final_write_reserve_seconds": 0.0,
                    "hard_deadline_overrun_seconds": max(
                        0.0,
                        observed_finish_consumed - float(resolved.shared_wall_seconds),
                    ),
                    "deadline_reached": (observed_finish_consumed >= resolved.shared_wall_seconds),
                }
            )
            summary = write_terminal_artifacts(contract_errors)

        # Verification is a read-only post-hoc phase.  It is intentionally
        # outside the experiment budget and cannot resume or add samples.
        verification = verify_discovery_artifact(store.root)
        if not verification["valid"]:
            verification_errors = [
                str(error) for error in cast(list[object], verification["errors"])
            ]
            merged_errors = list(dict.fromkeys([*contract_errors, *verification_errors]))
            terminal_stop["stop_reason"] = "contract_failed"
            summary = write_terminal_artifacts(merged_errors)
            verification = verify_discovery_artifact(store.root)
        store.write_json("verification.json", cast(Mapping[str, Any], verification))
        return summary
    finally:
        # A run-local cooperative handler must never leak into the caller.
        interrupts.restore()
