"""Fresh-seed OI-1000 confirmation protocol.

This module is deliberately separate from :mod:`calibration`.  The old
``algorithm-calibration-v1`` matrix remains the compatibility baseline; this
protocol owns its own seed registry, gate literals, artifact layout, and
reducer.  The public CLI is intentionally not extended: callers use the
library API from an explicitly approved experiment runner.
"""

from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import os
import random
import re
import resource
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, ProcessPoolExecutor, wait
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import jsonschema  # type: ignore[import-untyped]

from strategy2048.experiments.artifacts import (
    ArtifactError,
    ArtifactStore,
    KnowledgeManifest,
    canonical_json,
    config_hash,
)
from strategy2048.experiments.discovery import (
    DiscoveryArmConfig,
    DiscoveryLearnerConfig,
    DiscoveryPilotConfig,
    _build_agent,
    _initial_snapshot,
    _InterruptController,
    _restore_metrics,
    _RunState,
    _save_checkpoint,
    _train_one_episode,
)
from strategy2048.experiments.evaluation import FrozenPolicyAgent, evaluate_frozen
from strategy2048.learning.td import DEFAULT_TUPLES

CONFIRMATION_SCHEMA_VERSION = "oi-baseline-confirmation-v1"
CONFIRMATION_SCHEMA_PATH = (
    Path(__file__).parents[3] / "schemas/oi-baseline-confirmation.v1.schema.json"
)
CONFIRMATION_CANDIDATES = (("td0_zero", 0.0), ("td0_oi_1000", 1000.0))
CONFIRMATION_ARM_IDS = tuple(item[0] for item in CONFIRMATION_CANDIDATES)
CONFIRMATION_GATES = (
    "oi-baseline-confirmed",
    "oi-baseline-rejected",
    "continue",
    "inconclusive",
    "performance-blocked",
    "contract-failed",
)
CONFIRMATION_COHORT_SIZE = 2
CONFIRMATION_MIN_FRESH_SEEDS = 4
CONFIRMATION_MAX_FRESH_SEEDS = 8
CONFIRMATION_TRAINING_TARGET_EPISODE = 200
CONFIRMATION_AUDIT_EPISODES = 50
CONFIRMATION_WALL_SECONDS = 1800
CONFIRMATION_FINALIZATION_RESERVE_SECONDS = 30.0
CONFIRMATION_PREFLIGHT_WALL_SECONDS = 240
CONFIRMATION_GATE_MIN_MEDIAN_SCORE_GAIN = 0.15
CONFIRMATION_GATE_MIN_POSITIVE_SHARE = 0.75
CONFIRMATION_GATE_SEVERE_REGRESSION = -0.20
CONFIRMATION_GATE_MIN_MEDIAN_TILE_REACH_DELTA = 0.0
CONFIRMATION_THREAD_ENV = {
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
CONFIRMATION_PREFLIGHT_FIXTURE_SEEDS = (
    "confirmation-preflight-01-v1",
    "confirmation-preflight-02-v1",
)
FORMAL_CONFIRMATION_SCHEMA_VERSION = "oi-baseline-confirmation-formal-campaign-v1"
FORMAL_CONFIRMATION_LEDGER_SCHEMA_VERSION = "oi-baseline-confirmation-ledger-v1"
FORMAL_CONFIRMATION_PHASES = (
    "source-runner",
    "source-verification",
    "derived-contract",
    "independent-check",
)

# These are the historical calibration roots and are never eligible as fresh
# confirmation evidence.  The resolver also requires them to be repeated in
# the resolved config so the denylist is part of the immutable artifact.
CONFIRMATION_LEGACY_SEEDS = (
    "calibration-train-a-v1",
    "calibration-train-b-v1",
    "calibration-selection-v1",
    "calibration-audit-v1",
)

ConfirmationInitialization = Literal["zero", "optimistic"]
ConfirmationGate = Literal[
    "oi-baseline-confirmed",
    "oi-baseline-rejected",
    "continue",
    "inconclusive",
    "performance-blocked",
    "contract-failed",
]


class ConfirmationConfigError(ValueError):
    """The versioned confirmation configuration is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class ConfirmationCandidateConfig:
    id: str
    initialization: ConfirmationInitialization
    optimistic_total_value: float

    def to_json(self) -> dict[str, object]:
        return {
            "id": self.id,
            "initialization": self.initialization,
            "optimistic_total_value": self.optimistic_total_value,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationResourceConfig:
    worker_count: int
    start_method: Literal["spawn"]
    thread_env: tuple[tuple[str, str], ...]
    host: str
    cpu_affinity: tuple[int, ...]

    def to_json(self) -> dict[str, object]:
        return {
            "worker_count": self.worker_count,
            "start_method": self.start_method,
            "thread_env": dict(self.thread_env),
            "host": self.host,
            "cpu_affinity": list(self.cpu_affinity),
        }


@dataclass(frozen=True, slots=True)
class ConfirmationConfig:
    experiment_id: str
    output_root: str
    training_seeds: tuple[str, ...]
    evaluation_root_seed: str
    legacy_seed_denylist: tuple[str, ...]
    learner: DiscoveryLearnerConfig
    candidates: tuple[ConfirmationCandidateConfig, ConfirmationCandidateConfig]
    resources: ConfirmationResourceConfig
    max_steps_per_episode: int | None = None
    schema_version: str = CONFIRMATION_SCHEMA_VERSION
    cohort_size: int = CONFIRMATION_COHORT_SIZE
    minimum_fresh_seeds: int = CONFIRMATION_MIN_FRESH_SEEDS
    maximum_fresh_seeds: int = CONFIRMATION_MAX_FRESH_SEEDS
    training_target_episode: int = CONFIRMATION_TRAINING_TARGET_EPISODE
    audit_evaluation_episodes: int = CONFIRMATION_AUDIT_EPISODES
    campaign_wall_seconds: int = CONFIRMATION_WALL_SECONDS
    finalization_reserve_seconds: float = CONFIRMATION_FINALIZATION_RESERVE_SECONDS
    scaling_preflight_wall_seconds: int = CONFIRMATION_PREFLIGHT_WALL_SECONDS
    minimum_median_score_gain: float = CONFIRMATION_GATE_MIN_MEDIAN_SCORE_GAIN
    minimum_positive_share: float = CONFIRMATION_GATE_MIN_POSITIVE_SHARE
    severe_regression_threshold: float = CONFIRMATION_GATE_SEVERE_REGRESSION
    minimum_median_tile_reach_delta: float = CONFIRMATION_GATE_MIN_MEDIAN_TILE_REACH_DELTA

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "output_root": self.output_root,
            "training_seeds": list(self.training_seeds),
            "evaluation_root_seed": self.evaluation_root_seed,
            "legacy_seed_denylist": list(self.legacy_seed_denylist),
            "cohort_size": self.cohort_size,
            "minimum_fresh_seeds": self.minimum_fresh_seeds,
            "maximum_fresh_seeds": self.maximum_fresh_seeds,
            "training_target_episode": self.training_target_episode,
            "audit_evaluation_episodes": self.audit_evaluation_episodes,
            "max_steps_per_episode": self.max_steps_per_episode,
            "campaign_wall_seconds": self.campaign_wall_seconds,
            "finalization_reserve_seconds": self.finalization_reserve_seconds,
            "scaling_preflight_wall_seconds": self.scaling_preflight_wall_seconds,
            "candidates": [candidate.to_json() for candidate in self.candidates],
            "learner": self.learner.to_json(),
            "gate": {
                "minimum_median_score_gain": self.minimum_median_score_gain,
                "minimum_positive_share": self.minimum_positive_share,
                "severe_regression_threshold": self.severe_regression_threshold,
                "minimum_median_tile_reach_delta": self.minimum_median_tile_reach_delta,
            },
            "resources": self.resources.to_json(),
        }


def _load_schema() -> dict[str, Any]:
    value = json.loads(CONFIRMATION_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationConfigError("confirmation schema must be an object")
    return value


def _schema_error(error: jsonschema.ValidationError) -> str:
    location = ".".join(str(part) for part in error.absolute_path)
    prefix = f" at {location}" if location else ""
    return f"confirmation config schema validation failed{prefix}: {error.message}"


def resolve_confirmation_config(value: Mapping[str, Any]) -> ConfirmationConfig:
    """Validate and canonicalize the fixed fresh-seed confirmation protocol."""

    raw = dict(value)
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(raw)
    except jsonschema.ValidationError as error:
        raise ConfirmationConfigError(_schema_error(error)) from error

    training_values = tuple(cast(list[str], raw["training_seeds"]))
    legacy_values_raw = tuple(cast(list[str], raw["legacy_seed_denylist"]))
    if set(legacy_values_raw) != set(CONFIRMATION_LEGACY_SEEDS):
        raise ConfirmationConfigError(
            "legacy_seed_denylist must contain the fixed calibration roots"
        )
    legacy_values = CONFIRMATION_LEGACY_SEEDS
    all_seed_names = set(training_values) | {cast(str, raw["evaluation_root_seed"])}
    if all_seed_names & set(legacy_values):
        raise ConfirmationConfigError("fresh confirmation seeds must be disjoint from legacy seeds")
    if all_seed_names & set(CONFIRMATION_PREFLIGHT_FIXTURE_SEEDS):
        raise ConfirmationConfigError(
            "formal confirmation seeds must be disjoint from preflight fixtures"
        )
    if len(all_seed_names) != len(training_values) + 1:
        raise ConfirmationConfigError(
            "training and evaluation seed roots must be pairwise distinct"
        )

    candidates_by_id: dict[str, Mapping[str, Any]] = {}
    for item in cast(list[Mapping[str, Any]], raw["candidates"]):
        candidate_id = cast(str, item["id"])
        if candidate_id in candidates_by_id:
            raise ConfirmationConfigError(f"duplicate confirmation candidate id: {candidate_id}")
        candidates_by_id[candidate_id] = item
    if set(candidates_by_id) != set(CONFIRMATION_ARM_IDS):
        raise ConfirmationConfigError("confirmation requires exactly td0_zero and td0_oi_1000")
    candidates: list[ConfirmationCandidateConfig] = []
    for candidate_id, expected_value in CONFIRMATION_CANDIDATES:
        item = candidates_by_id[candidate_id]
        expected_initialization = "zero" if expected_value == 0.0 else "optimistic"
        initialization = cast(str, item["initialization"])
        actual_value = float(item["optimistic_total_value"])
        if initialization != expected_initialization:
            raise ConfirmationConfigError(
                f"{candidate_id} initialization must be {expected_initialization}"
            )
        if not math.isfinite(actual_value) or actual_value != expected_value:
            raise ConfirmationConfigError(
                f"{candidate_id} optimistic_total_value must be {expected_value:g}"
            )
        candidates.append(
            ConfirmationCandidateConfig(
                id=candidate_id,
                initialization=cast(ConfirmationInitialization, initialization),
                optimistic_total_value=actual_value,
            )
        )

    learner_raw = cast(Mapping[str, Any], raw["learner"])
    tuples_raw = learner_raw.get("tuples")
    tuples = (
        DEFAULT_TUPLES
        if tuples_raw is None
        else tuple(
            tuple(int(index) for index in item) for item in cast(list[list[int]], tuples_raw)
        )
    )
    if not tuples or any(len(item) != len(tuples[0]) for item in tuples):
        raise ConfirmationConfigError("all learner tuples must have the same non-zero length")
    alpha = float(learner_raw["alpha"])
    gamma = float(learner_raw["gamma"])
    if not math.isfinite(alpha) or not 0.0 < alpha <= 1.0:
        raise ConfirmationConfigError("learner.alpha must be finite and in (0, 1]")
    if not math.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ConfirmationConfigError("learner.gamma must be finite and in [0, 1]")

    gate_raw = cast(Mapping[str, Any], raw["gate"])
    resource_raw = cast(Mapping[str, Any], raw["resources"])
    thread_raw = cast(Mapping[str, Any], resource_raw["thread_env"])
    thread_env = tuple(sorted((str(key), str(value)) for key, value in thread_raw.items()))
    if dict(thread_env) != CONFIRMATION_THREAD_ENV:
        raise ConfirmationConfigError(
            "confirmation resource thread_env must pin every backend to 1"
        )
    affinity = tuple(sorted(cast(list[int], resource_raw["cpu_affinity"])))
    if len(set(affinity)) != len(affinity):
        raise ConfirmationConfigError("resources.cpu_affinity must not contain duplicates")

    config = ConfirmationConfig(
        experiment_id=cast(str, raw["experiment_id"]),
        output_root=cast(str, raw["output_root"]),
        training_seeds=training_values,
        evaluation_root_seed=cast(str, raw["evaluation_root_seed"]),
        legacy_seed_denylist=legacy_values,
        learner=DiscoveryLearnerConfig(
            alpha=alpha,
            gamma=gamma,
            symmetry=cast(bool, learner_raw["symmetry"]),
            value_cardinality=cast(int, learner_raw["value_cardinality"]),
            tuples=tuples,
        ),
        candidates=cast(
            tuple[ConfirmationCandidateConfig, ConfirmationCandidateConfig], tuple(candidates)
        ),
        resources=ConfirmationResourceConfig(
            worker_count=cast(int, resource_raw["worker_count"]),
            start_method="spawn",
            thread_env=thread_env,
            host=cast(str, resource_raw["host"]),
            cpu_affinity=affinity,
        ),
        max_steps_per_episode=cast(int | None, raw.get("max_steps_per_episode")),
        minimum_median_score_gain=float(gate_raw["minimum_median_score_gain"]),
        minimum_positive_share=float(gate_raw["minimum_positive_share"]),
        severe_regression_threshold=float(gate_raw["severe_regression_threshold"]),
        minimum_median_tile_reach_delta=float(gate_raw["minimum_median_tile_reach_delta"]),
    )
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(config.to_json())
    except jsonschema.ValidationError as error:
        raise ConfirmationConfigError(_schema_error(error)) from error
    return config


def load_confirmation_config(path: str | Path) -> ConfirmationConfig:
    with Path(path).open("rb") as handle:
        return resolve_confirmation_config(tomllib.load(handle))


def confirmation_config_hash(config: ConfirmationConfig | Mapping[str, Any]) -> str:
    """Return the digest of the immutable resolved protocol configuration."""

    resolved = config.to_json() if isinstance(config, ConfirmationConfig) else dict(config)
    return config_hash(resolved)


def _finite_number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ArtifactError(f"{field} must be finite")
    return result


def _score_and_tile_values(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Accept the canonical paired shape and the two-arm aggregate shape."""

    if "zero_mean_score" in record:
        zero_score = _finite_number(record["zero_mean_score"], field="zero_mean_score")
        oi_score = _finite_number(record.get("oi_mean_score"), field="oi_mean_score")
        zero_tile = _finite_number(record.get("zero_256_reach_rate"), field="zero_256_reach_rate")
        oi_tile = _finite_number(record.get("oi_256_reach_rate"), field="oi_256_reach_rate")
        return zero_score, oi_score, zero_tile, oi_tile
    zero = record.get("zero")
    optimistic = record.get("td0_oi_1000", record.get("oi"))
    if not isinstance(zero, Mapping) or not isinstance(optimistic, Mapping):
        raise ArtifactError("paired audit record must contain zero and OI aggregates")
    zero_score = _finite_number(zero.get("mean_score"), field="zero.mean_score")
    oi_score = _finite_number(optimistic.get("mean_score"), field="oi.mean_score")
    zero_tile = _finite_number(zero.get("tile_reach_rate_256"), field="zero.tile_reach_rate_256")
    oi_tile = _finite_number(optimistic.get("tile_reach_rate_256"), field="oi.tile_reach_rate_256")
    return zero_score, oi_score, zero_tile, oi_tile


def _bootstrap_median_interval(values: Sequence[float], *, repetitions: int = 1024) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(0x2048)
    medians = [
        statistics.median(generator.choice(values) for _ in values) for _ in range(repetitions)
    ]
    medians.sort()
    lower = int(0.025 * (len(medians) - 1))
    upper = int(0.975 * (len(medians) - 1))
    return [medians[lower], medians[upper]]


def reduce_confirmation_gate(
    paired_records: Sequence[Mapping[str, Any]],
    *,
    minimum_fresh_seeds: int = CONFIRMATION_MIN_FRESH_SEEDS,
    maximum_fresh_seeds: int = CONFIRMATION_MAX_FRESH_SEEDS,
    minimum_median_score_gain: float = CONFIRMATION_GATE_MIN_MEDIAN_SCORE_GAIN,
    minimum_positive_share: float = CONFIRMATION_GATE_MIN_POSITIVE_SHARE,
    severe_regression_threshold: float = CONFIRMATION_GATE_SEVERE_REGRESSION,
    minimum_median_tile_reach_delta: float = CONFIRMATION_GATE_MIN_MEDIAN_TILE_REACH_DELTA,
) -> dict[str, Any]:
    """Recompute the predeclared paired engineering gate from raw seed records."""

    if minimum_fresh_seeds <= 0 or maximum_fresh_seeds < minimum_fresh_seeds:
        raise ArtifactError("invalid confirmation seed bounds")
    if len(paired_records) > maximum_fresh_seeds:
        raise ArtifactError("confirmation gate received more than the predeclared seed maximum")
    identities: set[str] = set()
    score_gains: list[float] = []
    tile_deltas: list[float] = []
    effects: list[dict[str, Any]] = []
    for index, record in enumerate(paired_records):
        identity = record.get("training_seed", str(index))
        if not isinstance(identity, str) or identity in identities:
            raise ArtifactError("confirmation gate contains duplicate training seed records")
        identities.add(identity)
        zero_score, oi_score, zero_tile, oi_tile = _score_and_tile_values(record)
        if zero_score <= 0.0:
            raise ArtifactError("paired score denominator must be positive")
        score_gain = (oi_score - zero_score) / zero_score
        tile_delta = oi_tile - zero_tile
        if not math.isfinite(score_gain) or not math.isfinite(tile_delta):
            raise ArtifactError("paired gate metrics must be finite")
        score_gains.append(score_gain)
        tile_deltas.append(tile_delta)
        effects.append(
            {
                "training_seed": identity,
                "score_gain": score_gain,
                "tile_reach_delta": tile_delta,
                "severe_regression": score_gain <= severe_regression_threshold,
            }
        )

    seed_count = len(score_gains)
    median_score_gain = statistics.median(score_gains) if score_gains else 0.0
    median_tile_delta = statistics.median(tile_deltas) if tile_deltas else 0.0
    positive_count = sum(value > 0.0 for value in score_gains)
    positive_share = positive_count / seed_count if seed_count else 0.0
    severe_count = sum(value <= severe_regression_threshold for value in score_gains)
    enough = seed_count >= minimum_fresh_seeds
    confirmed = enough and (
        median_score_gain >= minimum_median_score_gain
        and positive_share >= minimum_positive_share
        and severe_count == 0
        and median_tile_delta >= minimum_median_tile_reach_delta
    )
    rejected = enough and (median_score_gain <= 0.0 or positive_share <= 0.25 or severe_count >= 2)
    if confirmed:
        decision: ConfirmationGate = "oi-baseline-confirmed"
        stop_reason = "gate_confirmed"
    elif rejected:
        decision = "oi-baseline-rejected"
        stop_reason = "gate_rejected"
    elif seed_count >= maximum_fresh_seeds:
        decision = "inconclusive"
        stop_reason = "maximum_fresh_seeds_reached_without_gate"
    else:
        decision = "continue"
        stop_reason = "cohort_complete_gate_mixed"
    return {
        "schema_version": "oi-baseline-confirmation-gate-v1",
        "decision": decision,
        "stop_reason": stop_reason,
        "fresh_seed_count": seed_count,
        "score_gains": score_gains,
        "tile_reach_deltas": tile_deltas,
        "median_score_gain": median_score_gain,
        "median_tile_reach_delta": median_tile_delta,
        "positive_seed_count": positive_count,
        "positive_seed_share": positive_share,
        "severe_regression_count": severe_count,
        "per_seed_effects": effects,
        "diagnostics": {
            "mean_score_gain": statistics.fmean(score_gains) if score_gains else 0.0,
            "worst_score_gain": min(score_gains) if score_gains else 0.0,
            "mean_tile_reach_delta": statistics.fmean(tile_deltas) if tile_deltas else 0.0,
            "score_gain_bootstrap_median_interval": _bootstrap_median_interval(score_gains),
            "tile_reach_delta_bootstrap_median_interval": _bootstrap_median_interval(tile_deltas),
        },
        "thresholds": {
            "minimum_fresh_seeds": minimum_fresh_seeds,
            "maximum_fresh_seeds": maximum_fresh_seeds,
            "minimum_median_score_gain": minimum_median_score_gain,
            "minimum_positive_share": minimum_positive_share,
            "severe_regression_threshold": severe_regression_threshold,
            "minimum_median_tile_reach_delta": minimum_median_tile_reach_delta,
        },
    }


def compute_confirmation_gate(
    config: ConfirmationConfig, paired_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    return reduce_confirmation_gate(
        paired_records,
        minimum_fresh_seeds=config.minimum_fresh_seeds,
        maximum_fresh_seeds=config.maximum_fresh_seeds,
        minimum_median_score_gain=config.minimum_median_score_gain,
        minimum_positive_share=config.minimum_positive_share,
        severe_regression_threshold=config.severe_regression_threshold,
        minimum_median_tile_reach_delta=config.minimum_median_tile_reach_delta,
    )


_RUNTIME_KEYS = frozenset(
    {
        "wall_seconds",
        "process_cpu_seconds",
        "rss_bytes",
        "peak_rss_bytes",
        "timestamp",
        "started_at",
        "finished_at",
        "pid",
        "temporary_path",
        "attempt_path",
        "completion_order",
        "runtime_telemetry",
        "telemetry",
        "metrics",
    }
)


def scientific_projection(value: object) -> object:
    """Remove scheduling/resource observations before hashing scientific data."""

    if isinstance(value, Mapping):
        return {
            str(key): scientific_projection(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            if str(key) not in _RUNTIME_KEYS
        }
    if isinstance(value, list):
        return [scientific_projection(item) for item in value]
    if isinstance(value, tuple):
        return [scientific_projection(item) for item in value]
    return value


def scientific_digest(value: object) -> str:
    payload = scientific_projection(value)
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def validate_runtime_telemetry(
    telemetry: Mapping[str, Any],
    *,
    budget_seconds: float = CONFIRMATION_WALL_SECONDS,
    max_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Validate runtime observations without allowing them into the science digest."""

    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)):
        raise ArtifactError("runtime telemetry budget must be numeric")
    budget = float(budget_seconds)
    if not math.isfinite(budget) or budget < 0.0:
        raise ArtifactError("runtime telemetry budget must be finite and non-negative")
    if max_rss_bytes is not None and (type(max_rss_bytes) is not int or max_rss_bytes < 0):
        raise ArtifactError("runtime telemetry RSS bound must be a non-negative integer")

    def walk(value: Mapping[str, Any], prefix: str = "") -> tuple[dict[str, Any], float, float]:
        result: dict[str, Any] = {}
        own_wall: float | None = None
        own_cpu: float | None = None
        child_wall = 0.0
        child_cpu = 0.0
        for key, raw_value in value.items():
            field = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(raw_value, Mapping):
                nested, nested_wall, nested_cpu = walk(cast(Mapping[str, Any], raw_value), field)
                result[str(key)] = nested
                child_wall += nested_wall
                child_cpu += nested_cpu
                continue
            if key in {"wall_seconds", "process_cpu_seconds"} or key.endswith(
                ("_wall_seconds", "_process_cpu_seconds")
            ):
                number = _finite_number(raw_value, field=field)
                if number < 0.0:
                    raise ArtifactError(f"{field} must be non-negative")
                result[str(key)] = number
                if key == "wall_seconds":
                    own_wall = number
                elif key == "process_cpu_seconds":
                    own_cpu = number
                elif key.endswith("_wall_seconds"):
                    child_wall += number
                else:
                    child_cpu += number
            elif key in {"rss_bytes", "peak_rss_bytes"}:
                if isinstance(raw_value, bool) or not isinstance(raw_value, int) or raw_value < 0:
                    raise ArtifactError(f"{field} must be a non-negative integer")
                if max_rss_bytes is not None and raw_value > max_rss_bytes:
                    raise ArtifactError(f"{field} exceeds the resource contract")
                result[str(key)] = raw_value
            else:
                result[str(key)] = raw_value
        if own_wall is not None and child_wall > own_wall + 1e-9:
            raise ArtifactError(f"{prefix or '<root>'} phase wall time exceeds its aggregate")
        if own_cpu is not None and child_cpu > own_cpu + 1e-9:
            raise ArtifactError(f"{prefix or '<root>'} phase CPU time exceeds its aggregate")
        return (
            result,
            own_wall if own_wall is not None else child_wall,
            own_cpu if own_cpu is not None else child_cpu,
        )

    normalized, total_wall, total_cpu = walk(telemetry)
    if total_wall > budget + 1e-9:
        raise ArtifactError("runtime telemetry exceeds the confirmation wall budget")
    if total_cpu < 0.0:
        raise ArtifactError("runtime telemetry CPU accounting is invalid")
    return normalized


def fixed_thread_environment() -> dict[str, str]:
    """Return a copy of the required single-thread numerical backend settings."""

    return dict(CONFIRMATION_THREAD_ENV)


def apply_thread_environment() -> None:
    """Pin BLAS/OpenMP backends before creating a process pool."""

    for key, value in CONFIRMATION_THREAD_ENV.items():
        os.environ[key] = value


def apply_resource_contract(config: ConfirmationConfig) -> None:
    """Apply the frozen numerical-thread and optional worker affinity contract."""

    apply_thread_environment()
    if not config.resources.cpu_affinity:
        return
    if not hasattr(os, "sched_setaffinity"):
        raise ArtifactError("confirmation CPU affinity is unavailable on this host")
    try:
        os.sched_setaffinity(0, set(config.resources.cpu_affinity))
    except OSError as error:
        raise ArtifactError("confirmation CPU affinity could not be applied") from error


def _peak_rss_bytes() -> int | None:
    """Return the worker's peak resident set size when the host exposes it."""

    try:
        raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, OSError):
        return None
    if isinstance(raw, bool) or not isinstance(raw, (int, float)) or raw < 0:
        return None
    # Linux reports KiB; macOS reports bytes.  The product runtime is Linux,
    # but keeping the fallback makes the artifact contract portable.
    multiplier = 1024 if sys.platform.startswith("linux") else 1
    return int(raw * multiplier)


@dataclass(frozen=True, slots=True)
class ConfirmationShardRequest:
    """Pickle-safe immutable input for one ``(candidate, training_seed)`` shard."""

    config: dict[str, Any]
    candidate_id: str
    training_seed: str
    destination: str
    deadline_seconds: float | None = None
    resume_from: str | None = None
    publish_deadline: float | None = None


def _confirmation_execution_config(
    config: ConfirmationConfig, candidate: ConfirmationCandidateConfig, training_seed: str
) -> DiscoveryPilotConfig:
    arm = cast(DiscoveryArmConfig, candidate)
    return DiscoveryPilotConfig(
        experiment_id=config.experiment_id,
        output_root=config.output_root,
        round_robin_training_chunk=10,
        training_seeds=(training_seed, training_seed),
        evaluation_root_seed=config.evaluation_root_seed,
        evaluation_episodes_per_checkpoint=config.audit_evaluation_episodes,
        diagnostic_score_milestone=1,
        diagnostic_tile_milestone=256,
        learner=config.learner,
        arms=(arm, arm),
        max_steps_per_episode=config.max_steps_per_episode,
        shared_wall_seconds=config.campaign_wall_seconds,
        finalization_reserve_seconds=config.finalization_reserve_seconds,
        checkpoint_episodes=(0, 40, config.training_target_episode),
        max_training_episodes_per_run=config.training_target_episode,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"confirmation JSON artifact is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError(f"confirmation JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise ArtifactError(f"confirmation JSONL artifact is not a regular file: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ArtifactError(f"confirmation JSONL record {line_number} is not an object")
        records.append(value)
    return records


def _assert_no_symlink_components(path: Path) -> None:
    """Reject a path whose lexical parent chain contains a symlink."""

    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ArtifactError(f"confirmation path contains a symlink component: {path}")


def _assert_no_symlinks_in_tree(root: Path) -> None:
    _assert_no_symlink_components(root)
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"confirmation artifact tree contains a symlink: {path}")


def _path_in_shard(root: Path, relative: object, *, field: str) -> Path:
    """Resolve an artifact-relative path while rejecting traversal and links."""

    if not isinstance(relative, str):
        raise ArtifactError(f"{field} must be a relative path")
    candidate = Path(relative)
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        raise ArtifactError(f"{field} must stay inside the shard")
    _assert_no_symlink_components(root)
    lexical = root
    for part in candidate.parts:
        lexical /= part
        if lexical.is_symlink():
            raise ArtifactError(f"{field} contains a symlink component")
    resolved_root = root.resolve(strict=True)
    resolved = lexical.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ArtifactError(f"{field} escapes the shard") from error
    return resolved


def _assert_disjoint_paths(left: Path, right: Path) -> None:
    """Reject overlapping lexical artifact roots before copying or publishing."""

    _assert_no_symlink_components(left)
    _assert_no_symlink_components(right)
    left_resolved = left.resolve(strict=True)
    right_resolved = right.resolve(strict=False)
    if (
        left_resolved == right_resolved
        or left_resolved in right_resolved.parents
        or right_resolved in left_resolved.parents
    ):
        raise ArtifactError("confirmation artifact roots must be disjoint")


def _shard_manifest(
    config: ConfirmationConfig, candidate: ConfirmationCandidateConfig, seed: str
) -> dict[str, Any]:
    return {
        "schema_version": "oi-baseline-confirmation-shard-v1",
        "experiment_id": config.experiment_id,
        "config_hash": confirmation_config_hash(config),
        "candidate_id": candidate.id,
        "training_seed": seed,
        "evaluation_root_seed": config.evaluation_root_seed,
        "resource_contract": config.resources.to_json(),
        "target_training_episode": config.training_target_episode,
        "audit_evaluation_episodes": config.audit_evaluation_episodes,
        "status": "completed",
        "scientific_digest": None,
        "runtime_telemetry": {},
    }


def _shard_scientific_digest(
    manifest: Mapping[str, Any],
    training: Sequence[Mapping[str, Any]],
    evaluation: Sequence[Mapping[str, Any]],
) -> str:
    digest_manifest = dict(manifest)
    digest_manifest["scientific_digest"] = None
    return scientific_digest(
        {"manifest": digest_manifest, "training": list(training), "evaluation": list(evaluation)}
    )


def _write_evaluation_records(
    store: ArtifactStore,
    config: ConfirmationConfig,
    candidate: ConfirmationCandidateConfig,
    training_seed: str,
    checkpoint: Mapping[str, Any],
    *,
    deadline: float | None = None,
) -> dict[str, Any]:
    if deadline is not None and time.monotonic() >= deadline:
        raise ArtifactError("confirmation shard exhausted its deadline before audit evaluation")
    execution = _confirmation_execution_config(config, candidate, training_seed)
    clone = _build_agent(execution, cast(DiscoveryArmConfig, candidate))
    checkpoint_path = store.root / str(checkpoint["checkpoint_directory"])
    step = checkpoint.get("global_env_step")
    if type(step) is not int or step < 0:
        raise ArtifactError("confirmation audit checkpoint step is invalid")
    clone.restore_checkpoint(checkpoint_path, step, config_hash=store.config_hash)
    frozen = FrozenPolicyAgent(clone.learner)
    started_cpu = time.process_time()
    result = evaluate_frozen(
        frozen,
        episodes=config.audit_evaluation_episodes,
        root_seed=config.evaluation_root_seed,
        purpose="confirmation-audit",
        environment_id=f"{config.experiment_id}-audit",
        max_steps=config.max_steps_per_episode,
        episode_ids=tuple(range(config.audit_evaluation_episodes)),
        deadline=deadline,
    )
    if result["completed_episodes"] != config.audit_evaluation_episodes:
        raise ArtifactError("confirmation shard exhausted its deadline during audit evaluation")
    process_cpu_seconds = max(0.0, time.process_time() - started_cpu)
    episodes = cast(list[dict[str, Any]], result["episodes"])
    for episode in episodes:
        store.append_jsonl(
            "run/evaluation/episodes.jsonl",
            {
                "schema_version": "oi-baseline-confirmation-evaluation-v1",
                "candidate_id": candidate.id,
                "training_seed": training_seed,
                "evaluation_root_seed": config.evaluation_root_seed,
                "checkpoint_episode": config.training_target_episode,
                "evaluation_episode_id": episode["episode_id"],
                "official_score": episode["official_score"],
                "max_tile": episode["max_tile"],
                "steps": episode["steps"],
                "terminated": episode["terminated"],
                "truncated": episode["truncated"],
                "wall_seconds": episode["wall_seconds"],
                "process_cpu_seconds": process_cpu_seconds / max(len(episodes), 1),
                "frozen_state_unchanged": result["state_unchanged"],
                "clone_state_hash_before": result["state_hash_before"],
                "clone_state_hash_after": result["state_hash_after"],
                "clone_table_hash_before": result["table_hash_before"],
                "clone_table_hash_after": result["table_hash_after"],
                "clone_counters_before": result["counters_before"],
                "clone_counters_after": result["counters_after"],
            },
        )
    summary = {
        "schema_version": "oi-baseline-confirmation-evaluation-summary-v1",
        "candidate_id": candidate.id,
        "training_seed": training_seed,
        "evaluation_root_seed": config.evaluation_root_seed,
        "checkpoint_episode": config.training_target_episode,
        "requested_episodes": config.audit_evaluation_episodes,
        "completed_episodes": result["completed_episodes"],
        "official_score": cast(Mapping[str, Any], result["score"]),
        "tile_reach_rate": result["tile_reach_rate"],
        "max_tile_mean": result["max_tile_mean"],
        "runtime_telemetry": {
            "wall_seconds": cast(Mapping[str, Any], result["metrics"])["wall_seconds"].get(
                "evaluation", 0.0
            ),
            "process_cpu_seconds": process_cpu_seconds,
        },
    }
    store.write_json("run/evaluation/summary.json", summary)
    return summary


def _normalize_training_records(
    path: Path, *, candidate_id: str, training_seed: str
) -> list[dict[str, Any]]:
    records = _read_jsonl(path)
    normalized: list[dict[str, Any]] = []
    for record in records:
        value = dict(record)
        value["candidate_id"] = candidate_id
        value["training_seed"] = training_seed
        normalized.append(value)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        "".join(canonical_json(record) + "\n" for record in normalized),
        encoding="utf-8",
    )
    temporary.replace(path)
    return normalized


def _open_confirmation_shard_store(root: Path, config: ConfirmationConfig) -> ArtifactStore:
    store = ArtifactStore.__new__(ArtifactStore)
    store.root = root
    store.resolved_config = config.to_json()
    store.config_hash = confirmation_config_hash(config)
    store.repo_root = Path(__file__).parents[3]
    store.started_at = time.time()
    return store


def _restore_confirmation_shard_state(
    root: Path,
    store: ArtifactStore,
    config: ConfirmationConfig,
    candidate: ConfirmationCandidateConfig,
    training_seed: str,
) -> tuple[_RunState, dict[str, Any]]:
    pointer = _read_json(root / "run/resume-checkpoint.json")
    if pointer.get("config_hash") != store.config_hash:
        raise ArtifactError("confirmation resume config hash mismatch")
    if pointer.get("arm_id") != candidate.id or pointer.get("training_seed") != training_seed:
        raise ArtifactError("confirmation resume shard identity mismatch")
    if pointer.get("completed_training_episodes") == config.training_target_episode:
        raise ArtifactError("completed confirmation shard cannot be resumed")
    execution = _confirmation_execution_config(config, candidate, training_seed)
    state = _RunState(
        arm=cast(DiscoveryArmConfig, candidate),
        training_seed=training_seed,
        agent=_build_agent(execution, cast(DiscoveryArmConfig, candidate)),
        relative_root="run",
    )
    checkpoint_directory = _path_in_shard(
        root, pointer.get("checkpoint_directory"), field="confirmation resume checkpoint_directory"
    )
    if not checkpoint_directory.is_dir():
        raise ArtifactError("confirmation resume checkpoint directory is invalid")
    snapshot = state.agent.restore_checkpoint(
        checkpoint_directory,
        0,
        config_hash=store.config_hash,
    )
    if pointer.get("learner_state_hash") != state.agent.learner.state_hash():
        raise ArtifactError("confirmation resume learner state hash mismatch")
    if pointer.get("table_hash") != state.agent.learner.table_hash():
        raise ArtifactError("confirmation resume table hash mismatch")
    if canonical_json(pointer.get("counters")) != canonical_json(state.agent.counters.to_json()):
        raise ArtifactError("confirmation resume learner counters mismatch")
    if canonical_json(pointer.get("environment")) != canonical_json(snapshot.to_json()):
        raise ArtifactError("confirmation resume environment mismatch")
    completed = pointer.get("completed_training_episodes")
    global_step = pointer.get("global_env_step")
    if (
        type(completed) is not int
        or completed < 0
        or type(global_step) is not int
        or global_step < 0
    ):
        raise ArtifactError("confirmation resume progress is malformed")
    state.completed_episodes = completed
    state.global_env_steps = global_step
    state.last_snapshot = snapshot
    state.metrics = _restore_metrics(pointer)
    for field_name in ("active_wall_seconds", "process_cpu_seconds"):
        raw = pointer.get(field_name, 0.0)
        if (
            isinstance(raw, bool)
            or not isinstance(raw, (int, float))
            or not math.isfinite(float(raw))
            or float(raw) < 0
        ):
            raise ArtifactError(f"confirmation resume {field_name} is invalid")
        setattr(state, field_name, float(raw))
    if pointer.get("resume_in_episode"):
        resume_episode = pointer.get("resume_episode")
        if not isinstance(resume_episode, Mapping):
            raise ArtifactError("confirmation resume episode accumulators are missing")
        env_steps_before = resume_episode.get("env_steps_before")
        counters_before = resume_episode.get("counters_before")
        wall_seconds = resume_episode.get("wall_seconds")
        process_cpu_seconds = resume_episode.get("process_cpu_seconds")
        if type(env_steps_before) is not int or env_steps_before < 0:
            raise ArtifactError("confirmation resume episode step accumulator is invalid")
        if not isinstance(counters_before, Mapping):
            raise ArtifactError("confirmation resume episode counters are invalid")
        for field_name, raw_value in (
            ("wall_seconds", wall_seconds),
            ("process_cpu_seconds", process_cpu_seconds),
        ):
            if (
                isinstance(raw_value, bool)
                or not isinstance(raw_value, (int, float))
                or not math.isfinite(float(raw_value))
                or float(raw_value) < 0.0
            ):
                raise ArtifactError(f"confirmation resume episode {field_name} is invalid")
        state.resume_snapshot = snapshot
        state.resume_episode_env_steps_before = env_steps_before
        state.resume_episode_counters_before = dict(counters_before)
        state.resume_episode_wall_seconds = float(cast(int | float, wall_seconds))
        state.resume_episode_process_cpu_seconds = float(cast(int | float, process_cpu_seconds))
    return state, pointer


def _run_confirmation_shard(
    temporary_root: Path,
    config: ConfirmationConfig,
    candidate: ConfirmationCandidateConfig,
    training_seed: str,
    *,
    deadline_seconds: float | None,
    execution_deadline: float | None = None,
    resume_from: Path | None = None,
) -> dict[str, Any]:
    apply_resource_contract(config)
    prior_wall = 0.0
    prior_cpu = 0.0
    execution = _confirmation_execution_config(config, candidate, training_seed)
    if resume_from is None:
        store = ArtifactStore(temporary_root, config.to_json(), repo_root=Path(__file__).parents[3])
        store.initialize(
            knowledge_manifest=KnowledgeManifest(
                experiment_kind="discovery",
                initialization={
                    "source": candidate.initialization,
                    "comparison_candidates": list(CONFIRMATION_ARM_IDS),
                    "optimistic_total_value": candidate.optimistic_total_value,
                },
            ),
            seed=training_seed,
            budget={
                "campaign_wall_seconds": config.campaign_wall_seconds,
                "shard_deadline_seconds": deadline_seconds,
            },
        )
        state = _RunState(
            arm=cast(DiscoveryArmConfig, candidate),
            training_seed=training_seed,
            agent=_build_agent(execution, cast(DiscoveryArmConfig, candidate)),
            relative_root="run",
        )
        state.last_snapshot = _initial_snapshot(execution, training_seed)
        checkpoint_records: dict[int, dict[str, Any]] = {}
    else:
        shutil.copytree(resume_from, temporary_root, dirs_exist_ok=True)
        store = _open_confirmation_shard_store(temporary_root, config)
        state, _ = _restore_confirmation_shard_state(
            temporary_root, store, config, candidate, training_seed
        )
        prior_wall = state.active_wall_seconds
        prior_cpu = state.process_cpu_seconds
        checkpoint_records = {
            int(record["checkpoint_episode"]): record
            for record in _read_jsonl(temporary_root / "checkpoints.jsonl")
            if record.get("kind") == "milestone"
        }
    started_wall = time.monotonic()
    started_cpu = time.process_time()
    checkpoint_zero = checkpoint_records.get(0)
    if checkpoint_zero is None:
        checkpoint_zero = _save_checkpoint(store, state, checkpoint_episode=0, kind="milestone")
    checkpoint_40 = checkpoint_records.get(40)
    if checkpoint_40 is None and state.completed_episodes >= 40:
        checkpoint_40 = _save_checkpoint(store, state, checkpoint_episode=40, kind="milestone")
    remaining_seconds = (
        max(0.0, deadline_seconds - prior_wall)
        if deadline_seconds is not None
        else max(0.0, config.campaign_wall_seconds - prior_wall)
    )
    if execution_deadline is not None:
        remaining_seconds = min(remaining_seconds, max(0.0, execution_deadline - time.monotonic()))
    deadline = started_wall + remaining_seconds
    interrupts = _InterruptController()
    interrupts.install()
    try:
        while state.completed_episodes < config.training_target_episode:
            completed = _train_one_episode(
                store,
                execution,
                state,
                clock=time.monotonic,
                deadline=deadline,
                process_clock=time.process_time,
                interrupts=interrupts,
                phase_hook=None,
            )
            if not completed:
                raise ArtifactError("confirmation shard exhausted its deadline before completion")
            if state.completed_episodes == 40 and checkpoint_40 is None:
                checkpoint_40 = _save_checkpoint(
                    store, state, checkpoint_episode=40, kind="milestone"
                )
        checkpoint_200 = checkpoint_records.get(config.training_target_episode)
        if checkpoint_200 is None:
            checkpoint_200 = _save_checkpoint(
                store, state, checkpoint_episode=config.training_target_episode, kind="milestone"
            )
    finally:
        interrupts.restore()
    training_records = _normalize_training_records(
        temporary_root / "run/training-episodes.jsonl",
        candidate_id=candidate.id,
        training_seed=training_seed,
    )
    audit_summary = _write_evaluation_records(
        store, config, candidate, training_seed, checkpoint_200, deadline=deadline
    )
    if resume_from is not None:
        for stale_name in ("partial-shard-manifest.json", "failure.json"):
            stale_path = temporary_root / stale_name
            if stale_path.is_file() or stale_path.is_symlink():
                stale_path.unlink()
    elapsed_wall = prior_wall + max(0.0, time.monotonic() - started_wall)
    elapsed_cpu = prior_cpu + max(0.0, time.process_time() - started_cpu)
    manifest = _shard_manifest(config, candidate, training_seed)
    peak_rss_bytes = _peak_rss_bytes()
    runtime_telemetry: dict[str, Any] = {
        "wall_seconds": elapsed_wall,
        "process_cpu_seconds": elapsed_cpu,
        "training_wall_seconds": state.active_wall_seconds,
        "training_process_cpu_seconds": state.process_cpu_seconds,
        "evaluation": audit_summary["runtime_telemetry"],
    }
    if peak_rss_bytes is not None:
        runtime_telemetry["peak_rss_bytes"] = peak_rss_bytes
    manifest.update(
        {
            "checkpoints": {
                "episode_0": checkpoint_zero,
                "episode_40": checkpoint_40,
                "episode_200": checkpoint_200,
            },
            "training_record_path": "run/training-episodes.jsonl",
            "evaluation_record_path": "run/evaluation/episodes.jsonl",
            "evaluation_summary_path": "run/evaluation/summary.json",
            "learner_state_hash": checkpoint_200["learner_state_hash"],
            "table_hash": checkpoint_200["table_hash"],
            "rng_lineage": cast(Mapping[str, Any], checkpoint_200["environment"])["rng"]["lineage"],
            "runtime_telemetry": runtime_telemetry,
        }
    )
    evaluation_records = _read_jsonl(temporary_root / "run/evaluation/episodes.jsonl")
    manifest["scientific_digest"] = _shard_scientific_digest(
        manifest, training_records, evaluation_records
    )
    store.write_json("shard-manifest.json", manifest)
    return manifest


def run_confirmation_shard(
    request: ConfirmationShardRequest | Mapping[str, Any],
) -> dict[str, Any]:
    """Run one private shard and publish it with one atomic directory rename."""

    if isinstance(request, ConfirmationShardRequest):
        shard_request = request
    else:
        raw_config = request.get("config")
        if not isinstance(raw_config, Mapping):
            raise ArtifactError("confirmation shard request is missing config")
        shard_request = ConfirmationShardRequest(
            config=dict(raw_config),
            candidate_id=cast(str, request.get("candidate_id")),
            training_seed=cast(str, request.get("training_seed")),
            destination=cast(str, request.get("destination")),
            deadline_seconds=cast(float | None, request.get("deadline_seconds")),
            resume_from=cast(str | None, request.get("resume_from")),
            publish_deadline=cast(float | None, request.get("publish_deadline")),
        )
    config = resolve_confirmation_config(shard_request.config)
    candidates = {candidate.id: candidate for candidate in config.candidates}
    candidate = candidates.get(shard_request.candidate_id)
    if candidate is None:
        raise ArtifactError(f"unknown confirmation candidate: {shard_request.candidate_id}")
    if shard_request.training_seed not in config.training_seeds:
        raise ArtifactError("confirmation shard training seed is not in the frozen registry")
    if shard_request.deadline_seconds is not None and (
        isinstance(shard_request.deadline_seconds, bool)
        or not isinstance(shard_request.deadline_seconds, (int, float))
        or not math.isfinite(float(shard_request.deadline_seconds))
        or float(shard_request.deadline_seconds) <= 0.0
    ):
        raise ArtifactError("confirmation shard deadline must be finite and positive")
    if shard_request.publish_deadline is not None and (
        isinstance(shard_request.publish_deadline, bool)
        or not isinstance(shard_request.publish_deadline, (int, float))
        or not math.isfinite(float(shard_request.publish_deadline))
    ):
        raise ArtifactError("confirmation shard publish deadline must be finite")
    destination = Path(shard_request.destination)
    if any(part in {"", ".", ".."} for part in destination.parts):
        raise ArtifactError("confirmation shard destination contains traversal")
    _assert_no_symlink_components(destination)
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(f"confirmation shard destination already exists: {destination}")
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{destination.name}.attempt-", dir=parent))
    try:
        resume_from = (
            Path(shard_request.resume_from) if shard_request.resume_from is not None else None
        )
        if resume_from is not None:
            if resume_from.is_symlink() or not resume_from.is_dir():
                raise ArtifactError("confirmation resume source is not a regular directory")
            _assert_disjoint_paths(resume_from, destination)
            _assert_no_symlinks_in_tree(resume_from)
            resume_manifest = _read_json(resume_from / "partial-shard-manifest.json")
            if resume_manifest.get("status") != "interrupted":
                raise ArtifactError(
                    "confirmation resume source is not an interrupted partial attempt"
                )
            if (resume_from / "shard-manifest.json").exists():
                raise ArtifactError("confirmation resume source is already a completed shard")
            if resume_manifest.get("config_hash") != confirmation_config_hash(config):
                raise ArtifactError("confirmation resume source config hash mismatch")
            if resume_manifest.get("candidate_id") != candidate.id:
                raise ArtifactError("confirmation resume source candidate mismatch")
            if resume_manifest.get("training_seed") != shard_request.training_seed:
                raise ArtifactError("confirmation resume source training seed mismatch")
        manifest = _run_confirmation_shard(
            temporary_root,
            config,
            candidate,
            shard_request.training_seed,
            deadline_seconds=shard_request.deadline_seconds,
            execution_deadline=shard_request.publish_deadline,
            resume_from=resume_from,
        )
        if (
            shard_request.publish_deadline is not None
            and time.monotonic() >= shard_request.publish_deadline
        ):
            raise ArtifactError("confirmation shard completed after the campaign publish deadline")
        if destination.exists() or destination.is_symlink():
            raise ArtifactError(
                f"confirmation shard destination appeared during publish: {destination}"
            )
        temporary_root.replace(destination)
        return {"destination": str(destination), "manifest": manifest}
    except BaseException as error:
        with suppress(OSError):
            resume_pointer = temporary_root / "run/resume-checkpoint.json"
            if resume_pointer.is_file():
                pointer = _read_json(resume_pointer)
                partial = _shard_manifest(config, candidate, shard_request.training_seed)
                partial.update(
                    {
                        "status": "interrupted",
                        "resume_checkpoint_path": "run/resume-checkpoint.json",
                        "consumed_wall_seconds": pointer.get("active_wall_seconds", 0.0),
                        "consumed_process_cpu_seconds": pointer.get("process_cpu_seconds", 0.0),
                    }
                )
                (temporary_root / "partial-shard-manifest.json").write_text(
                    canonical_json(partial) + "\n", encoding="utf-8"
                )
            (temporary_root / "failure.json").write_text(
                canonical_json(
                    {
                        "schema_version": "oi-baseline-confirmation-failure-v1",
                        "error_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        raise


def _aggregate_shard_audit(shard: Path) -> dict[str, Any]:
    manifest = _read_json(shard / "shard-manifest.json")
    records = _read_jsonl(shard / str(manifest["evaluation_record_path"]))
    if not records:
        raise ArtifactError("confirmation shard has no audit records")
    ids = [record.get("evaluation_episode_id") for record in records]
    if ids != list(range(len(records))):
        raise ArtifactError("confirmation audit episode identities are not contiguous")
    scores = [
        _finite_number(record.get("official_score"), field="official_score") for record in records
    ]
    tiles = [_finite_number(record.get("max_tile"), field="max_tile") for record in records]
    reach = sum(tile >= 256 for tile in tiles) / len(tiles)
    return {
        "candidate_id": manifest["candidate_id"],
        "training_seed": manifest["training_seed"],
        "evaluation_root_seed": manifest["evaluation_root_seed"],
        "episode_ids": ids,
        "mean_score": statistics.fmean(scores),
        "tile_reach_rate_256": reach,
        "max_tile_mean": statistics.fmean(tiles),
        "record_count": len(records),
    }


def verify_confirmation_shard(shard_directory: str | Path) -> dict[str, Any]:
    """Validate one atomically published shard without trusting its digest flag."""

    shard = Path(shard_directory)
    errors: list[str] = []
    try:
        if shard.is_symlink() or not shard.is_dir():
            raise ArtifactError("confirmation shard root is not a regular directory")
        _assert_no_symlink_components(shard)
        manifest = _read_json(shard / "shard-manifest.json")
        if manifest.get("schema_version") != "oi-baseline-confirmation-shard-v1":
            raise ArtifactError("unsupported confirmation shard schema")
        if manifest.get("status") != "completed":
            raise ArtifactError("confirmation shard is not completed")
        config = resolve_confirmation_config(_read_json(shard / "resolved-config.json"))
        candidate_id = manifest.get("candidate_id")
        seed = manifest.get("training_seed")
        if candidate_id not in CONFIRMATION_ARM_IDS:
            raise ArtifactError("shard candidate is outside the resolved matrix")
        if seed not in config.training_seeds:
            raise ArtifactError("shard training seed is outside the resolved registry")
        if manifest.get("config_hash") != confirmation_config_hash(config):
            raise ArtifactError("shard config hash mismatch")
        if manifest.get("experiment_id") != config.experiment_id:
            raise ArtifactError("shard experiment identity mismatch")
        if manifest.get("evaluation_root_seed") != config.evaluation_root_seed:
            raise ArtifactError("shard evaluation root seed mismatch")
        if manifest.get("target_training_episode") != config.training_target_episode:
            raise ArtifactError("shard training target drifted")
        if manifest.get("audit_evaluation_episodes") != config.audit_evaluation_episodes:
            raise ArtifactError("shard audit episode target drifted")
        if canonical_json(manifest.get("resource_contract")) != canonical_json(
            config.resources.to_json()
        ):
            raise ArtifactError("shard resource contract drifted")
        training_path = _path_in_shard(
            shard, manifest.get("training_record_path"), field="shard training_record_path"
        )
        evaluation_path = _path_in_shard(
            shard, manifest.get("evaluation_record_path"), field="shard evaluation_record_path"
        )
        summary_path = _path_in_shard(
            shard, manifest.get("evaluation_summary_path"), field="shard evaluation_summary_path"
        )
        training = _read_jsonl(training_path)
        if len(training) != config.training_target_episode:
            raise ArtifactError("shard training record count does not reach target")
        previous_step = 0
        previous_counters = {
            "action_value_calls": 0,
            "tuple_lookups": 0,
            "updates": 0,
            "tuple_updates": 0,
        }
        for expected_episode, record in enumerate(training):
            if record.get("candidate_id") != candidate_id:
                raise ArtifactError("shard training candidate identity mismatch")
            if record.get("training_seed") != seed or record.get("episode_id") != expected_episode:
                raise ArtifactError("shard training episode identity is not contiguous")
            steps = record.get("steps")
            global_step = record.get("global_env_step")
            if (
                type(steps) is not int
                or steps < 0
                or type(global_step) is not int
                or global_step < 0
            ):
                raise ArtifactError("shard training steps are malformed")
            if global_step != previous_step + steps:
                raise ArtifactError("shard training global step is discontinuous")
            delta = record.get("counter_delta")
            counters = record.get("counters")
            if not isinstance(delta, Mapping) or not isinstance(counters, Mapping):
                raise ArtifactError("shard training counters are missing")
            if any(type(delta.get(key)) is not int or delta[key] < 0 for key in previous_counters):
                raise ArtifactError("shard training counter deltas are malformed")
            if any(
                type(counters.get(key)) is not int or counters[key] < 0 for key in previous_counters
            ):
                raise ArtifactError("shard training counters are malformed")
            expected_counters = {
                key: previous_counters[key] + int(delta.get(key, 0)) for key in previous_counters
            }
            actual_counters = {key: int(counters.get(key, 0)) for key in previous_counters}
            if actual_counters != expected_counters:
                raise ArtifactError("shard training counters are discontinuous")
            lineage = record.get("environment_rng_lineage")
            if not isinstance(lineage, Mapping) or (
                lineage.get("root_seed") != seed
                or lineage.get("purpose") != "train-env"
                or lineage.get("environment_id") != f"{config.experiment_id}-training"
                or lineage.get("episode_id") != expected_episode
            ):
                raise ArtifactError("shard training RNG lineage mismatch")
            previous_step = global_step
            previous_counters = actual_counters
        execution = _confirmation_execution_config(
            config,
            next(candidate for candidate in config.candidates if candidate.id == candidate_id),
            cast(str, seed),
        )
        checkpoints = manifest.get("checkpoints")
        if not isinstance(checkpoints, Mapping):
            raise ArtifactError("shard checkpoints are missing")
        for checkpoint_name, checkpoint_episode in (
            ("episode_0", 0),
            ("episode_40", 40),
            ("episode_200", config.training_target_episode),
        ):
            checkpoint = checkpoints.get(checkpoint_name)
            if not isinstance(checkpoint, Mapping):
                raise ArtifactError(f"shard checkpoint is missing: {checkpoint_name}")
            step = checkpoint.get("global_env_step")
            if type(step) is not int or step < 0:
                raise ArtifactError(f"shard checkpoint step is malformed: {checkpoint_name}")
            checkpoint_dir = _path_in_shard(
                shard,
                checkpoint.get("checkpoint_directory"),
                field=f"shard {checkpoint_name} checkpoint_directory",
            )
            if not checkpoint_dir.is_dir():
                raise ArtifactError(f"shard checkpoint directory is missing: {checkpoint_name}")
            array_path = _path_in_shard(
                shard, checkpoint.get("array_path"), field=f"shard {checkpoint_name} array_path"
            )
            metadata_path = _path_in_shard(
                shard,
                checkpoint.get("metadata_path"),
                field=f"shard {checkpoint_name} metadata_path",
            )
            if (
                array_path != checkpoint_dir / f"{step}.npz"
                or metadata_path != checkpoint_dir / f"{step}.json"
            ):
                raise ArtifactError(
                    f"shard checkpoint path does not match its step: {checkpoint_name}"
                )
            if not array_path.is_file() or not metadata_path.is_file():
                raise ArtifactError(f"shard checkpoint pair is missing: {checkpoint_name}")
            agent = _build_agent(
                execution,
                cast(
                    DiscoveryArmConfig,
                    next(item for item in config.candidates if item.id == candidate_id),
                ),
            )
            snapshot = agent.restore_checkpoint(
                checkpoint_dir,
                step,
                config_hash=confirmation_config_hash(config),
            )
            if checkpoint.get("learner_state_hash") != agent.learner.state_hash():
                raise ArtifactError(f"shard checkpoint learner hash mismatch: {checkpoint_name}")
            if checkpoint.get("table_hash") != agent.learner.table_hash():
                raise ArtifactError(f"shard checkpoint table hash mismatch: {checkpoint_name}")
            if canonical_json(checkpoint.get("counters")) != canonical_json(
                agent.counters.to_json()
            ):
                raise ArtifactError(f"shard checkpoint counters mismatch: {checkpoint_name}")
            if canonical_json(checkpoint.get("environment")) != canonical_json(snapshot.to_json()):
                raise ArtifactError(f"shard checkpoint environment mismatch: {checkpoint_name}")
            if checkpoint.get("completed_training_episodes") != checkpoint_episode:
                raise ArtifactError(f"shard checkpoint episode mismatch: {checkpoint_name}")
            if checkpoint_episode == 0:
                if step != 0:
                    raise ArtifactError("episode-0 checkpoint has a non-zero global env step")
            else:
                endpoint = training[checkpoint_episode - 1]
                if (
                    checkpoint.get("global_env_step") != endpoint.get("global_env_step")
                    or checkpoint.get("learner_state_hash") != endpoint.get("learner_state_hash")
                    or canonical_json(checkpoint.get("counters"))
                    != canonical_json(endpoint.get("counters"))
                ):
                    raise ArtifactError(
                        f"shard checkpoint does not match training endpoint: {checkpoint_name}"
                    )
                environment = checkpoint.get("environment")
                if not isinstance(environment, Mapping):
                    raise ArtifactError(
                        f"shard checkpoint environment is missing: {checkpoint_name}"
                    )
                rng = environment.get("rng")
                lineage = rng.get("lineage") if isinstance(rng, Mapping) else None
                if (
                    not isinstance(lineage, Mapping)
                    or lineage.get("episode_id") != checkpoint_episode - 1
                ):
                    raise ArtifactError(
                        f"shard checkpoint environment episode mismatch: {checkpoint_name}"
                    )
            if checkpoint_episode == config.training_target_episode:
                if manifest.get("learner_state_hash") != checkpoint.get("learner_state_hash"):
                    raise ArtifactError(
                        "shard final learner hash drifted from episode-200 checkpoint"
                    )
                if manifest.get("table_hash") != checkpoint.get("table_hash"):
                    raise ArtifactError(
                        "shard final table hash drifted from episode-200 checkpoint"
                    )
                expected_lineage = (
                    checkpoint.get("environment", {}).get("rng", {}).get("lineage")
                    if isinstance(checkpoint.get("environment"), Mapping)
                    and isinstance(checkpoint.get("environment", {}).get("rng"), Mapping)
                    else None
                )
                if manifest.get("rng_lineage") != expected_lineage:
                    raise ArtifactError(
                        "shard final RNG lineage drifted from episode-200 checkpoint"
                    )
        evaluation = _read_jsonl(evaluation_path)
        if len(evaluation) != config.audit_evaluation_episodes:
            raise ArtifactError("shard audit record count does not reach target")
        for expected_episode_id, record in enumerate(evaluation):
            if (
                record.get("candidate_id") != candidate_id
                or record.get("training_seed") != seed
                or record.get("evaluation_root_seed") != config.evaluation_root_seed
                or record.get("checkpoint_episode") != config.training_target_episode
                or record.get("evaluation_episode_id") != expected_episode_id
            ):
                raise ArtifactError("shard evaluation identity is not contiguous")
            if record.get("frozen_state_unchanged") is not True:
                raise ArtifactError("shard frozen evaluation changed learner state")
            if (
                _finite_number(record.get("official_score"), field="evaluation.official_score")
                < 0.0
            ):
                raise ArtifactError("shard evaluation score is negative")
            max_tile = record.get("max_tile")
            steps = record.get("steps")
            if type(max_tile) is not int or max_tile < 0 or type(steps) is not int or steps < 0:
                raise ArtifactError("shard evaluation tile or steps are malformed")
        evaluation_summary = _read_json(summary_path)
        if (
            evaluation_summary.get("candidate_id") != candidate_id
            or evaluation_summary.get("training_seed") != seed
            or evaluation_summary.get("evaluation_root_seed") != config.evaluation_root_seed
            or evaluation_summary.get("checkpoint_episode") != config.training_target_episode
            or evaluation_summary.get("requested_episodes") != config.audit_evaluation_episodes
            or evaluation_summary.get("completed_episodes") != config.audit_evaluation_episodes
        ):
            raise ArtifactError("shard evaluation summary identity drifted")
        aggregate = _aggregate_shard_audit(shard)
        score_summary = evaluation_summary.get("official_score")
        tile_summary = evaluation_summary.get("tile_reach_rate")
        if (
            not isinstance(score_summary, Mapping)
            or score_summary.get("mean") != aggregate["mean_score"]
            or not isinstance(tile_summary, Mapping)
            or tile_summary.get("256") != aggregate["tile_reach_rate_256"]
            or evaluation_summary.get("max_tile_mean") != aggregate["max_tile_mean"]
        ):
            raise ArtifactError("shard evaluation summary does not match raw records")
        telemetry = manifest.get("runtime_telemetry")
        if not isinstance(telemetry, Mapping):
            raise ArtifactError("shard runtime telemetry is missing")
        validate_runtime_telemetry(telemetry, budget_seconds=config.campaign_wall_seconds)
        expected_digest = _shard_scientific_digest(manifest, training, evaluation)
        if manifest.get("scientific_digest") != expected_digest:
            raise ArtifactError("shard scientific digest mismatch")
    except (
        ArtifactError,
        KeyError,
        StopIteration,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        errors.append(str(error))
    return {
        "schema_version": "oi-baseline-confirmation-shard-verification-v1",
        "valid": not errors,
        "errors": errors,
        "shard_directory": str(shard),
    }


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")
    temporary.replace(path)


def _git_provenance() -> dict[str, Any]:
    repository_root = Path(__file__).parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repository_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "unknown", "dirty": None}


def _append_coordinator_progress(root: Path, event: str, **fields: object) -> None:
    path = root / "campaign-progress.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            canonical_json(
                {"schema_version": "oi-baseline-confirmation-progress-v1", "event": event, **fields}
            )
            + "\n"
        )


def _paired_audit_record(seed: str, aggregates: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    zero = aggregates.get("td0_zero")
    oi = aggregates.get("td0_oi_1000")
    if zero is None or oi is None:
        raise ArtifactError(f"paired audit is incomplete for training seed: {seed}")
    if zero["episode_ids"] != oi["episode_ids"]:
        raise ArtifactError(f"paired audit episode identities differ for training seed: {seed}")
    return {
        "training_seed": seed,
        "zero_mean_score": zero["mean_score"],
        "oi_mean_score": oi["mean_score"],
        "zero_256_reach_rate": zero["tile_reach_rate_256"],
        "oi_256_reach_rate": oi["tile_reach_rate_256"],
        "zero_max_tile_mean": zero["max_tile_mean"],
        "oi_max_tile_mean": oi["max_tile_mean"],
        "evaluation_episode_ids": zero["episode_ids"],
    }


def _campaign_shard_requests(
    config: ConfirmationConfig,
    source_root: Path,
    seeds: Sequence[str],
    remaining_seconds: float,
    *,
    publish_deadline: float | None = None,
) -> list[ConfirmationShardRequest]:
    requests: list[ConfirmationShardRequest] = []
    for seed in seeds:
        for candidate in config.candidates:
            destination = source_root / "shards" / seed / candidate.id
            requests.append(
                ConfirmationShardRequest(
                    config=config.to_json(),
                    candidate_id=candidate.id,
                    training_seed=seed,
                    destination=str(destination),
                    deadline_seconds=max(1.0, remaining_seconds),
                    publish_deadline=publish_deadline,
                )
            )
    return requests


def _run_shard_requests(
    requests: Sequence[ConfirmationShardRequest],
    *,
    worker_count: int,
    shard_runner: Any = None,
    deadline: float | None = None,
    clock: Any = time.monotonic,
) -> list[dict[str, Any]]:
    if shard_runner is not None:
        results: list[dict[str, Any]] = []
        for request in requests:
            if deadline is not None and clock() >= deadline:
                raise ArtifactError(
                    "confirmation campaign wall budget exhausted before shard completion"
                )
            result = shard_runner(request)
            if deadline is not None and clock() >= deadline:
                raise ArtifactError(
                    "confirmation campaign wall budget exhausted after shard completion"
                )
            if not isinstance(result, dict):
                raise ArtifactError("confirmation shard runner returned a non-object result")
            results.append(result)
        return results
    apply_thread_environment()
    context = multiprocessing.get_context("spawn")
    results = []
    pool = ProcessPoolExecutor(max_workers=worker_count, mp_context=context)
    pending: set[Any] = set()
    try:
        futures = {pool.submit(run_confirmation_shard, request): request for request in requests}
        pending = set(futures)
        while pending:
            timeout = None if deadline is None else max(0.0, deadline - clock())
            if timeout is not None and timeout <= 0.0:
                raise ArtifactError(
                    "confirmation campaign wall budget exhausted before shard completion"
                )
            done, pending = wait(pending, timeout=timeout, return_when=FIRST_EXCEPTION)
            if not done:
                raise ArtifactError(
                    "confirmation campaign wall budget exhausted before shard completion"
                )
            if deadline is not None and clock() >= deadline:
                raise ArtifactError(
                    "confirmation campaign wall budget exhausted after shard completion"
                )
            for future in done:
                result = future.result()
                if not isinstance(result, dict):
                    raise ArtifactError("confirmation worker returned a non-object result")
                results.append(result)
    except BaseException:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    return results


def run_confirmation_campaign(
    config: str | Path | Mapping[str, Any] | ConfirmationConfig,
    *,
    artifact_directory: str | Path | None = None,
    max_workers: int | None = None,
    shard_runner: Any = None,
    clock: Any = time.monotonic,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Run predeclared cohorts with coordinator-owned deterministic reduction.

    ``shard_runner`` is an internal deterministic fixture seam.  Production
    callers leave it unset so work is isolated in a spawned process pool; unit
    tests can provide a bounded fake without changing the scientific reducer.
    """

    if isinstance(config, (str, Path)):
        resolved = load_confirmation_config(config)
    elif isinstance(config, ConfirmationConfig):
        resolved = resolve_confirmation_config(config.to_json())
    else:
        resolved = resolve_confirmation_config(config)
    if deadline is not None and (
        isinstance(deadline, bool)
        or not isinstance(deadline, (int, float))
        or not math.isfinite(float(deadline))
    ):
        raise ArtifactError("confirmation campaign deadline must be finite")
    worker_count = resolved.resources.worker_count
    if max_workers is not None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ArtifactError("confirmation worker_count must be a positive integer")
        if max_workers != worker_count:
            raise ArtifactError(
                "confirmation worker_count override drifts from the frozen resource contract"
            )
    root = (
        Path(artifact_directory)
        if artifact_directory is not None
        else Path(resolved.output_root) / resolved.experiment_id
    )
    if any(part in {"", ".", ".."} for part in root.parts):
        raise ArtifactError("confirmation campaign destination contains traversal")
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ArtifactError("confirmation campaign destination contains a symlink component")
    if root.exists() and (root.is_symlink() or any(root.iterdir())):
        raise ArtifactError(f"confirmation campaign destination is not empty: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source_root = root / "source"
    source_root.mkdir(parents=True, exist_ok=True)
    resolved_json = resolved.to_json()
    resolved_hash = confirmation_config_hash(resolved)
    campaign_manifest = {
        "schema_version": "oi-baseline-confirmation-campaign-v1",
        "campaign_id": root.name,
        "experiment_id": resolved.experiment_id,
        "config_hash": resolved_hash,
        "fresh_seed_registry": list(resolved.training_seeds),
        "legacy_seed_denylist": list(resolved.legacy_seed_denylist),
        "cohort_order": [
            list(resolved.training_seeds[index : index + resolved.cohort_size])
            for index in range(0, len(resolved.training_seeds), resolved.cohort_size)
        ],
        "resource_contract": resolved.resources.to_json(),
        "source_provenance": _git_provenance(),
        "campaign_wall_seconds": resolved.campaign_wall_seconds,
        "status": "running",
    }
    _write_atomic_json(root / "campaign-manifest.json", campaign_manifest)
    _write_atomic_json(source_root / "resolved-config.json", resolved_json)
    started = clock()
    hard_deadline = (
        float(deadline) if deadline is not None else started + resolved.campaign_wall_seconds
    )
    process_started = time.process_time()
    paired_records: list[dict[str, Any]] = []
    cohort_records: list[dict[str, Any]] = []
    worker_runtime_records: list[dict[str, Any]] = []
    stop_reason = "campaign_completed"
    gate: ConfirmationGate = "continue"
    try:
        for cohort_index, offset in enumerate(
            range(0, resolved.maximum_fresh_seeds, resolved.cohort_size), 1
        ):
            cohort_seeds = resolved.training_seeds[offset : offset + resolved.cohort_size]
            if len(cohort_seeds) != resolved.cohort_size:
                raise ArtifactError("confirmation cohort registry is not complete")
            remaining = hard_deadline - clock()
            if remaining <= resolved.finalization_reserve_seconds:
                gate = "performance-blocked"
                stop_reason = "campaign_wall_budget_before_cohort"
                break
            work_remaining = remaining - resolved.finalization_reserve_seconds
            _append_coordinator_progress(
                root,
                "cohort_started",
                cohort_index=cohort_index,
                training_seeds=list(cohort_seeds),
            )
            work_deadline = hard_deadline - resolved.finalization_reserve_seconds
            requests = _campaign_shard_requests(
                resolved,
                source_root,
                cohort_seeds,
                work_remaining,
                publish_deadline=work_deadline,
            )
            shard_results = _run_shard_requests(
                requests,
                worker_count=worker_count,
                shard_runner=shard_runner,
                deadline=work_deadline,
                clock=clock,
            )
            for result in shard_results:
                manifest = result.get("manifest")
                if not isinstance(manifest, Mapping):
                    raise ArtifactError("confirmation worker result is missing its manifest")
                runtime = manifest.get("runtime_telemetry")
                if not isinstance(runtime, Mapping):
                    raise ArtifactError("confirmation worker result is missing runtime telemetry")
                worker_runtime_records.append(
                    {
                        "candidate_id": manifest.get("candidate_id"),
                        "training_seed": manifest.get("training_seed"),
                        "runtime_telemetry": validate_runtime_telemetry(
                            runtime, budget_seconds=resolved.campaign_wall_seconds
                        ),
                    }
                )
            cohort_aggregates: dict[str, dict[str, Mapping[str, Any]]] = {}
            for seed in cohort_seeds:
                by_candidate: dict[str, Mapping[str, Any]] = {}
                for candidate in resolved.candidates:
                    shard = source_root / "shards" / seed / candidate.id
                    report = verify_confirmation_shard(shard)
                    if report["valid"] is not True:
                        raise ArtifactError(
                            f"confirmation shard verification failed: {seed}/{candidate.id}: "
                            + "; ".join(cast(list[str], report["errors"]))
                        )
                    by_candidate[candidate.id] = _aggregate_shard_audit(shard)
                cohort_aggregates[seed] = by_candidate
                paired_records.append(_paired_audit_record(seed, by_candidate))
            gate_result = compute_confirmation_gate(resolved, paired_records)
            gate = cast(ConfirmationGate, gate_result["decision"])
            cohort_record = {
                "schema_version": "oi-baseline-confirmation-cohort-v1",
                "cohort_index": cohort_index,
                "training_seeds": list(cohort_seeds),
                "paired_records": [
                    item for item in paired_records if item["training_seed"] in cohort_seeds
                ],
                "gate": gate_result,
            }
            cohort_records.append(cohort_record)
            _write_atomic_json(source_root / "cohorts" / f"{cohort_index:04d}.json", cohort_record)
            _append_coordinator_progress(
                root,
                "cohort_completed",
                cohort_index=cohort_index,
                decision=gate,
                fresh_seed_count=len(paired_records),
            )
            if gate in {"oi-baseline-confirmed", "oi-baseline-rejected", "inconclusive"}:
                stop_reason = cast(str, gate_result["stop_reason"])
                break
            gate = "continue"
        else:
            if gate == "continue":
                gate = "inconclusive"
                stop_reason = "maximum_fresh_seeds_reached_without_gate"
    except Exception as error:
        message = str(error)
        if "deadline" in message or "wall budget" in message or "timed out" in message:
            gate = "performance-blocked"
            stop_reason = message
        else:
            gate = "contract-failed"
            stop_reason = message
    consumed = max(0.0, clock() - started)
    if consumed > resolved.campaign_wall_seconds and gate not in {"contract-failed"}:
        gate = "performance-blocked"
        stop_reason = "campaign_wall_budget_exhausted"
    _append_coordinator_progress(
        root,
        "campaign_stopped",
        gate=gate,
        stop_reason=stop_reason,
        fresh_seed_count=len(paired_records),
        wall_seconds=consumed,
    )
    summary = {
        "schema_version": "oi-baseline-confirmation-source-summary-v1",
        "campaign_id": root.name,
        "experiment_id": resolved.experiment_id,
        "config_hash": resolved_hash,
        "gate": gate,
        "stop_reason": stop_reason,
        "fresh_seed_count": len(paired_records),
        "paired_records": paired_records,
        "cohorts": cohort_records,
        "runtime_telemetry": {
            "wall_seconds": consumed,
            "process_cpu_seconds": max(0.0, time.process_time() - process_started),
            "worker_count": worker_count,
            "worker_wall_seconds": sum(
                float(cast(Mapping[str, Any], item["runtime_telemetry"]).get("wall_seconds", 0.0))
                for item in worker_runtime_records
            ),
            "worker_process_cpu_seconds": sum(
                float(
                    cast(Mapping[str, Any], item["runtime_telemetry"]).get(
                        "process_cpu_seconds", 0.0
                    )
                )
                for item in worker_runtime_records
            ),
        },
        "worker_runtime_telemetry": sorted(
            worker_runtime_records,
            key=lambda item: (str(item.get("training_seed")), str(item.get("candidate_id"))),
        ),
        "evidence_boundary": {
            "statistical_significance_claimed": False,
            "old_calibration_seeds_included": False,
            "search_or_curriculum_used": False,
        },
    }
    _write_atomic_json(source_root / "source-summary.json", summary)
    try:
        from strategy2048.experiments.confirmation_contract import verify_confirmation_source

        source_verification = verify_confirmation_source(source_root)
    except Exception as error:
        source_verification = {
            "schema_version": "oi-baseline-confirmation-source-verification-v1",
            "valid": False,
            "gate": "contract-failed",
            "errors": [str(error)],
            "source_directory": str(source_root),
        }
    _write_atomic_json(source_root / "source-verification.json", source_verification)
    if source_verification.get("valid") is not True:
        if summary.get("gate") == "performance-blocked":
            summary["stop_reason"] = "performance_blocked_with_partial_source_verification"
        else:
            summary["gate"] = "contract-failed"
            summary["stop_reason"] = "source_verification_failed"
        _write_atomic_json(source_root / "source-summary.json", summary)
        gate = cast(ConfirmationGate, summary["gate"])
    campaign_manifest["status"] = gate
    campaign_manifest["finished_wall_seconds"] = consumed
    _write_atomic_json(root / "campaign-manifest.json", campaign_manifest)
    return summary


def _formal_ledger_path(root: Path) -> Path:
    return root / "campaign-ledger.jsonl"


def _read_formal_ledger(root: Path) -> list[dict[str, Any]]:
    path = _formal_ledger_path(root)
    if not path.exists():
        return []
    return _read_jsonl(path)


def _append_formal_ledger(root: Path, event: str, **fields: object) -> dict[str, Any]:
    record = {
        "schema_version": FORMAL_CONFIRMATION_LEDGER_SCHEMA_VERSION,
        "event": event,
        **fields,
    }
    path = _formal_ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(record) + "\n")
        handle.flush()
        with suppress(OSError):
            os.fsync(handle.fileno())
    return record


def _formal_latest_phase_event(
    events: Sequence[Mapping[str, Any]], phase: str
) -> Mapping[str, Any] | None:
    latest: Mapping[str, Any] | None = None
    for event in events:
        if event.get("phase") == phase:
            latest = event
    return latest


def _formal_consumed_wall_seconds(events: Sequence[Mapping[str, Any]]) -> float:
    previous = 0.0
    for event in events:
        raw = event.get("consumed_wall_seconds")
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            continue
        value = float(raw)
        if not math.isfinite(value) or value < 0.0:
            raise ArtifactError("formal campaign ledger consumed wall time is invalid")
        if value + 1e-9 < previous:
            raise ArtifactError("formal campaign ledger wall time is not monotonic")
        previous = value
    return previous


def _formal_attempt_directory(root: Path, phase: str) -> Path:
    attempts = root / "attempts"
    _assert_no_symlink_components(attempts)
    attempts.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f"{phase}-", dir=attempts))


def _formal_promote_directory(attempt: Path, destination: Path) -> None:
    _assert_no_symlink_components(attempt)
    _assert_no_symlink_components(destination)
    if not attempt.is_dir() or attempt.is_symlink():
        raise ArtifactError(f"formal campaign phase output is not a regular directory: {attempt}")
    if destination.exists() or destination.is_symlink():
        raise ArtifactError(f"formal campaign sibling destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt.replace(destination)


def _formal_promote_source_attempt(attempt: Path, root: Path) -> Path:
    source_attempt = attempt / "source"
    source_destination = root / "source"
    _formal_promote_directory(source_attempt, source_destination)
    for name in ("campaign-manifest.json", "campaign-progress.jsonl"):
        source_file = attempt / name
        destination = root / name
        if source_file.is_file() and not destination.exists():
            source_file.replace(destination)
    return source_destination


def _formal_source_summary(source: Path) -> dict[str, Any]:
    summary = _read_json(source / "source-summary.json")
    gate = summary.get("gate")
    if gate not in {
        "oi-baseline-confirmed",
        "oi-baseline-rejected",
        "inconclusive",
    }:
        raise ArtifactError("formal confirmation source is not a complete decision")
    return summary


def _formal_phase_artifact(events: Sequence[Mapping[str, Any]], phase: str) -> Path | None:
    event = _formal_latest_phase_event(events, phase)
    if event is None:
        return None
    event_kind = event.get("event")
    if event_kind in {"phase_completed", "phase_recovered"}:
        value = event.get("artifact_directory")
    elif event_kind == "phase_succeeded":
        value = event.get("attempt_directory")
    else:
        return None
    return Path(value) if isinstance(value, str) else None


def _formal_result(
    root: Path,
    resolved: ConfirmationConfig,
    *,
    status: str,
    gate: str,
    stop_reason: str,
    consumed_wall_seconds: float,
    events: Sequence[Mapping[str, Any]],
    source_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    phases: dict[str, str] = {}
    artifacts: dict[str, str | None] = {}
    for phase in FORMAL_CONFIRMATION_PHASES:
        latest = _formal_latest_phase_event(events, phase)
        phases[phase] = str(latest.get("event", "pending")) if latest is not None else "pending"
        artifact = _formal_phase_artifact(events, phase)
        artifacts[phase] = str(artifact) if artifact is not None else None
    result = {
        "schema_version": FORMAL_CONFIRMATION_SCHEMA_VERSION,
        "campaign_id": root.name,
        "experiment_id": resolved.experiment_id,
        "config_hash": confirmation_config_hash(resolved),
        "status": status,
        "gate": gate,
        "stop_reason": stop_reason,
        "campaign_wall_seconds": resolved.campaign_wall_seconds,
        "finalization_reserve_seconds": resolved.finalization_reserve_seconds,
        "consumed_wall_seconds": consumed_wall_seconds,
        "remaining_wall_seconds": max(0.0, resolved.campaign_wall_seconds - consumed_wall_seconds),
        "phases": phases,
        "artifacts": artifacts,
        "ledger_path": str(_formal_ledger_path(root)),
        "source_summary": dict(source_summary) if source_summary is not None else None,
    }
    _write_atomic_json(root / "formal-campaign-summary.json", result)
    return result


def _mapping_runner_entry(connection: Any, runner: Any, kwargs: Mapping[str, Any]) -> None:
    """Run one pickle-safe mapping-returning callable in a spawned child."""

    try:
        apply_thread_environment()
        result = runner(**dict(kwargs))
        connection.send({"ok": True, "result": result})
    except BaseException as error:
        with suppress(OSError):
            connection.send({"ok": False, "error": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _run_mapping_with_watchdog(
    runner: Any,
    *,
    kwargs: Mapping[str, Any],
    timeout_seconds: float,
) -> tuple[Mapping[str, Any] | None, str | None, bool]:
    """Run ``runner`` under a hard spawn-process timeout.

    Returns ``(result, error, timed_out)``.  A timed-out child is terminated,
    then killed if necessary, and joined before control returns.
    """

    if timeout_seconds <= 0.0 or not math.isfinite(timeout_seconds):
        return None, "runner deadline reached before process start", True
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_mapping_runner_entry,
        args=(child_connection, runner, dict(kwargs)),
    )
    started = time.monotonic()
    try:
        process.start()
    except BaseException as start_error:
        child_connection.close()
        parent_connection.close()
        return (
            None,
            f"runner could not start: {type(start_error).__name__}: {start_error}",
            False,
        )
    child_connection.close()
    try:
        process.join(max(0.0, timeout_seconds - (time.monotonic() - started)))
        if process.is_alive():
            process.terminate()
            process.join(min(1.0, max(0.1, timeout_seconds)))
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(1.0)
            if process.is_alive():
                return None, "runner timed out and could not be reaped", True
            return None, "runner timed out and was terminated", True
        if not parent_connection.poll(0.2):
            return None, f"runner exited without a result (exitcode={process.exitcode})", False
        envelope = parent_connection.recv()
        if not isinstance(envelope, Mapping) or envelope.get("ok") is not True:
            runner_error = (
                str(envelope.get("error", "runner failed"))
                if isinstance(envelope, Mapping)
                else "runner failed"
            )
            return None, runner_error, False
        result = envelope.get("result")
        if not isinstance(result, Mapping):
            return None, "runner returned a non-object result", False
        return result, None, False
    finally:
        parent_connection.close()


def run_confirmation_formal_campaign(
    config: str | Path | Mapping[str, Any] | ConfirmationConfig,
    *,
    artifact_directory: str | Path | None = None,
    max_workers: int | None = None,
    shard_runner: Any = None,
    reducer_commit: str | None = None,
    reducer_dirty: bool = False,
    replay_workers: int = 1,
    source_runner: Any = None,
    source_verifier: Any = None,
    contract_runner: Any = None,
    checker_runner: Any = None,
    clock: Any = time.monotonic,
    resume: bool = False,
) -> dict[str, Any]:
    """Run source, verification, bundle, and checker under one resumable ledger.

    The default phase runners are the production confirmation APIs.  The runner
    hooks are intentionally injectable for bounded contract fixtures; every hook
    still has to produce the expected durable artifact before the coordinator
    promotes it to the canonical sibling.
    """

    if isinstance(config, (str, Path)):
        resolved = load_confirmation_config(config)
    elif isinstance(config, ConfirmationConfig):
        resolved = resolve_confirmation_config(config.to_json())
    else:
        resolved = resolve_confirmation_config(config)
    if type(replay_workers) is not int or replay_workers <= 0:
        raise ArtifactError("formal confirmation replay worker count must be positive")
    worker_count = resolved.resources.worker_count
    if max_workers is not None:
        if type(max_workers) is not int or max_workers <= 0:
            raise ArtifactError("confirmation worker_count must be a positive integer")
        if max_workers != worker_count:
            raise ArtifactError(
                "confirmation worker_count override drifts from the frozen resource contract"
            )
    root = (
        Path(artifact_directory)
        if artifact_directory is not None
        else Path(resolved.output_root) / f"{resolved.experiment_id}-formal"
    )
    if any(part in {"", ".", ".."} for part in root.parts):
        raise ArtifactError("formal confirmation destination contains traversal")
    _assert_no_symlink_components(root)
    if root.exists() and not root.is_dir():
        raise ArtifactError("formal confirmation destination is not a directory")
    root.mkdir(parents=True, exist_ok=True)
    events = _read_formal_ledger(root)
    resolved_hash = confirmation_config_hash(resolved)
    if not events:
        _append_formal_ledger(
            root,
            "campaign_started",
            campaign_id=root.name,
            experiment_id=resolved.experiment_id,
            config_hash=resolved_hash,
            campaign_wall_seconds=resolved.campaign_wall_seconds,
            finalization_reserve_seconds=resolved.finalization_reserve_seconds,
            consumed_wall_seconds=0.0,
        )
        events = _read_formal_ledger(root)
    else:
        completed_summary = root / "formal-campaign-summary.json"
        if completed_summary.is_file():
            stored_summary = _read_json(completed_summary)
            if stored_summary.get("status") == "completed":
                return stored_summary
        if not resume:
            raise ArtifactError("existing formal campaign requires explicit resume")
        first = events[0]
        if (
            first.get("event") != "campaign_started"
            or first.get("config_hash") != resolved_hash
            or first.get("campaign_wall_seconds") != resolved.campaign_wall_seconds
        ):
            raise ArtifactError("formal campaign ledger does not match the resolved config")
        _append_formal_ledger(
            root,
            "campaign_resumed",
            campaign_id=root.name,
            consumed_wall_seconds=_formal_consumed_wall_seconds(events),
        )
        events = _read_formal_ledger(root)
    for phase in FORMAL_CONFIRMATION_PHASES:
        latest = _formal_latest_phase_event(events, phase)
        if latest is not None and latest.get("event") == "phase_started":
            prior = _formal_consumed_wall_seconds(events)
            allocated = latest.get("allocated_wall_seconds", 0.0)
            if isinstance(allocated, bool) or not isinstance(allocated, (int, float)):
                allocated = 0.0
            consumed = min(
                resolved.campaign_wall_seconds,
                prior + max(0.0, float(allocated)),
            )
            _append_formal_ledger(
                root,
                "phase_failed",
                phase=phase,
                attempt_directory=latest.get("attempt_directory"),
                phase_wall_seconds=max(0.0, consumed - prior),
                consumed_wall_seconds=consumed,
                error=f"previous invocation ended during {phase}",
            )
            events = _read_formal_ledger(root)
    prior_consumed = _formal_consumed_wall_seconds(events)
    invocation_started = clock()
    invocation_deadline = invocation_started + max(
        0.0, resolved.campaign_wall_seconds - prior_consumed
    )
    data_deadline = invocation_deadline - resolved.finalization_reserve_seconds
    source_summary: Mapping[str, Any] | None = None
    source_runner = source_runner or run_confirmation_campaign
    if source_verifier is None:
        from strategy2048.experiments.confirmation_contract import verify_confirmation_source

        source_verifier = verify_confirmation_source
    if contract_runner is None:
        from strategy2048.experiments.confirmation_contract import build_confirmation_contract

        contract_runner = build_confirmation_contract
    if checker_runner is None:
        from strategy2048.experiments.confirmation_contract import write_confirmation_checker_report

        checker_runner = write_confirmation_checker_report
    if reducer_commit is None:
        reducer_commit = cast(str, _git_provenance().get("commit"))
    if (
        not isinstance(reducer_commit, str)
        or re.fullmatch(r"[0-9a-f]{7,64}", reducer_commit) is None
    ):
        raise ArtifactError("formal confirmation reducer commit is missing or malformed")

    def stop(status: str, gate: str, reason: str, consumed: float) -> dict[str, Any]:
        nonlocal events
        _append_formal_ledger(
            root,
            "campaign_stopped",
            status=status,
            gate=gate,
            stop_reason=reason,
            consumed_wall_seconds=consumed,
        )
        events = _read_formal_ledger(root)
        return _formal_result(
            root,
            resolved,
            status=status,
            gate=gate,
            stop_reason=reason,
            consumed_wall_seconds=consumed,
            events=events,
            source_summary=source_summary,
        )

    def phase_run(
        phase: str,
        *,
        deadline: float,
        runner: Any,
        kwargs: Mapping[str, Any],
        output_argument: str | None = None,
        output_subdirectory: str | None = None,
        deadline_argument: str | None = None,
    ) -> tuple[Mapping[str, Any] | None, Path | None, float, str | None, bool]:
        nonlocal events
        latest = _formal_latest_phase_event(events, phase)
        if latest is not None and latest.get("event") in {"phase_completed", "phase_recovered"}:
            artifact = _formal_phase_artifact(events, phase)
            return None, artifact, 0.0, None, False
        if clock() >= deadline:
            return None, None, 0.0, "phase deadline reached before work started", True
        attempt = _formal_attempt_directory(root, phase)
        phase_started = clock()
        prior = _formal_consumed_wall_seconds(events)
        timeout_seconds = max(0.0, deadline - phase_started)
        call_kwargs = dict(kwargs)
        if output_argument is not None:
            output = attempt if output_subdirectory is None else attempt / output_subdirectory
            call_kwargs[output_argument] = str(output)
        if deadline_argument is not None:
            call_kwargs[deadline_argument] = deadline
        _append_formal_ledger(
            root,
            "phase_started",
            phase=phase,
            attempt_directory=str(attempt),
            allocated_wall_seconds=timeout_seconds,
            consumed_wall_seconds=prior,
        )
        events = _read_formal_ledger(root)
        result, error, timed_out = _run_mapping_with_watchdog(
            runner,
            kwargs=call_kwargs,
            timeout_seconds=timeout_seconds,
        )
        elapsed = max(0.0, clock() - phase_started)
        charged = timeout_seconds if timed_out else elapsed
        consumed = min(resolved.campaign_wall_seconds, prior + charged)
        if error is None and clock() >= deadline:
            error = f"formal {phase} exceeded its monotonic deadline"
            timed_out = True
            consumed = min(resolved.campaign_wall_seconds, prior + timeout_seconds)
        if error is None:
            _append_formal_ledger(
                root,
                "phase_succeeded",
                phase=phase,
                attempt_directory=str(attempt),
                phase_wall_seconds=elapsed,
                consumed_wall_seconds=consumed,
            )
            events = _read_formal_ledger(root)
            return result, attempt, elapsed, None, False
        _append_formal_ledger(
            root,
            "phase_failed",
            phase=phase,
            attempt_directory=str(attempt),
            phase_wall_seconds=charged,
            consumed_wall_seconds=consumed,
            error=error,
            timed_out=timed_out,
        )
        events = _read_formal_ledger(root)
        return None, attempt, charged, error, timed_out

    source_runner_event = _formal_latest_phase_event(events, "source-runner")
    source_attempt = _formal_phase_artifact(events, "source-runner")
    if source_runner_event is None or source_runner_event.get("event") == "phase_failed":
        result, source_attempt, elapsed, phase_error, timed_out = phase_run(
            "source-runner",
            deadline=data_deadline,
            runner=source_runner,
            kwargs={
                "config": resolved,
                "max_workers": worker_count,
                "shard_runner": shard_runner,
            },
            output_argument="artifact_directory",
            deadline_argument="deadline",
        )
        consumed = _formal_consumed_wall_seconds(events)
        if phase_error is not None or source_attempt is None:
            reason = phase_error or "source runner did not produce an attempt directory"
            status = (
                "performance-blocked"
                if timed_out or clock() >= data_deadline
                else "contract-failed"
            )
            return stop(status, status, reason, consumed)
        events = _read_formal_ledger(root)
    elif source_attempt is None:
        return stop(
            "contract-failed",
            "contract-failed",
            "source runner ledger artifact is missing",
            prior_consumed,
        )

    source_verification_event = _formal_latest_phase_event(events, "source-verification")
    source_canonical = root / "source"
    if source_verification_event is None or source_verification_event.get("event") not in {
        "phase_completed",
        "phase_recovered",
    }:
        verification_result, _, elapsed, phase_error, timed_out = phase_run(
            "source-verification",
            deadline=data_deadline,
            runner=source_verifier,
            kwargs={"source_directory": source_attempt / "source"},
        )
        consumed = _formal_consumed_wall_seconds(events)
        if phase_error is not None or verification_result is None:
            status = "performance-blocked" if timed_out else "contract-failed"
            return stop(
                status,
                status,
                phase_error or "source verification returned no result",
                consumed,
            )
        if verification_result.get("valid") is not True:
            return stop(
                "contract-failed",
                "contract-failed",
                "source verification rejected the source",
                consumed,
            )
        try:
            summary = _formal_source_summary(source_attempt / "source")
            _write_atomic_json(
                source_attempt / "source" / "source-verification.json", verification_result
            )
            source_canonical = _formal_promote_source_attempt(source_attempt, root)
            source_summary = summary
            _append_formal_ledger(
                root,
                "phase_completed",
                phase="source-runner",
                artifact_directory=str(source_canonical),
                phase_wall_seconds=0.0,
                consumed_wall_seconds=consumed,
            )
            _append_formal_ledger(
                root,
                "phase_completed",
                phase="source-verification",
                artifact_directory=str(source_canonical),
                phase_wall_seconds=elapsed,
                consumed_wall_seconds=consumed,
            )
            events = _read_formal_ledger(root)
        except (ArtifactError, OSError, ValueError) as error:
            return stop("contract-failed", "contract-failed", str(error), consumed)
    else:
        source_canonical = _formal_phase_artifact(events, "source-verification") or source_canonical
        if not source_canonical.is_dir():
            return stop(
                "contract-failed",
                "contract-failed",
                "completed source artifact is missing",
                _formal_consumed_wall_seconds(events),
            )
        source_summary = _formal_source_summary(source_canonical)

    if source_summary is None:
        source_summary = _formal_source_summary(source_canonical)
    from strategy2048.experiments.confirmation_contract import artifact_tree_sha256

    source_digest_before = artifact_tree_sha256(source_canonical)
    if _formal_consumed_wall_seconds(events) >= resolved.campaign_wall_seconds:
        return stop(
            "performance-blocked",
            "performance-blocked",
            "campaign wall budget exhausted before derived contract",
            _formal_consumed_wall_seconds(events),
        )

    derived_event = _formal_latest_phase_event(events, "derived-contract")
    derived_canonical = root / "derived-contract"
    if derived_event is None or derived_event.get("event") not in {
        "phase_completed",
        "phase_recovered",
    }:
        result, attempt, elapsed, phase_error, timed_out = phase_run(
            "derived-contract",
            deadline=data_deadline,
            runner=contract_runner,
            kwargs={
                "source_directory": source_canonical,
                "reducer_commit": reducer_commit,
                "reducer_dirty": reducer_dirty,
                "replay_workers": replay_workers,
            },
            output_argument="destination",
            output_subdirectory="output",
        )
        consumed = _formal_consumed_wall_seconds(events)
        if phase_error is not None or attempt is None:
            status = "performance-blocked" if timed_out else "contract-failed"
            return stop(
                status,
                status,
                phase_error or "derived contract runner returned no result",
                consumed,
            )
        contract_output = attempt / "output"
        contract_path = contract_output / "confirmation-contract.json"
        if result is None or not contract_path.is_file():
            return stop(
                "contract-failed",
                "contract-failed",
                "derived contract output is incomplete",
                consumed,
            )
        lineage = result.get("lineage_proof")
        if (
            not isinstance(lineage, Mapping)
            or lineage.get("all_confirm_lineages_verified") is not True
        ):
            return stop(
                "contract-failed",
                "contract-failed",
                "derived contract replay is incomplete",
                consumed,
            )
        try:
            _formal_promote_directory(contract_output, derived_canonical)
        except (ArtifactError, OSError) as error:
            return stop("contract-failed", "contract-failed", str(error), consumed)
        _append_formal_ledger(
            root,
            "phase_completed",
            phase="derived-contract",
            artifact_directory=str(derived_canonical),
            phase_wall_seconds=elapsed,
            consumed_wall_seconds=consumed,
        )
        events = _read_formal_ledger(root)
        if artifact_tree_sha256(source_canonical) != source_digest_before:
            return stop(
                "contract-failed",
                "contract-failed",
                "source changed while building the derived contract",
                _formal_consumed_wall_seconds(events),
            )
    else:
        derived_canonical = _formal_phase_artifact(events, "derived-contract") or derived_canonical
        if not (derived_canonical / "confirmation-contract.json").is_file():
            return stop(
                "contract-failed",
                "contract-failed",
                "completed derived contract is missing",
                _formal_consumed_wall_seconds(events),
            )

    if _formal_consumed_wall_seconds(events) >= resolved.campaign_wall_seconds:
        return stop(
            "performance-blocked",
            "performance-blocked",
            "campaign wall budget exhausted before independent checker",
            _formal_consumed_wall_seconds(events),
        )
    checker_event = _formal_latest_phase_event(events, "independent-check")
    checker_canonical = root / "independent-check"
    if checker_event is None or checker_event.get("event") not in {
        "phase_completed",
        "phase_recovered",
    }:
        result, attempt, elapsed, phase_error, timed_out = phase_run(
            "independent-check",
            deadline=data_deadline,
            runner=checker_runner,
            kwargs={
                "source_directory": source_canonical,
                "destination": derived_canonical,
                "replay_workers": replay_workers,
            },
            output_argument="report_directory",
            output_subdirectory="output",
        )
        consumed = _formal_consumed_wall_seconds(events)
        if phase_error is not None or attempt is None:
            status = "performance-blocked" if timed_out else "contract-failed"
            return stop(
                status,
                status,
                phase_error or "independent checker returned no result",
                consumed,
            )
        checker_output = attempt / "output"
        report_path = checker_output / "verification-report.json"
        if result is None or not report_path.is_file() or result.get("valid") is not True:
            return stop(
                "contract-failed",
                "contract-failed",
                "independent checker did not produce a valid report",
                consumed,
            )
        try:
            _formal_promote_directory(checker_output, checker_canonical)
        except (ArtifactError, OSError) as error:
            return stop("contract-failed", "contract-failed", str(error), consumed)
        _append_formal_ledger(
            root,
            "phase_completed",
            phase="independent-check",
            artifact_directory=str(checker_canonical),
            phase_wall_seconds=elapsed,
            consumed_wall_seconds=consumed,
        )
        events = _read_formal_ledger(root)
        if artifact_tree_sha256(source_canonical) != source_digest_before:
            return stop(
                "contract-failed",
                "contract-failed",
                "source changed while running the independent checker",
                _formal_consumed_wall_seconds(events),
            )
    else:
        checker_canonical = _formal_phase_artifact(events, "independent-check") or checker_canonical
        report = _read_json(checker_canonical / "verification-report.json")
        if report.get("valid") is not True:
            return stop(
                "contract-failed",
                "contract-failed",
                "completed checker report is not valid",
                _formal_consumed_wall_seconds(events),
            )
    final_consumed = _formal_consumed_wall_seconds(events)
    if final_consumed > resolved.campaign_wall_seconds:
        return stop(
            "performance-blocked",
            "performance-blocked",
            "campaign wall budget exhausted during finalization",
            final_consumed,
        )
    return stop(
        "completed", str(source_summary.get("gate")), "formal_campaign_completed", final_consumed
    )


def _run_preflight_point_with_watchdog(
    fixture_runner: Any,
    *,
    worker_count: int,
    fixture_seeds: tuple[str, ...],
    timeout_seconds: float,
) -> tuple[Mapping[str, Any] | None, str | None]:
    result, error, _ = _run_mapping_with_watchdog(
        fixture_runner,
        kwargs={
            "worker_count": worker_count,
            "fixture_seeds": fixture_seeds,
        },
        timeout_seconds=timeout_seconds,
    )
    return result, error


def run_confirmation_scaling_preflight(
    config: ConfirmationConfig | Mapping[str, Any],
    *,
    fixture_runner: Any,
    fixture_seeds: Sequence[str] = CONFIRMATION_PREFLIGHT_FIXTURE_SEEDS,
    hard_cap_seconds: float = CONFIRMATION_PREFLIGHT_WALL_SECONDS,
    worker_points: Sequence[int] = (1, 2, 4),
    max_rss_bytes: int | None = None,
    max_cpu_efficiency: float | None = None,
    clock: Any = time.monotonic,
) -> dict[str, Any]:
    """Evaluate bounded worker points without touching formal campaign seeds.

    ``fixture_runner`` is intentionally injected: the preflight is a separately
    approved operation and callers must choose the approved CPU host before
    invoking it.  It returns a record for one worker point containing at least a
    scientific digest and throughput; the function owns ordering, equivalence,
    budget, and near-elbow selection.
    """

    resolved = (
        config if isinstance(config, ConfirmationConfig) else resolve_confirmation_config(config)
    )
    if hard_cap_seconds <= 0.0 or not math.isfinite(hard_cap_seconds):
        raise ArtifactError("scaling preflight hard cap must be positive and finite")
    if any(type(point) is not int for point in worker_points):
        raise ArtifactError("scaling preflight worker points must be integers")
    points = tuple(dict.fromkeys(worker_points))
    if points != tuple(sorted(points)) or any(point <= 0 for point in points):
        raise ArtifactError("scaling preflight worker points must be positive and sorted")
    if not {1, 2, 4}.issubset(points):
        raise ArtifactError("scaling preflight requires worker points 1, 2, and 4")
    if 8 in points and not {1, 2, 4}.issubset(points):
        raise ArtifactError("scaling preflight worker 8 requires declared points 1, 2, and 4")
    if (
        not fixture_seeds
        or len(set(fixture_seeds)) != len(fixture_seeds)
        or set(fixture_seeds)
        & (
            set(resolved.training_seeds)
            | set(resolved.legacy_seed_denylist)
            | {resolved.evaluation_root_seed}
        )
    ):
        raise ArtifactError("scaling preflight fixture seeds must be independent")
    started = clock()
    measurements: list[dict[str, Any]] = []
    stop_reason = "all_declared_points_complete"
    for point_index, worker_count in enumerate(points):
        if clock() - started >= hard_cap_seconds:
            stop_reason = "preflight_wall_cap_reached"
            break
        if worker_count == 8 and {1, 2, 4}.issubset(
            {item["worker_count"] for item in measurements}
        ):
            by_workers = {item["worker_count"]: item for item in measurements}
            gain_4_over_2 = (
                float(by_workers[4]["throughput"]) - float(by_workers[2]["throughput"])
            ) / max(float(by_workers[2]["throughput"]), 1e-12)
            if gain_4_over_2 < 0.15:
                stop_reason = "worker_8_point_not_justified_by_worker_4_gain"
                break
        remaining_seconds = max(0.0, hard_cap_seconds - (clock() - started))
        result, worker_error = _run_preflight_point_with_watchdog(
            fixture_runner,
            worker_count=worker_count,
            fixture_seeds=tuple(fixture_seeds),
            timeout_seconds=remaining_seconds,
        )
        if worker_error is not None or result is None:
            return {
                "schema_version": "oi-baseline-confirmation-scaling-preflight-v1",
                "valid": False,
                "selected_worker_count": None,
                "measurements": measurements,
                "stop_reason": (
                    "preflight_worker_point_timed_out"
                    if "timed out" in (worker_error or "")
                    else "preflight_worker_point_failed"
                ),
                "errors": [worker_error or "preflight worker returned no result"],
                "failed_worker_count": worker_count,
                "fixture_seeds": list(fixture_seeds),
                "thread_env": fixed_thread_environment(),
                "start_method": resolved.resources.start_method,
                "hard_cap_seconds": hard_cap_seconds,
            }
        digest = result.get("scientific_digest")
        throughput = _finite_number(result.get("throughput"), field="preflight.throughput")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ArtifactError("scaling preflight scientific digest is malformed")
        if throughput < 0.0:
            raise ArtifactError("scaling preflight throughput must be non-negative")
        telemetry = result.get("runtime_telemetry", {})
        if not isinstance(telemetry, Mapping):
            raise ArtifactError("scaling preflight runtime telemetry is malformed")
        normalized = validate_runtime_telemetry(
            telemetry,
            budget_seconds=hard_cap_seconds,
            max_rss_bytes=max_rss_bytes,
        )
        cpu_efficiency = result.get("cpu_efficiency")
        if cpu_efficiency is not None:
            cpu_efficiency_value = _finite_number(cpu_efficiency, field="preflight.cpu_efficiency")
            if cpu_efficiency_value < 0.0 or (
                max_cpu_efficiency is not None and cpu_efficiency_value > max_cpu_efficiency
            ):
                raise ArtifactError(
                    "scaling preflight CPU efficiency exceeds the resource contract"
                )
        measurements.append(
            {
                "worker_count": worker_count,
                "scientific_digest": digest,
                "throughput": throughput,
                "runtime_telemetry": normalized,
                "cpu_efficiency": cpu_efficiency,
            }
        )
        if clock() - started >= hard_cap_seconds and point_index < len(points) - 1:
            stop_reason = "preflight_wall_cap_reached"
            break
    if not measurements:
        return {
            "schema_version": "oi-baseline-confirmation-scaling-preflight-v1",
            "valid": False,
            "selected_worker_count": None,
            "measurements": [],
            "stop_reason": stop_reason,
            "fixture_seeds": list(fixture_seeds),
        }
    if stop_reason == "preflight_wall_cap_reached" and len(measurements) < len(points):
        return {
            "schema_version": "oi-baseline-confirmation-scaling-preflight-v1",
            "valid": False,
            "selected_worker_count": None,
            "measurements": measurements,
            "stop_reason": stop_reason,
            "errors": ["scaling preflight hard cap reached before all required points completed"],
            "fixture_seeds": list(fixture_seeds),
            "thread_env": fixed_thread_environment(),
            "start_method": resolved.resources.start_method,
            "hard_cap_seconds": hard_cap_seconds,
        }
    digests = {measurement["scientific_digest"] for measurement in measurements}
    if len(digests) != 1:
        raise ArtifactError("scaling preflight scientific projection drifted across workers")
    by_workers = {measurement["worker_count"]: measurement for measurement in measurements}
    selected = measurements[-1]["worker_count"]
    for lower, upper in zip(measurements, measurements[1:], strict=False):
        lower_throughput = float(lower["throughput"])
        if lower_throughput <= 0.0:
            selected = lower["worker_count"]
            break
        gain = (float(upper["throughput"]) - lower_throughput) / lower_throughput
        if gain < 0.15:
            selected = lower["worker_count"]
            break
    if 4 in by_workers and 2 in by_workers:
        gain_4_over_2 = (
            float(by_workers[4]["throughput"]) - float(by_workers[2]["throughput"])
        ) / max(float(by_workers[2]["throughput"]), 1e-12)
        if gain_4_over_2 >= 0.15 and 8 not in by_workers and clock() - started < hard_cap_seconds:
            # A caller that declared only 1/2/4 is not allowed to silently
            # invent an 8-worker point; record the gate for the next approved
            # invocation instead.
            stop_reason = "worker_8_point_requires_explicit_declaration"
    return {
        "schema_version": "oi-baseline-confirmation-scaling-preflight-v1",
        "valid": True,
        "selected_worker_count": selected,
        "measurements": measurements,
        "fixture_seeds": list(fixture_seeds),
        "thread_env": fixed_thread_environment(),
        "start_method": resolved.resources.start_method,
        "stop_reason": stop_reason,
        "hard_cap_seconds": hard_cap_seconds,
    }


run_scaling_preflight = run_confirmation_scaling_preflight


__all__ = [
    "CONFIRMATION_ARM_IDS",
    "CONFIRMATION_CANDIDATES",
    "CONFIRMATION_GATES",
    "CONFIRMATION_LEGACY_SEEDS",
    "CONFIRMATION_SCHEMA_PATH",
    "CONFIRMATION_SCHEMA_VERSION",
    "ConfirmationCandidateConfig",
    "ConfirmationConfig",
    "ConfirmationConfigError",
    "ConfirmationGate",
    "ConfirmationResourceConfig",
    "ConfirmationShardRequest",
    "apply_thread_environment",
    "compute_confirmation_gate",
    "confirmation_config_hash",
    "fixed_thread_environment",
    "load_confirmation_config",
    "reduce_confirmation_gate",
    "resolve_confirmation_config",
    "run_confirmation_campaign",
    "run_confirmation_formal_campaign",
    "run_confirmation_scaling_preflight",
    "run_confirmation_shard",
    "run_scaling_preflight",
    "scientific_digest",
    "scientific_projection",
    "validate_runtime_telemetry",
    "verify_confirmation_shard",
]
