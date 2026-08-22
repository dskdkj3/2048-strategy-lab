"""Derived, read-only contract projection for calibration artifacts.

The v1 calibration directory remains the immutable scientific source.  This
module validates that source, derives a complete machine-readable projection,
and deterministically replays confirmation lineages into a separate bundle.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]

from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.artifacts import ArtifactError, canonical_json, config_hash
from strategy2048.experiments.calibration import (
    CALIBRATION_CONFIRM_EPISODE,
    CALIBRATION_SCREEN_EPISODE,
    CalibrationConfig,
    _candidate,
    _coverage_summary,
    _evaluation_records,
    _execution_config,
    _latest_stop,
    _milestone_records,
    _stage_decisions,
    _training_records,
    recompute_calibration_summary,
    resolve_calibration_config,
    verify_calibration_artifact,
)
from strategy2048.experiments.discovery import (
    REPOSITORY_ROOT,
    DiscoveryArmConfig,
    _build_agent,
    _counter_delta,
    _read_json,
)
from strategy2048.experiments.evaluation import _percentile
from strategy2048.rules.core import max_tile_value

CONTRACT_SCHEMA_VERSION = "algorithm-calibration-contract-v2"
PROJECTION_SCHEMA_VERSION = "algorithm-calibration-projection-v2"
TREE_HASH_SCHEMA_VERSION = "artifact-tree-sha256-v1"
CONTRACT_SCHEMA_PATH = REPOSITORY_ROOT / "schemas/algorithm-calibration-contract.v2.schema.json"
TILE_THRESHOLDS = (128, 256, 512, 1024, 2048, 4096)
SCORE_MILESTONE = 5000
TILE_MILESTONE = 256


def _load_schema() -> dict[str, Any]:
    value = json.loads(CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ArtifactError("calibration contract schema must be an object")
    return value


def _validate_schema(value: Mapping[str, Any]) -> None:
    try:
        jsonschema.Draft202012Validator(_load_schema()).validate(dict(value))
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f" at {location}" if location else ""
        raise ArtifactError(
            f"calibration contract schema validation failed{prefix}: {error.message}"
        ) from error


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir():
        raise ArtifactError(f"artifact directory does not exist: {root}")
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ArtifactError(f"artifact tree contains a symlink: {path}")
        if path.is_file():
            files.append(path)
        elif not path.is_dir():
            raise ArtifactError(f"artifact tree contains a non-regular entry: {path}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix().encode("utf-8"))


def artifact_tree_sha256(artifact_directory: str | Path) -> str:
    """Hash relative paths and raw bytes without absolute-path or metadata drift."""

    root = Path(artifact_directory).resolve(strict=True)
    digest = hashlib.sha256()
    digest.update(TREE_HASH_SCHEMA_VERSION.encode("ascii") + b"\0")
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def _nonnegative_int(value: object, *, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ArtifactError(f"{field} must be a non-negative integer")
    return value


def _finite_nonnegative(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ArtifactError(f"{field} must be numeric")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ArtifactError(f"{field} must be finite and non-negative")
    return numeric


def _required_mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ArtifactError(f"{field} must be an object")
    return cast(Mapping[str, Any], value)


def _counter_values(value: object, *, field: str) -> dict[str, int]:
    mapping = _required_mapping(value, field=field)
    return {
        name: _nonnegative_int(mapping.get(name), field=f"{field}.{name}")
        for name in ("action_value_calls", "tuple_lookups", "updates", "tuple_updates")
    }


def _group_records(
    records: Sequence[Mapping[str, Any]],
    *,
    candidate_field: str,
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        candidate_id = record.get(candidate_field)
        training_seed = record.get("training_seed")
        if not isinstance(candidate_id, str) or not isinstance(training_seed, str):
            raise ArtifactError("record candidate/training-seed identity is malformed")
        grouped.setdefault((candidate_id, training_seed), []).append(record)
    return grouped


def validate_training_structure(artifact_directory: str | Path) -> dict[str, Any]:
    """Validate episode/counter/checkpoint continuity without replaying training."""

    root = Path(artifact_directory)
    config = resolve_calibration_config(_read_json(root / "resolved-config.json"))
    candidate_ids = {candidate.id for candidate in config.candidates}
    seed_ids = set(config.training_seeds)
    grouped = _group_records(_training_records(root), candidate_field="arm_id")
    milestones = _milestone_records(root)
    latest_stop = _latest_stop(root) or {}
    completed = latest_stop.get("stop_reason") == "completed"
    survivor: str | None = None
    if completed:
        decisions = _stage_decisions(root)
        if decisions:
            survivor_value = decisions[0].get("survivor_candidate_id")
            survivor = survivor_value if isinstance(survivor_value, str) else None
    result: dict[str, Any] = {}

    for candidate_id, training_seed in sorted(
        (candidate.id, seed) for candidate in config.candidates for seed in config.training_seeds
    ):
        records = grouped.get((candidate_id, training_seed), [])
        if len(records) > config.confirm_target_episode:
            raise ArtifactError(
                f"training record count exceeds confirmation target: {candidate_id}/{training_seed}"
            )
        if completed:
            expected_count = (
                config.confirm_target_episode
                if survivor is not None and candidate_id in {"td0_zero", survivor}
                else config.screen_target_episode
            )
            if len(records) != expected_count:
                raise ArtifactError(
                    f"completed training record count mismatch: "
                    f"{candidate_id}/{training_seed}; expected {expected_count}, "
                    f"got {len(records)}"
                )
        previous_global_step = 0
        previous_counters = {
            "action_value_calls": 0,
            "tuple_lookups": 0,
            "updates": 0,
            "tuple_updates": 0,
        }
        for expected_episode, record in enumerate(records):
            if (
                record.get("arm_id") not in candidate_ids
                or record.get("training_seed") not in seed_ids
            ):
                raise ArtifactError("training record identity is outside the resolved matrix")
            episode_id = _nonnegative_int(record.get("episode_id"), field="episode_id")
            if episode_id != expected_episode:
                raise ArtifactError(
                    f"training episode sequence mismatch: {candidate_id}/{training_seed}; "
                    f"expected {expected_episode}, got {episode_id}"
                )
            steps = _nonnegative_int(record.get("steps"), field="steps")
            global_step = _nonnegative_int(record.get("global_env_step"), field="global_env_step")
            if global_step != previous_global_step + steps:
                raise ArtifactError(
                    f"training global env step is discontinuous: "
                    f"{candidate_id}/{training_seed}/{episode_id}"
                )
            _finite_nonnegative(record.get("wall_seconds"), field="wall_seconds")
            _finite_nonnegative(record.get("process_cpu_seconds", 0.0), field="process_cpu_seconds")
            _nonnegative_int(record.get("official_score"), field="official_score")
            _nonnegative_int(record.get("max_tile"), field="max_tile")
            delta = _counter_values(record.get("counter_delta"), field="counter_delta")
            counters = _counter_values(record.get("counters"), field="counters")
            expected_counters = {
                name: previous_counters[name] + delta[name] for name in previous_counters
            }
            if counters != expected_counters:
                raise ArtifactError(
                    f"training counters are discontinuous: "
                    f"{candidate_id}/{training_seed}/{episode_id}"
                )
            lineage = _required_mapping(
                record.get("environment_rng_lineage"), field="environment_rng_lineage"
            )
            if (
                lineage.get("root_seed") != training_seed
                or lineage.get("purpose") != "train-env"
                or lineage.get("environment_id") != f"{config.experiment_id}-training"
                or lineage.get("episode_id") != episode_id
            ):
                raise ArtifactError(
                    f"training RNG lineage mismatch: {candidate_id}/{training_seed}/{episode_id}"
                )
            state_hash = record.get("learner_state_hash")
            if not isinstance(state_hash, str) or len(state_hash) != 64:
                raise ArtifactError("training learner state hash is malformed")
            previous_global_step = global_step
            previous_counters = counters

        for checkpoint_episode in (
            0,
            config.screen_target_episode,
            config.confirm_target_episode,
        ):
            checkpoint = milestones.get((candidate_id, training_seed, checkpoint_episode))
            if checkpoint is None:
                continue
            if checkpoint.get("completed_training_episodes") != checkpoint_episode:
                raise ArtifactError(
                    f"milestone completed episode mismatch: "
                    f"{candidate_id}/{training_seed}/{checkpoint_episode}"
                )
            if checkpoint_episode == 0:
                if checkpoint.get("global_env_step") != 0:
                    raise ArtifactError("episode-0 milestone has a non-zero global env step")
                continue
            if len(records) < checkpoint_episode:
                raise ArtifactError(
                    f"milestone has no corresponding training record: "
                    f"{candidate_id}/{training_seed}/{checkpoint_episode}"
                )
            endpoint = records[checkpoint_episode - 1]
            if (
                checkpoint.get("global_env_step") != endpoint.get("global_env_step")
                or checkpoint.get("learner_state_hash") != endpoint.get("learner_state_hash")
                or canonical_json(checkpoint.get("counters"))
                != canonical_json(endpoint.get("counters"))
            ):
                raise ArtifactError(
                    f"milestone does not match its training endpoint: "
                    f"{candidate_id}/{training_seed}/{checkpoint_episode}"
                )
            environment = _required_mapping(checkpoint.get("environment"), field="environment")
            rng = _required_mapping(environment.get("rng"), field="environment.rng")
            lineage = _required_mapping(rng.get("lineage"), field="environment.rng.lineage")
            if lineage.get("episode_id") != checkpoint_episode - 1:
                raise ArtifactError(
                    f"milestone environment episode mismatch: "
                    f"{candidate_id}/{training_seed}/{checkpoint_episode}"
                )

        result[f"{candidate_id}/{training_seed}"] = {
            "training_episode_count": len(records),
            "last_global_env_step": previous_global_step,
            "last_learner_state_hash": (records[-1].get("learner_state_hash") if records else None),
        }
    return result


def _aggregate_evaluations(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ArtifactError("evaluation aggregate cannot be empty")
    scores = sorted(
        _finite_nonnegative(record.get("official_score"), field="official_score")
        for record in records
    )
    tiles = [_nonnegative_int(record.get("max_tile"), field="max_tile") for record in records]
    return {
        "count": len(records),
        "official_score": {
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "p25": _percentile(scores, 0.25),
            "p75": _percentile(scores, 0.75),
            "p95": _percentile(scores, 0.95),
            "range": [min(scores), max(scores)],
        },
        "max_tile": max(tiles),
        "max_tile_mean": statistics.fmean(tiles),
        "tile_reach_rate": {
            str(tile): sum(value >= tile for value in tiles) / len(tiles)
            for tile in TILE_THRESHOLDS
        },
        "env_steps": sum(
            _nonnegative_int(record.get("steps"), field="steps") for record in records
        ),
        "wall_seconds": sum(
            _finite_nonnegative(record.get("wall_seconds"), field="wall_seconds")
            for record in records
        ),
        "process_cpu_seconds": sum(
            _finite_nonnegative(record.get("process_cpu_seconds", 0.0), field="process_cpu_seconds")
            for record in records
        ),
    }


def _training_cost(records: Sequence[Mapping[str, Any]], checkpoint_episode: int) -> dict[str, Any]:
    prefix = [
        record
        for record in records
        if _nonnegative_int(record.get("episode_id"), field="episode_id") < checkpoint_episode
    ]
    return {
        "episodes": len(prefix),
        "env_steps": sum(_nonnegative_int(record.get("steps"), field="steps") for record in prefix),
        "updates": sum(
            _counter_values(record.get("counter_delta"), field="counter_delta")["updates"]
            for record in prefix
        ),
        "wall_seconds": sum(
            _finite_nonnegative(record.get("wall_seconds"), field="wall_seconds")
            for record in prefix
        ),
        "process_cpu_seconds": sum(
            _finite_nonnegative(record.get("process_cpu_seconds", 0.0), field="process_cpu_seconds")
            for record in prefix
        ),
    }


def _positive_rate(numerator: float, denominator: float, *, field: str) -> float:
    if not math.isfinite(numerator):
        raise ArtifactError(f"{field} numerator is not finite")
    if not math.isfinite(denominator) or denominator <= 0.0:
        raise ArtifactError(f"{field} denominator must be finite and positive")
    return numerator / denominator


def _evaluation_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, str, int], list[Mapping[str, Any]]]:
    index: dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = {}
    seen: set[tuple[str, str, str, int, int]] = set()
    for record in records:
        candidate_id = record.get("candidate_id")
        training_seed = record.get("training_seed")
        suite = record.get("suite")
        checkpoint = record.get("checkpoint_episode")
        episode_id = record.get("evaluation_episode_id")
        if (
            not isinstance(candidate_id, str)
            or not isinstance(training_seed, str)
            or not isinstance(suite, str)
            or type(checkpoint) is not int
            or type(episode_id) is not int
        ):
            raise ArtifactError("evaluation identity is malformed")
        identity = (candidate_id, training_seed, suite, checkpoint, episode_id)
        if identity in seen:
            raise ArtifactError(f"duplicate evaluation record: {identity}")
        seen.add(identity)
        index.setdefault((candidate_id, training_seed, suite, checkpoint), []).append(record)
    for values in index.values():
        values.sort(key=lambda record: cast(int, record["evaluation_episode_id"]))
    return index


def _milestone_attainment(
    evaluations: Mapping[tuple[str, str, str, int], Sequence[Mapping[str, Any]]],
    training: Sequence[Mapping[str, Any]],
    *,
    candidate_id: str,
    training_seed: str,
) -> dict[str, Any]:
    observed: list[tuple[int, Sequence[Mapping[str, Any]]]] = []
    for checkpoint in (0, CALIBRATION_SCREEN_EPISODE, CALIBRATION_CONFIRM_EPISODE):
        suites = ("selection",) if checkpoint in {0, CALIBRATION_SCREEN_EPISODE} else ("audit",)
        for suite in suites:
            records = evaluations.get((candidate_id, training_seed, suite, checkpoint))
            if records:
                observed.append((checkpoint, records))

    def attainment(kind: str, target: int) -> dict[str, Any]:
        for checkpoint, records in observed:
            aggregate = _aggregate_evaluations(records)
            value = (
                cast(float, cast(Mapping[str, Any], aggregate["official_score"])["mean"])
                if kind == "score"
                else cast(int, aggregate["max_tile"])
            )
            if value >= target:
                return {
                    "target": target,
                    "status": "attained",
                    "checkpoint_episode": checkpoint,
                    "observed_value": value,
                    "training_cost": _training_cost(training, checkpoint),
                }
        return {
            "target": target,
            "status": "not-attained",
            "checkpoint_episode": None,
            "observed_value": None,
            "training_cost": None,
        }

    return {
        "score": attainment("score", SCORE_MILESTONE),
        "tile": attainment("tile", TILE_MILESTONE),
    }


def _metric_projection(root: Path, config: CalibrationConfig) -> dict[str, Any]:
    training_records = _training_records(root)
    evaluation_records = _evaluation_records(root)
    training_by_run = _group_records(training_records, candidate_field="arm_id")
    evaluation = _evaluation_index(evaluation_records)
    coverage = _coverage_summary(root, config)
    coverage_runs = cast(Mapping[str, Any], coverage.get("runs", {}))
    runs: dict[str, Any] = {}

    for candidate in config.candidates:
        for training_seed in config.training_seeds:
            key = (candidate.id, training_seed)
            records = training_by_run.get(key, [])
            evaluations: dict[str, Any] = {}
            for (candidate_id, seed, suite, checkpoint), values in sorted(evaluation.items()):
                if (candidate_id, seed) != key:
                    continue
                evaluations.setdefault(suite, {})[str(checkpoint)] = _aggregate_evaluations(values)
            runs[f"{candidate.id}/{training_seed}"] = {
                "candidate_id": candidate.id,
                "training_seed": training_seed,
                "training": _training_cost(records, len(records)),
                "training_at_checkpoint": {
                    str(checkpoint): _training_cost(records, checkpoint)
                    for checkpoint in (
                        0,
                        config.screen_target_episode,
                        config.confirm_target_episode,
                    )
                    if len(records) >= checkpoint
                },
                "evaluation": evaluations,
                "milestone_efficiency": _milestone_attainment(
                    evaluation,
                    records,
                    candidate_id=candidate.id,
                    training_seed=training_seed,
                ),
                "exploration": coverage_runs.get(f"{candidate.id}/{training_seed}"),
            }

    paired: list[dict[str, Any]] = []
    for candidate in config.candidates:
        if candidate.id == "td0_zero":
            continue
        for training_seed in config.training_seeds:
            for suite, checkpoint in (
                ("selection", 0),
                ("selection", config.screen_target_episode),
                ("audit", config.confirm_target_episode),
            ):
                zero_records = evaluation.get(("td0_zero", training_seed, suite, checkpoint))
                candidate_records = evaluation.get((candidate.id, training_seed, suite, checkpoint))
                if not zero_records and not candidate_records:
                    continue
                if suite == "audit" and not candidate_records:
                    # Only zero and the selected survivor enter confirmation.
                    continue
                if not zero_records or not candidate_records:
                    raise ArtifactError(
                        f"paired evaluation is incomplete: {candidate.id}/{training_seed}/"
                        f"{suite}/{checkpoint}"
                    )
                zero_ids = [cast(int, record["evaluation_episode_id"]) for record in zero_records]
                candidate_ids = [
                    cast(int, record["evaluation_episode_id"]) for record in candidate_records
                ]
                if zero_ids != candidate_ids:
                    raise ArtifactError("paired evaluation episode IDs do not match")
                zero_mean = cast(
                    float,
                    cast(Mapping[str, Any], _aggregate_evaluations(zero_records)["official_score"])[
                        "mean"
                    ],
                )
                candidate_mean = cast(
                    float,
                    cast(
                        Mapping[str, Any],
                        _aggregate_evaluations(candidate_records)["official_score"],
                    )["mean"],
                )
                if zero_mean <= 0.0:
                    raise ArtifactError("paired relative score denominator must be positive")
                paired.append(
                    {
                        "candidate_id": candidate.id,
                        "training_seed": training_seed,
                        "suite": suite,
                        "checkpoint_episode": checkpoint,
                        "evaluation_episode_ids": zero_ids,
                        "zero_official_score_mean": zero_mean,
                        "candidate_official_score_mean": candidate_mean,
                        "absolute_difference": candidate_mean - zero_mean,
                        "relative_difference": (candidate_mean - zero_mean) / zero_mean,
                    }
                )

    screen_efficiency: list[dict[str, Any]] = []
    for candidate in config.candidates:
        for training_seed in config.training_seeds:
            initial = evaluation.get((candidate.id, training_seed, "selection", 0))
            screened = evaluation.get(
                (candidate.id, training_seed, "selection", config.screen_target_episode)
            )
            if not initial or not screened:
                continue
            initial_mean = cast(
                float,
                cast(Mapping[str, Any], _aggregate_evaluations(initial)["official_score"])["mean"],
            )
            screened_mean = cast(
                float,
                cast(Mapping[str, Any], _aggregate_evaluations(screened)["official_score"])["mean"],
            )
            gain = screened_mean - initial_mean
            cost = _training_cost(
                training_by_run.get((candidate.id, training_seed), []),
                config.screen_target_episode,
            )
            screen_efficiency.append(
                {
                    "candidate_id": candidate.id,
                    "training_seed": training_seed,
                    "score_gain": gain,
                    "training_cost": cost,
                    "score_gain_per_episode": _positive_rate(
                        gain, float(cost["episodes"]), field="score_gain_per_episode"
                    ),
                    "score_gain_per_env_step": _positive_rate(
                        gain, float(cost["env_steps"]), field="score_gain_per_env_step"
                    ),
                    "score_gain_per_wall_second": _positive_rate(
                        gain, float(cost["wall_seconds"]), field="score_gain_per_wall_second"
                    ),
                    "score_gain_per_process_cpu_second": _positive_rate(
                        gain,
                        float(cost["process_cpu_seconds"]),
                        field="score_gain_per_process_cpu_second",
                    ),
                }
            )

    summary = recompute_calibration_summary(root)
    survivor = summary.get("survivor_candidate_id")
    confirm_efficiency: list[dict[str, Any]] = []
    if isinstance(survivor, str):
        for training_seed in config.training_seeds:
            zero_audit = evaluation.get(
                ("td0_zero", training_seed, "audit", config.confirm_target_episode)
            )
            survivor_audit = evaluation.get(
                (survivor, training_seed, "audit", config.confirm_target_episode)
            )
            if not zero_audit or not survivor_audit:
                continue
            zero_mean = cast(
                float,
                cast(Mapping[str, Any], _aggregate_evaluations(zero_audit)["official_score"])[
                    "mean"
                ],
            )
            survivor_mean = cast(
                float,
                cast(
                    Mapping[str, Any],
                    _aggregate_evaluations(survivor_audit)["official_score"],
                )["mean"],
            )
            advantage = survivor_mean - zero_mean
            records = training_by_run.get((survivor, training_seed), [])
            cost40 = _training_cost(records, config.screen_target_episode)
            cost200 = _training_cost(records, config.confirm_target_episode)
            delta = {
                field: cast(float | int, cost200[field]) - cast(float | int, cost40[field])
                for field in (
                    "episodes",
                    "env_steps",
                    "updates",
                    "wall_seconds",
                    "process_cpu_seconds",
                )
            }
            confirm_efficiency.append(
                {
                    "candidate_id": survivor,
                    "training_seed": training_seed,
                    "paired_audit_advantage": advantage,
                    "training_cost_delta_40_to_200": delta,
                    "advantage_per_episode": _positive_rate(
                        advantage, float(delta["episodes"]), field="advantage_per_episode"
                    ),
                    "advantage_per_env_step": _positive_rate(
                        advantage, float(delta["env_steps"]), field="advantage_per_env_step"
                    ),
                    "advantage_per_wall_second": _positive_rate(
                        advantage,
                        float(delta["wall_seconds"]),
                        field="advantage_per_wall_second",
                    ),
                    "advantage_per_process_cpu_second": _positive_rate(
                        advantage,
                        float(delta["process_cpu_seconds"]),
                        field="advantage_per_process_cpu_second",
                    ),
                }
            )

    return {
        "schema_version": PROJECTION_SCHEMA_VERSION,
        "runs": runs,
        "paired_comparisons": paired,
        "learning_efficiency": {
            "screen_own_gain": screen_efficiency,
            "confirm_paired_advantage_efficiency": confirm_efficiency,
            "cross_suite_own_gain_computed": False,
        },
        "exploration": coverage,
        "milestone_thresholds": {
            "official_score": SCORE_MILESTONE,
            "tile": TILE_MILESTONE,
        },
    }


def replay_confirmation_lineages(artifact_directory: str | Path) -> dict[str, Any]:
    """Restore episode 40 and reproduce every persisted episode through 200."""

    root = Path(artifact_directory)
    config = resolve_calibration_config(_read_json(root / "resolved-config.json"))
    structure = validate_training_structure(root)
    milestones = _milestone_records(root)
    training_by_run = _group_records(_training_records(root), candidate_field="arm_id")
    execution = _execution_config(
        config,
        evaluation_root_seed=config.selection_evaluation_root_seed,
        evaluation_episodes=config.screen_evaluation_episodes,
    )
    target_keys = sorted(
        (candidate_id, training_seed)
        for candidate_id, training_seed, checkpoint_episode in milestones
        if checkpoint_episode == config.confirm_target_episode
    )
    proofs: list[dict[str, Any]] = []
    for candidate_id, training_seed in target_keys:
        checkpoint40 = milestones.get((candidate_id, training_seed, config.screen_target_episode))
        checkpoint200 = milestones.get((candidate_id, training_seed, config.confirm_target_episode))
        if checkpoint40 is None or checkpoint200 is None:
            raise ArtifactError("confirmation lineage is missing an endpoint checkpoint")
        records = training_by_run.get((candidate_id, training_seed), [])
        if len(records) != config.confirm_target_episode:
            raise ArtifactError(
                f"confirmation lineage has {len(records)} records instead of "
                f"{config.confirm_target_episode}: {candidate_id}/{training_seed}"
            )
        candidate = cast(DiscoveryArmConfig, _candidate(config, candidate_id))
        agent = _build_agent(execution, candidate)
        step40 = _nonnegative_int(
            checkpoint40.get("global_env_step"), field="checkpoint40.global_env_step"
        )
        agent.restore_checkpoint(
            root / cast(str, checkpoint40["checkpoint_directory"]),
            step40,
            config_hash=config_hash(config.to_json()),
        )
        global_step = step40
        final_environment: dict[str, Any] | None = None
        for episode_id in range(config.screen_target_episode, config.confirm_target_episode):
            expected = records[episode_id]
            counters_before = agent.counters.to_json()
            env = OracleEnv(
                root_seed=training_seed,
                environment_id=f"{config.experiment_id}-training",
                max_steps=config.max_steps_per_episode,
            )
            observation = env.reset(episode_id=episode_id, purpose="train-env")
            steps = 0
            while not observation.terminated and not observation.truncated:
                action = agent.learner.choose_action(observation)
                transition = env.step(action)
                agent.learner.observe(transition, transition.observation)
                observation = transition.observation
                steps += 1
            global_step += steps
            snapshot = env.snapshot()
            actual = {
                "official_score": observation.score,
                "max_tile": max_tile_value(observation.board),
                "steps": steps,
                "terminated": observation.terminated,
                "truncated": observation.truncated,
                "global_env_step": global_step,
                "counter_delta": _counter_delta(counters_before, agent.counters.to_json()),
                "counters": agent.counters.to_json(),
                "learner_state_hash": agent.learner.state_hash(),
                "environment_rng_lineage": dict(snapshot.rng.lineage),
            }
            for field, value in actual.items():
                if canonical_json(expected.get(field)) != canonical_json(value):
                    raise ArtifactError(
                        f"confirmation replay mismatch: {candidate_id}/{training_seed}/"
                        f"{episode_id}/{field}"
                    )
            final_environment = snapshot.to_json()
        if final_environment is None:
            raise ArtifactError("confirmation replay produced no final environment")
        final_checks = {
            "completed_training_episodes": checkpoint200.get("completed_training_episodes")
            == config.confirm_target_episode,
            "global_env_step": checkpoint200.get("global_env_step") == global_step,
            "learner_state_hash": checkpoint200.get("learner_state_hash")
            == agent.learner.state_hash(),
            "table_hash": checkpoint200.get("table_hash") == agent.learner.table_hash(),
            "counters": canonical_json(checkpoint200.get("counters"))
            == canonical_json(agent.counters.to_json()),
            "environment": canonical_json(checkpoint200.get("environment"))
            == canonical_json(final_environment),
        }
        if not all(final_checks.values()):
            failed = sorted(name for name, valid in final_checks.items() if not valid)
            raise ArtifactError(
                f"confirmation final checkpoint mismatch: {candidate_id}/{training_seed}: "
                + ", ".join(failed)
            )
        proofs.append(
            {
                "candidate_id": candidate_id,
                "training_seed": training_seed,
                "start_checkpoint_episode": config.screen_target_episode,
                "end_checkpoint_episode": config.confirm_target_episode,
                "replayed_episode_start": config.screen_target_episode,
                "replayed_episode_end_inclusive": config.confirm_target_episode - 1,
                "replayed_episode_count": (
                    config.confirm_target_episode - config.screen_target_episode
                ),
                "start_learner_state_hash": checkpoint40.get("learner_state_hash"),
                "end_learner_state_hash": checkpoint200.get("learner_state_hash"),
                "end_table_hash": checkpoint200.get("table_hash"),
                "verified": True,
            }
        )
    return {
        "schema_version": "algorithm-calibration-lineage-proof-v1",
        "structural_runs": structure,
        "confirmation_lineages": proofs,
        "all_confirm_lineages_verified": all(proof["verified"] is True for proof in proofs),
    }


def _source_run_commit(root: Path) -> str:
    manifest = _read_json(root / "run-manifest.json")
    source = _required_mapping(manifest.get("source"), field="run-manifest.source")
    commit = source.get("commit")
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{7,64}", commit) is None:
        raise ArtifactError("source run commit is missing")
    if source.get("dirty") is not False:
        raise ArtifactError("formal calibration contract requires a clean source run commit")
    return commit


def _reducer_metadata(repo_root: Path) -> tuple[str, bool]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ArtifactError("cannot resolve reducer Git provenance") from error
    return commit, dirty


def recompute_calibration_contract(
    artifact_directory: str | Path,
    *,
    reducer_commit: str,
    reducer_dirty: bool = False,
) -> dict[str, Any]:
    root = Path(artifact_directory)
    if reducer_dirty:
        raise ArtifactError("formal calibration contract requires a clean reducer commit")
    source_verification = verify_calibration_artifact(root)
    if source_verification.get("valid") is not True:
        raise ArtifactError(
            "source calibration artifact is invalid: "
            + "; ".join(str(item) for item in source_verification.get("errors", []))
        )
    config = resolve_calibration_config(_read_json(root / "resolved-config.json"))
    source_result = recompute_calibration_summary(root)
    if source_result.get("stop_reason") != "completed":
        raise ArtifactError("formal calibration contract requires a completed source artifact")
    value = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "provenance": {
            "source_artifact_tree_sha256": artifact_tree_sha256(root),
            "source_artifact_tree_hash_schema": TREE_HASH_SCHEMA_VERSION,
            "source_run_commit": _source_run_commit(root),
            "source_config_hash": config_hash(config.to_json()),
            "reducer_commit": reducer_commit,
            "reducer_dirty": False,
        },
        "source_result": source_result,
        "projection": _metric_projection(root, config),
        "lineage_proof": replay_confirmation_lineages(root),
        "evidence_boundary": {
            "source_artifact_modified": False,
            "formal_training_run_started": False,
            "scientific_sample_count_expanded": False,
            "named_strategy_detector_run": False,
            "strong_replay_is_post_run_verification": True,
        },
    }
    _validate_schema(value)
    return value


def _assert_disjoint(source: Path, destination: Path) -> None:
    source = source.resolve(strict=True)
    destination = destination.resolve(strict=False)
    if destination == source or source in destination.parents or destination in source.parents:
        raise ArtifactError("source and derived contract destinations must be disjoint")


def build_calibration_contract(
    artifact_directory: str | Path,
    destination_directory: str | Path,
    *,
    reducer_commit: str | None = None,
    reducer_dirty: bool | None = None,
) -> dict[str, Any]:
    """Create a new sibling bundle without modifying or overwriting the source."""

    source = Path(artifact_directory).resolve(strict=True)
    destination = Path(destination_directory)
    _assert_disjoint(source, destination)
    if destination.exists():
        raise ArtifactError(f"derived contract destination already exists: {destination}")
    if reducer_commit is None or reducer_dirty is None:
        detected_commit, detected_dirty = _reducer_metadata(REPOSITORY_ROOT)
        reducer_commit = detected_commit if reducer_commit is None else reducer_commit
        reducer_dirty = detected_dirty if reducer_dirty is None else reducer_dirty
    if not isinstance(reducer_commit, str) or not reducer_commit:
        raise ArtifactError("reducer commit must be a non-empty string")
    if re.fullmatch(r"[0-9a-f]{7,64}", reducer_commit) is None:
        raise ArtifactError("reducer commit must be a hexadecimal Git revision")
    if reducer_dirty is not False:
        raise ArtifactError("formal calibration contract requires a clean reducer commit")

    before = artifact_tree_sha256(source)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ArtifactError(f"derived contract temporary destination exists: {temporary}")
    try:
        value = recompute_calibration_contract(
            source,
            reducer_commit=reducer_commit,
            reducer_dirty=False,
        )
        temporary.mkdir(parents=True)
        (temporary / "calibration-contract.json").write_text(
            canonical_json(value) + "\n", encoding="utf-8"
        )
        stored = _read_json(temporary / "calibration-contract.json")
        _validate_schema(stored)
        after = artifact_tree_sha256(source)
        if before != after:
            raise ArtifactError("source artifact changed while deriving the calibration contract")
        temporary.replace(destination)
        return value
    except Exception:
        if temporary.is_dir():
            shutil.rmtree(temporary)
        raise


def verify_calibration_contract(
    artifact_directory: str | Path,
    contract_directory: str | Path,
) -> dict[str, Any]:
    """Recompute projection and replay; never trust the stored attestation alone."""

    source = Path(artifact_directory)
    contract_root = Path(contract_directory)
    errors: list[str] = []
    try:
        _assert_disjoint(source, contract_root)
        contract_files = [
            path.relative_to(contract_root).as_posix() for path in _regular_files(contract_root)
        ]
        if contract_files != ["calibration-contract.json"]:
            raise ArtifactError("derived calibration contract contains unexpected files")
        path = contract_root / "calibration-contract.json"
        if not path.is_file() or path.is_symlink():
            raise ArtifactError("derived calibration contract is missing or symlinked")
        stored = _read_json(path)
        _validate_schema(stored)
        provenance = _required_mapping(stored.get("provenance"), field="provenance")
        reducer_commit = provenance.get("reducer_commit")
        if not isinstance(reducer_commit, str) or not reducer_commit:
            raise ArtifactError("derived contract reducer commit is missing")
        expected = recompute_calibration_contract(
            source,
            reducer_commit=reducer_commit,
            reducer_dirty=False,
        )
        if canonical_json(stored) != canonical_json(expected):
            raise ArtifactError("stored calibration contract does not match raw-derived replay")
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")
    return {
        "schema_version": "algorithm-calibration-contract-verification-v1",
        "valid": not errors,
        "gate": (
            cast(Mapping[str, Any], stored.get("source_result", {})).get("gate")
            if not errors
            else "contract-failed"
        )
        if "stored" in locals()
        else "contract-failed",
        "errors": errors,
        "artifact_directory": str(source),
        "contract_directory": str(contract_root),
    }
