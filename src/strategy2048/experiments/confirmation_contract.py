"""Independent source, bundle, and strong-replay contracts for confirmation."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import re
import tempfile
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

import jsonschema  # type: ignore[import-untyped]

from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.artifacts import ArtifactError, canonical_json
from strategy2048.experiments.confirmation import (
    CONFIRMATION_ARM_IDS,
    ConfirmationConfig,
    _aggregate_shard_audit,
    _path_in_shard,
    _read_json,
    _read_jsonl,
    _shard_scientific_digest,
    apply_thread_environment,
    confirmation_config_hash,
    reduce_confirmation_gate,
    resolve_confirmation_config,
    scientific_digest,
    validate_runtime_telemetry,
    verify_confirmation_shard,
)
from strategy2048.experiments.discovery import (
    DiscoveryArmConfig,
    _build_agent,
    _counter_delta,
)

CONTRACT_SCHEMA_VERSION = "oi-baseline-confirmation-contract-v1"
PROJECTION_SCHEMA_VERSION = "oi-baseline-confirmation-projection-v1"
CONTRACT_SCHEMA_PATH = (
    Path(__file__).parents[3] / "schemas/oi-baseline-confirmation-contract.v1.schema.json"
)
TREE_HASH_SCHEMA_VERSION = "sha256-relative-files-v1"


def _assert_no_symlink_components(path: Path) -> None:
    if any(item.is_symlink() for item in (path, *path.parents)):
        raise ArtifactError(f"confirmation contract path contains a symlink component: {path}")


def _validate_schema(value: Mapping[str, Any]) -> None:
    schema = json.loads(CONTRACT_SCHEMA_PATH.read_text(encoding="utf-8"))
    try:
        jsonschema.Draft202012Validator(schema).validate(dict(value))
    except jsonschema.ValidationError as error:
        location = ".".join(str(part) for part in error.absolute_path)
        prefix = f" at {location}" if location else ""
        raise ArtifactError(
            f"confirmation contract schema validation failed{prefix}: {error.message}"
        ) from error


def _regular_files(root: Path) -> list[Path]:
    if root.is_symlink() or not root.is_dir():
        raise ArtifactError(f"confirmation contract root is not a real directory: {root}")
    files: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ArtifactError(f"confirmation contract rejects symlink: {path}")
        if path.is_file():
            files.append(path)
    return files


def artifact_tree_sha256(artifact_directory: str | Path) -> str:
    """Hash regular artifact files by relative path and content, rejecting links."""

    root = Path(artifact_directory)
    digest = hashlib.sha256()
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_root(path: str | Path) -> Path:
    candidate = Path(path)
    _assert_no_symlink_components(candidate)
    nested = candidate / "source"
    if nested.is_symlink():
        raise ArtifactError("confirmation source rejects a symlinked source directory")
    if nested.is_dir():
        return nested
    return candidate


def _cohort_paths(source: Path) -> list[Path]:
    cohorts = source / "cohorts"
    if not cohorts.is_dir() or cohorts.is_symlink():
        raise ArtifactError("confirmation source is missing cohorts directory")
    entries = list(cohorts.iterdir())
    if any(path.is_symlink() or not path.is_file() or path.suffix != ".json" for path in entries):
        raise ArtifactError("confirmation cohorts contain an unexpected entry")
    paths = sorted(entries)
    expected_names = [f"{index:04d}.json" for index in range(1, len(paths) + 1)]
    if [path.name for path in paths] != expected_names:
        raise ArtifactError("confirmation cohorts are not contiguous or contain unexpected files")
    return paths


def _campaign_manifest(source: Path) -> dict[str, Any]:
    campaign = source.parent / "campaign-manifest.json"
    if campaign.is_file():
        return _read_json(campaign)
    return {}


def _paired_records_from_source(source: Path, config: ConfirmationConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        seed_root = source / "shards" / seed
        if not seed_root.is_dir():
            break
        aggregates: dict[str, Mapping[str, Any]] = {}
        for candidate_id in CONFIRMATION_ARM_IDS:
            shard = seed_root / candidate_id
            if not shard.is_dir():
                break
            report = verify_confirmation_shard(shard)
            if report["valid"] is not True:
                raise ArtifactError(
                    f"confirmation shard verification failed: {seed}/{candidate_id}: "
                    + "; ".join(cast(list[str], report["errors"]))
                )
            aggregates[candidate_id] = _aggregate_shard_audit(shard)
        if len(aggregates) != len(CONFIRMATION_ARM_IDS):
            break
        zero = aggregates["td0_zero"]
        oi = aggregates["td0_oi_1000"]
        if zero["episode_ids"] != oi["episode_ids"]:
            raise ArtifactError(f"paired audit episode identities differ for {seed}")
        records.append(
            {
                "training_seed": seed,
                "zero_mean_score": zero["mean_score"],
                "oi_mean_score": oi["mean_score"],
                "zero_256_reach_rate": zero["tile_reach_rate_256"],
                "oi_256_reach_rate": oi["tile_reach_rate_256"],
                "zero_max_tile_mean": zero["max_tile_mean"],
                "oi_max_tile_mean": oi["max_tile_mean"],
                "evaluation_episode_ids": zero["episode_ids"],
            }
        )
    return records


def recompute_confirmation_source_summary(source_directory: str | Path) -> dict[str, Any]:
    """Recompute cohort order, paired effects, and the gate from shard raw data."""

    source = _source_root(source_directory)
    config = resolve_confirmation_config(_read_json(source / "resolved-config.json"))
    paired = _paired_records_from_source(source, config)
    gates: list[dict[str, Any]] = []
    for count in range(config.cohort_size, len(paired) + 1, config.cohort_size):
        if count < config.minimum_fresh_seeds:
            gates.append(
                reduce_confirmation_gate(
                    paired[:count], minimum_fresh_seeds=config.minimum_fresh_seeds
                )
            )
            continue
        gates.append(
            reduce_confirmation_gate(
                paired[:count],
                minimum_fresh_seeds=config.minimum_fresh_seeds,
                maximum_fresh_seeds=config.maximum_fresh_seeds,
                minimum_median_score_gain=config.minimum_median_score_gain,
                minimum_positive_share=config.minimum_positive_share,
                severe_regression_threshold=config.severe_regression_threshold,
                minimum_median_tile_reach_delta=config.minimum_median_tile_reach_delta,
            )
        )
        if gates[-1]["decision"] in {
            "oi-baseline-confirmed",
            "oi-baseline-rejected",
            "inconclusive",
        }:
            break
    final_gate = (
        gates[-1]
        if gates
        else reduce_confirmation_gate([], minimum_fresh_seeds=config.minimum_fresh_seeds)
    )
    return {
        "schema_version": "oi-baseline-confirmation-source-summary-v1",
        "experiment_id": config.experiment_id,
        "config_hash": confirmation_config_hash(config),
        "gate": final_gate["decision"],
        "stop_reason": final_gate["stop_reason"],
        "fresh_seed_count": len(paired),
        "paired_records": paired,
        "cohort_gates": gates,
        "evidence_boundary": {
            "statistical_significance_claimed": False,
            "old_calibration_seeds_included": False,
            "search_or_curriculum_used": False,
        },
    }


def _summary_contract_projection(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only source-summary fields that are derived from scientific raw data."""

    return {
        "schema_version": value.get("schema_version"),
        "experiment_id": value.get("experiment_id"),
        "config_hash": value.get("config_hash"),
        "gate": value.get("gate"),
        "stop_reason": value.get("stop_reason"),
        "fresh_seed_count": value.get("fresh_seed_count"),
        "paired_records": value.get("paired_records"),
        "evidence_boundary": value.get("evidence_boundary"),
    }


def verify_confirmation_source(source_directory: str | Path) -> dict[str, Any]:
    """Fail closed on seed reuse, cohort drift, shard gaps, or summary drift."""

    try:
        source = _source_root(source_directory)
    except ArtifactError as error:
        return {
            "schema_version": "oi-baseline-confirmation-source-verification-v1",
            "valid": False,
            "gate": "contract-failed",
            "errors": [str(error)],
            "source_directory": str(source_directory),
        }
    errors: list[str] = []
    try:
        config = resolve_confirmation_config(_read_json(source / "resolved-config.json"))
        campaign = _campaign_manifest(source)
        if campaign:
            if campaign.get("schema_version") != "oi-baseline-confirmation-campaign-v1":
                raise ArtifactError("unsupported confirmation campaign schema")
            if campaign.get("config_hash") != confirmation_config_hash(config):
                raise ArtifactError("confirmation campaign config hash mismatch")
            if campaign.get("experiment_id") != config.experiment_id:
                raise ArtifactError("confirmation campaign experiment identity mismatch")
            if campaign.get("fresh_seed_registry") != list(config.training_seeds):
                raise ArtifactError("confirmation campaign fresh seed registry drifted")
            if campaign.get("legacy_seed_denylist") != list(config.legacy_seed_denylist):
                raise ArtifactError("confirmation campaign legacy seed denylist drifted")
            expected_cohort_order = [
                list(config.training_seeds[index : index + config.cohort_size])
                for index in range(0, len(config.training_seeds), config.cohort_size)
            ]
            if campaign.get("cohort_order") != expected_cohort_order:
                raise ArtifactError("confirmation campaign cohort order drifted")
            if campaign.get("resource_contract") != config.resources.to_json():
                raise ArtifactError("confirmation campaign resource contract drifted")
            if campaign.get("campaign_wall_seconds") != config.campaign_wall_seconds:
                raise ArtifactError("confirmation campaign wall budget drifted")
        pairs = _paired_records_from_source(source, config)
        if len(pairs) < config.minimum_fresh_seeds:
            raise ArtifactError("confirmation source has fewer than the minimum fresh seeds")
        if len(pairs) > config.maximum_fresh_seeds:
            raise ArtifactError("confirmation source exceeds the maximum fresh seeds")
        if len(pairs) % config.cohort_size != 0:
            raise ArtifactError("confirmation source contains an incomplete cohort")
        cohorts = _cohort_paths(source)
        expected_cohorts = len(pairs) // config.cohort_size
        if len(cohorts) != expected_cohorts:
            raise ArtifactError("confirmation cohort count does not match completed paired shards")
        for index, cohort_path in enumerate(cohorts, 1):
            cohort = _read_json(cohort_path)
            expected_seeds = list(
                config.training_seeds[(index - 1) * config.cohort_size : index * config.cohort_size]
            )
            if cohort.get("training_seeds") != expected_seeds:
                raise ArtifactError("confirmation cohort order or seed identity drifted")
            expected_pairs = [pair for pair in pairs if pair.get("training_seed") in expected_seeds]
            if canonical_json(cohort.get("paired_records")) != canonical_json(expected_pairs):
                raise ArtifactError("confirmation cohort paired records drifted from raw shards")
        shard_root = source / "shards"
        if not shard_root.is_dir() or shard_root.is_symlink():
            raise ArtifactError("confirmation source is missing shards directory")
        if any(path.is_symlink() or not path.is_dir() for path in shard_root.iterdir()):
            raise ArtifactError("confirmation source shards contain an unexpected entry")
        present_seed_dirs = {
            path.name
            for path in shard_root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
        expected_seed_dirs = set(config.training_seeds[: len(pairs)])
        if present_seed_dirs != expected_seed_dirs:
            raise ArtifactError("confirmation source has a seed gap or unpaired seed directory")
        unexpected_seed_dirs = sorted(
            path.name
            for path in shard_root.iterdir()
            if path.is_dir()
            and path.name not in config.training_seeds
            and not path.name.startswith(".")
        )
        if unexpected_seed_dirs:
            raise ArtifactError(
                "confirmation source contains unexpected or legacy seed directories: "
                + ", ".join(unexpected_seed_dirs)
            )
        for seed in config.training_seeds[: len(pairs)]:
            seed_root = shard_root / seed
            entries = list(seed_root.iterdir())
            if any(path.is_symlink() or not path.is_dir() for path in entries):
                raise ArtifactError(
                    f"confirmation seed shard tree contains an unexpected entry: {seed}"
                )
            if {path.name for path in entries} != set(CONFIRMATION_ARM_IDS):
                raise ArtifactError(f"confirmation seed shard candidates are incomplete: {seed}")
        recomputed = recompute_confirmation_source_summary(source)
        stored = _read_json(source / "source-summary.json")
        if canonical_json(_summary_contract_projection(stored)) != canonical_json(
            _summary_contract_projection(recomputed)
        ):
            raise ArtifactError("confirmation source summary does not match raw shard records")
        for index, cohort_path in enumerate(cohorts):
            stored_cohort = _read_json(cohort_path)
            expected_gate = recomputed["cohort_gates"][index]
            if canonical_json(stored_cohort.get("gate")) != canonical_json(expected_gate):
                raise ArtifactError("confirmation cohort gate does not match raw shard records")
        verification_path = source / "source-verification.json"
        # The campaign coordinator runs this verifier before it atomically
        # publishes the attestation file.  Treat the attestation as an output
        # that can be checked when present, not as an input required to verify
        # the raw source itself; otherwise source verification is circular.
        if verification_path.is_file():
            source_attestation = _read_json(verification_path)
            if source_attestation.get("valid") is not True:
                raise ArtifactError("stored confirmation source verification is not valid")
        for shard in sorted((source / "shards").glob("*/*")):
            if shard.is_symlink():
                raise ArtifactError("confirmation source contains a shard symlink")
            if shard.is_dir() and shard.name.startswith("."):
                raise ArtifactError("confirmation source contains an unpublished shard attempt")
            if shard.is_dir() and shard.name not in CONFIRMATION_ARM_IDS:
                raise ArtifactError(f"confirmation source contains unexpected shard: {shard}")
    except (ArtifactError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    reported_gate = "contract-failed"
    summary_path = source / "source-summary.json"
    if summary_path.is_file() and not summary_path.is_symlink():
        try:
            reported_gate = str(_read_json(summary_path).get("gate", "contract-failed"))
        except (ArtifactError, json.JSONDecodeError, ValueError):
            reported_gate = "contract-failed"
    return {
        "schema_version": "oi-baseline-confirmation-source-verification-v1",
        "valid": not errors,
        "gate": reported_gate,
        "errors": errors,
        "source_directory": str(source),
    }


def _lineage_actual_fields(
    agent: Any,
    observation: Any,
    global_step: int,
    counters_before: Mapping[str, Any],
    snapshot: Any,
) -> dict[str, Any]:
    counters_after = agent.counters.to_json()
    return {
        "official_score": observation.score,
        "max_tile": max(0 if cell == 0 else 1 << cell for cell in observation.board),
        "steps": observation.step_id,
        "terminated": observation.terminated,
        "truncated": observation.truncated,
        "global_env_step": global_step,
        "counter_delta": _counter_delta(counters_before, counters_after),
        "counters": counters_after,
        "learner_state_hash": agent.learner.state_hash(),
        "environment_rng_lineage": dict(snapshot.rng.lineage),
    }


def replay_confirmation_lineage(shard_directory: str | Path) -> dict[str, Any]:
    """Replay one episode-40→200 lineage from its immutable checkpoint pair."""

    shard = Path(shard_directory)
    config = resolve_confirmation_config(_read_json(shard / "resolved-config.json"))
    manifest = _read_json(shard / "shard-manifest.json")
    candidate_id = cast(str, manifest["candidate_id"])
    seed = cast(str, manifest["training_seed"])
    records = _read_jsonl(
        _path_in_shard(
            shard, manifest.get("training_record_path"), field="lineage training_record_path"
        )
    )
    checkpoints = cast(Mapping[str, Any], manifest["checkpoints"])
    checkpoint_40 = cast(Mapping[str, Any], checkpoints["episode_40"])
    checkpoint_200 = cast(Mapping[str, Any], checkpoints["episode_200"])
    if len(records) != config.training_target_episode:
        raise ArtifactError("confirmation lineage training record count is incomplete")
    candidate = next(item for item in config.candidates if item.id == candidate_id)
    execution = _build_agent
    from strategy2048.experiments.confirmation import _confirmation_execution_config

    run_config = _confirmation_execution_config(config, candidate, seed)
    agent = execution(run_config, cast(DiscoveryArmConfig, candidate))
    step_40 = checkpoint_40.get("global_env_step")
    if type(step_40) is not int or step_40 < 0:
        raise ArtifactError("confirmation lineage checkpoint 40 step is invalid")
    agent.restore_checkpoint(
        _path_in_shard(
            shard,
            checkpoint_40.get("checkpoint_directory"),
            field="lineage checkpoint 40 directory",
        ),
        step_40,
        config_hash=confirmation_config_hash(config),
    )
    global_step = step_40
    final_environment: Mapping[str, Any] | None = None
    for episode_id in range(40, config.training_target_episode):
        if episode_id >= len(records):
            raise ArtifactError("confirmation lineage is missing a training record")
        expected = records[episode_id]
        counters_before = agent.counters.to_json()
        env = OracleEnv(
            root_seed=seed,
            environment_id=f"{config.experiment_id}-training",
            max_steps=config.max_steps_per_episode,
        )
        observation = env.reset(episode_id=episode_id, purpose="train-env")
        while not observation.terminated and not observation.truncated:
            action = agent.learner.choose_action(observation)
            transition = env.step(action)
            agent.learner.observe(transition, transition.observation)
            observation = transition.observation
        global_step += observation.step_id
        actual = _lineage_actual_fields(
            agent, observation, global_step, counters_before, env.snapshot()
        )
        for field, value in actual.items():
            if canonical_json(expected.get(field)) != canonical_json(value):
                raise ArtifactError(
                    f"confirmation replay mismatch: {candidate_id}/{seed}/{episode_id}/{field}"
                )
        final_environment = env.snapshot().to_json()
    if final_environment is None:
        raise ArtifactError("confirmation replay produced no final environment")
    checks = {
        "global_env_step": checkpoint_200.get("global_env_step") == global_step,
        "learner_state_hash": checkpoint_200.get("learner_state_hash")
        == agent.learner.state_hash(),
        "table_hash": checkpoint_200.get("table_hash") == agent.learner.table_hash(),
        "counters": canonical_json(checkpoint_200.get("counters"))
        == canonical_json(agent.counters.to_json()),
        "environment": canonical_json(checkpoint_200.get("environment"))
        == canonical_json(final_environment),
    }
    if not all(checks.values()):
        raise ArtifactError(
            f"confirmation final checkpoint mismatch: {candidate_id}/{seed}: "
            + ", ".join(sorted(name for name, valid in checks.items() if not valid))
        )
    return {
        "schema_version": "oi-baseline-confirmation-lineage-proof-v1",
        "candidate_id": candidate_id,
        "training_seed": seed,
        "start_checkpoint_episode": 40,
        "end_checkpoint_episode": config.training_target_episode,
        "replayed_episode_start": 40,
        "replayed_episode_end_inclusive": config.training_target_episode - 1,
        "replayed_episode_count": config.training_target_episode - 40,
        "start_learner_state_hash": checkpoint_40.get("learner_state_hash"),
        "end_learner_state_hash": checkpoint_200.get("learner_state_hash"),
        "end_table_hash": checkpoint_200.get("table_hash"),
        "verified": True,
    }


def replay_confirmation_lineages(
    source_directory: str | Path, *, max_workers: int = 1
) -> dict[str, Any]:
    """Replay every completed shard, sorting proofs independent of completion order."""

    if type(max_workers) is not int or max_workers <= 0:
        raise ArtifactError("confirmation replay worker count must be positive")
    source = _source_root(source_directory)
    config = resolve_confirmation_config(_read_json(source / "resolved-config.json"))
    shard_paths = sorted(
        source / "shards" / seed / candidate_id
        for seed in config.training_seeds
        for candidate_id in CONFIRMATION_ARM_IDS
        if (source / "shards" / seed / candidate_id).is_dir()
    )
    if not shard_paths:
        raise ArtifactError("confirmation source contains no completed shards")
    apply_thread_environment()
    if max_workers <= 1:
        proofs = [replay_confirmation_lineage(path) for path in shard_paths]
    else:
        context = multiprocessing.get_context("spawn")
        proofs = []
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=context) as pool:
            futures = [pool.submit(replay_confirmation_lineage, path) for path in shard_paths]
            for future in as_completed(futures):
                proofs.append(future.result())
    proofs.sort(key=lambda item: (str(item["candidate_id"]), str(item["training_seed"])))
    return {
        "schema_version": "oi-baseline-confirmation-lineage-proof-v1",
        "confirmation_lineages": proofs,
        "all_confirm_lineages_verified": all(proof.get("verified") is True for proof in proofs),
    }


def _source_git_commit(source: Path) -> str:
    campaign = _campaign_manifest(source)
    provenance = campaign.get("source_provenance")
    if not isinstance(provenance, Mapping):
        # Keep the error at the contract boundary: source verification can
        # still validate raw records, while a derived contract must prove the
        # immutable source revision and cleanliness.
        raise ArtifactError("source run provenance is missing")
    revision = provenance.get("commit")
    if isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{7,64}", revision):
        if provenance.get("dirty") is not False:
            raise ArtifactError("formal confirmation contract requires a clean source run commit")
        return revision
    raise ArtifactError("source run commit is missing")


def recompute_confirmation_contract(
    source_directory: str | Path,
    *,
    reducer_commit: str,
    reducer_dirty: bool = False,
    replay_workers: int = 1,
) -> dict[str, Any]:
    source = _source_root(source_directory)
    source_before = artifact_tree_sha256(source)
    if reducer_dirty:
        raise ArtifactError("confirmation contract requires a clean reducer commit")
    if re.fullmatch(r"[0-9a-f]{7,64}", reducer_commit) is None:
        raise ArtifactError("confirmation reducer commit is malformed")
    source_report = verify_confirmation_source(source)
    if source_report["valid"] is not True:
        raise ArtifactError("confirmation source is invalid: " + "; ".join(source_report["errors"]))
    config = resolve_confirmation_config(_read_json(source / "resolved-config.json"))
    source_summary = recompute_confirmation_source_summary(source)
    if source_summary["gate"] not in {
        "oi-baseline-confirmed",
        "oi-baseline-rejected",
        "inconclusive",
    }:
        raise ArtifactError("confirmation contract requires a complete decision source")
    shard_projection: list[dict[str, Any]] = []
    for seed in config.training_seeds:
        for candidate_id in CONFIRMATION_ARM_IDS:
            shard = source / "shards" / seed / candidate_id
            if not shard.is_dir():
                continue
            manifest = _read_json(shard / "shard-manifest.json")
            training = _read_jsonl(shard / str(manifest["training_record_path"]))
            evaluation = _read_jsonl(shard / str(manifest["evaluation_record_path"]))
            telemetry = manifest.get("runtime_telemetry")
            if not isinstance(telemetry, Mapping):
                raise ArtifactError("confirmation shard telemetry is missing")
            validate_runtime_telemetry(telemetry, budget_seconds=config.campaign_wall_seconds)
            shard_projection.append(
                {
                    "candidate_id": candidate_id,
                    "training_seed": seed,
                    "scientific_digest": _shard_scientific_digest(manifest, training, evaluation),
                    "training_episode_count": len(training),
                    "evaluation_episode_count": len(evaluation),
                    "learner_state_hash": manifest.get("learner_state_hash"),
                    "table_hash": manifest.get("table_hash"),
                    "rng_lineage": manifest.get("rng_lineage"),
                    "runtime_telemetry": telemetry,
                }
            )
    lineage_proof = replay_confirmation_lineages(source, max_workers=replay_workers)
    source_after = artifact_tree_sha256(source)
    if source_after != source_before:
        raise ArtifactError("confirmation source changed while recomputing the contract")
    value = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "projection_schema_version": PROJECTION_SCHEMA_VERSION,
        "provenance": {
            "source_artifact_tree_sha256": source_before,
            "source_artifact_tree_hash_schema": TREE_HASH_SCHEMA_VERSION,
            "source_run_commit": _source_git_commit(source),
            "source_config_hash": confirmation_config_hash(config),
            "reducer_commit": reducer_commit,
            "reducer_dirty": False,
        },
        "source_result": source_summary,
        "projection": {
            "config": config.to_json(),
            "paired_comparisons": source_summary["paired_records"],
            "shards": shard_projection,
            "scientific_digest": scientific_digest(
                {"source_result": source_summary, "shards": shard_projection}
            ),
        },
        "lineage_proof": lineage_proof,
        "evidence_boundary": {
            "source_artifact_modified": False,
            "formal_training_run_started": True,
            "scientific_sample_count_expanded": False,
            "strong_replay_is_post_run_verification": True,
            "statistical_significance_claimed": False,
        },
    }
    _validate_schema(value)
    return value


def _assert_disjoint(source: Path, destination: Path) -> None:
    if any(part in {"", ".", ".."} for part in destination.parts):
        raise ArtifactError("confirmation destination contains traversal")
    _assert_no_symlink_components(source)
    _assert_no_symlink_components(destination)
    source_resolved = source.resolve(strict=True)
    destination_resolved = destination.resolve(strict=False)
    if (
        source_resolved == destination_resolved
        or source_resolved in destination_resolved.parents
        or destination_resolved in source_resolved.parents
    ):
        raise ArtifactError("confirmation source and destination must be disjoint")


def build_confirmation_contract(
    source_directory: str | Path,
    destination: str | Path,
    *,
    reducer_commit: str,
    reducer_dirty: bool = False,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Build an immutable derived sibling without modifying the source tree."""

    source = _source_root(source_directory)
    target = Path(destination)
    _assert_disjoint(source, target)
    if target.exists() or target.is_symlink():
        raise ArtifactError(f"confirmation contract destination already exists: {target}")
    contract = recompute_confirmation_contract(
        source,
        reducer_commit=reducer_commit,
        reducer_dirty=reducer_dirty,
        replay_workers=replay_workers,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.tmp-", dir=target.parent))
    try:
        _write_atomic_json(temporary / "confirmation-contract.json", contract)
        source_after = artifact_tree_sha256(source)
        if source_after != contract["provenance"]["source_artifact_tree_sha256"]:
            raise ArtifactError("confirmation source changed while building the contract")
        temporary.replace(target)
    except BaseException:
        # Preserve the temporary evidence for diagnosis; callers must not infer
        # a valid bundle from an interrupted publication.
        raise
    return contract


def verify_confirmation_contract(
    source_directory: str | Path,
    destination: str | Path,
    *,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Recompute source, projection, and replay independently of stored flags."""

    try:
        source = _source_root(source_directory)
    except ArtifactError as error:
        return {
            "schema_version": "oi-baseline-confirmation-contract-verification-v1",
            "valid": False,
            "gate": "contract-failed",
            "errors": [str(error)],
            "source_directory": str(source_directory),
            "destination": str(destination),
        }
    target = Path(destination)
    errors: list[str] = []
    try:
        _assert_no_symlink_components(target)
        if target.is_symlink() or not target.is_dir():
            raise ArtifactError("confirmation contract destination is not a regular directory")
        stored = _read_json(target / "confirmation-contract.json")
        reducer_commit = cast(Mapping[str, Any], stored["provenance"])["reducer_commit"]
        if not isinstance(reducer_commit, str):
            raise ArtifactError("confirmation contract reducer commit is missing")
        recomputed = recompute_confirmation_contract(
            source,
            reducer_commit=reducer_commit,
            reducer_dirty=False,
            replay_workers=replay_workers,
        )
        if canonical_json(stored) != canonical_json(recomputed):
            raise ArtifactError("confirmation contract differs from independent recomputation")
    except (ArtifactError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        errors.append(str(error))
    reported_gate = "contract-failed"
    contract_path = target / "confirmation-contract.json"
    if contract_path.is_file() and not contract_path.is_symlink():
        try:
            reported_gate = str(
                _read_json(contract_path).get("source_result", {}).get("gate", "contract-failed")
            )
        except (ArtifactError, json.JSONDecodeError, ValueError):
            reported_gate = "contract-failed"
    return {
        "schema_version": "oi-baseline-confirmation-contract-verification-v1",
        "valid": not errors,
        "gate": reported_gate,
        "errors": errors,
        "source_directory": str(source),
        "destination": str(target),
    }


def write_confirmation_checker_report(
    source_directory: str | Path,
    destination: str | Path,
    report_directory: str | Path,
    *,
    replay_workers: int = 1,
) -> dict[str, Any]:
    """Run the independent checker and atomically publish its sibling report."""

    report_root = Path(report_directory)
    _assert_no_symlink_components(report_root)
    if report_root.exists() or report_root.is_symlink():
        raise ArtifactError(f"confirmation checker destination already exists: {report_root}")
    source = _source_root(source_directory)
    _assert_disjoint(source, report_root)
    _assert_disjoint(Path(destination), report_root)
    report = verify_confirmation_contract(
        source_directory,
        destination,
        replay_workers=replay_workers,
    )
    report_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{report_root.name}.tmp-", dir=report_root.parent))
    try:
        _write_atomic_json(temporary / "verification-report.json", report)
        temporary.replace(report_root)
    except BaseException:
        raise
    return report


__all__ = [
    "CONTRACT_SCHEMA_PATH",
    "CONTRACT_SCHEMA_VERSION",
    "PROJECTION_SCHEMA_VERSION",
    "artifact_tree_sha256",
    "build_confirmation_contract",
    "recompute_confirmation_contract",
    "recompute_confirmation_source_summary",
    "replay_confirmation_lineage",
    "replay_confirmation_lineages",
    "verify_confirmation_contract",
    "verify_confirmation_source",
    "write_confirmation_checker_report",
]
