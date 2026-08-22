from __future__ import annotations

import json
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest

import strategy2048.experiments.confirmation_contract as confirmation_contract
from strategy2048.experiments.confirmation import (
    CONFIRMATION_LEGACY_SEEDS,
    ConfirmationConfigError,
    ConfirmationShardRequest,
    confirmation_config_hash,
    load_confirmation_config,
    reduce_confirmation_gate,
    resolve_confirmation_config,
    run_confirmation_campaign,
    run_confirmation_formal_campaign,
    run_confirmation_scaling_preflight,
    run_confirmation_shard,
    scientific_digest,
    scientific_projection,
    validate_runtime_telemetry,
    verify_confirmation_shard,
)
from strategy2048.experiments.confirmation_contract import replay_confirmation_lineage

REPOSITORY_ROOT = Path(__file__).parents[2]


def _config() -> dict[str, Any]:
    value = load_confirmation_config(
        REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml"
    ).to_json()
    value["max_steps_per_episode"] = 1
    return value


def _paired(seed: str, gain: float = 0.2, tile_delta: float = 0.1) -> dict[str, Any]:
    return {
        "training_seed": seed,
        "zero_mean_score": 100.0,
        "oi_mean_score": 100.0 * (1.0 + gain),
        "zero_256_reach_rate": 0.2,
        "oi_256_reach_rate": 0.2 + tile_delta,
    }


def _preflight_near_elbow_runner(
    *, worker_count: int, fixture_seeds: tuple[str, ...]
) -> dict[str, Any]:
    assert fixture_seeds[0].startswith("confirmation-preflight-")
    return {
        "scientific_digest": "a" * 64,
        "throughput": {1: 100.0, 2: 120.0, 4: 140.0}[worker_count],
        "runtime_telemetry": {"wall_seconds": 1.0, "process_cpu_seconds": 1.0},
    }


def _preflight_no_worker_8_runner(
    *, worker_count: int, fixture_seeds: tuple[str, ...]
) -> dict[str, Any]:
    del fixture_seeds
    if worker_count == 8:
        raise AssertionError("worker 8 must not run without sufficient 4-over-2 gain")
    return {
        "scientific_digest": "b" * 64,
        "throughput": {1: 100.0, 2: 120.0, 4: 120.0}[worker_count],
        "runtime_telemetry": {"wall_seconds": 1.0, "process_cpu_seconds": 1.0},
    }


def _preflight_hanging_runner(
    *, worker_count: int, fixture_seeds: tuple[str, ...]
) -> dict[str, Any]:
    del fixture_seeds
    if worker_count == 1:
        return {
            "scientific_digest": "c" * 64,
            "throughput": 100.0,
            "runtime_telemetry": {
                "wall_seconds": 0.01,
                "process_cpu_seconds": 0.01,
            },
        }
    time.sleep(10.0)
    raise AssertionError("watchdog did not terminate the hanging fixture")


def _formal_fixture_source_runner(
    *,
    config: Any,
    max_workers: int,
    shard_runner: Any,
    artifact_directory: str,
    deadline: float,
) -> dict[str, Any]:
    del config, max_workers, shard_runner, deadline
    source = Path(artifact_directory) / "source"
    source.mkdir(parents=True)
    summary = {
        "schema_version": "oi-baseline-confirmation-source-summary-v1",
        "gate": "oi-baseline-confirmed",
        "stop_reason": "confirmation_gate_reached",
    }
    source.joinpath("source-summary.json").write_text(
        json.dumps(summary, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def _formal_fixture_source_verifier(*, source_directory: str) -> dict[str, Any]:
    assert Path(source_directory, "source-summary.json").is_file()
    return {"valid": True, "errors": []}


def _formal_fixture_contract_runner(
    *,
    source_directory: str,
    destination: str,
    reducer_commit: str,
    reducer_dirty: bool,
    replay_workers: int,
) -> dict[str, Any]:
    del source_directory, reducer_commit, reducer_dirty, replay_workers
    target = Path(destination)
    target.mkdir(parents=True)
    result = {"lineage_proof": {"all_confirm_lineages_verified": True}}
    target.joinpath("confirmation-contract.json").write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def _formal_fixture_checker_runner(
    *,
    source_directory: str,
    destination: str,
    report_directory: str,
    replay_workers: int,
) -> dict[str, Any]:
    del source_directory, destination, replay_workers
    target = Path(report_directory)
    target.mkdir(parents=True)
    result = {"valid": True, "errors": []}
    target.joinpath("verification-report.json").write_text(
        json.dumps(result, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def test_confirmation_config_freezes_fresh_registry_and_round_trip() -> None:
    config = load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml")
    assert len(config.training_seeds) == 8
    assert set(config.legacy_seed_denylist) == set(CONFIRMATION_LEGACY_SEEDS)
    assert config.to_json() == json.loads(json.dumps(config.to_json()))


def test_confirmation_config_rejects_legacy_seed_reuse() -> None:
    value = _config()
    value["training_seeds"][0] = CONFIRMATION_LEGACY_SEEDS[0]
    with pytest.raises(ConfirmationConfigError, match="disjoint"):
        resolve_confirmation_config(value)


def test_gate_requires_minimum_cohort_and_is_predeclared() -> None:
    assert reduce_confirmation_gate([_paired("a"), _paired("b")])["decision"] == "continue"
    result = reduce_confirmation_gate([_paired(str(index)) for index in range(4)])
    assert result["decision"] == "oi-baseline-confirmed"
    rejected = reduce_confirmation_gate(
        [_paired("a", gain=-0.2), _paired("b", gain=-0.25), _paired("c"), _paired("d")]
    )
    assert rejected["decision"] == "oi-baseline-rejected"


def test_scientific_projection_excludes_runtime_observations() -> None:
    value = {
        "score": 10,
        "wall_seconds": 3.0,
        "runtime_telemetry": {"rss_bytes": 10},
        "nested": {"pid": 12, "steps": 4},
    }
    projected = scientific_projection(value)
    assert projected == {"nested": {"steps": 4}, "score": 10}
    assert scientific_digest({**value, "wall_seconds": 99.0}) == scientific_digest(value)


def test_runtime_telemetry_accounts_nested_phase_totals() -> None:
    with pytest.raises(ValueError, match="wall budget"):
        validate_runtime_telemetry(
            {"training": {"wall_seconds": 1000.0}, "audit": {"wall_seconds": 1000.0}},
            budget_seconds=1800.0,
        )


def test_runtime_telemetry_does_not_double_count_aggregate_and_phases() -> None:
    normalized = validate_runtime_telemetry(
        {
            "wall_seconds": 100.0,
            "process_cpu_seconds": 80.0,
            "training_wall_seconds": 60.0,
            "training_process_cpu_seconds": 50.0,
            "evaluation": {"wall_seconds": 40.0, "process_cpu_seconds": 30.0},
        },
        budget_seconds=100.0,
    )
    assert normalized["wall_seconds"] == 100.0
    with pytest.raises(ValueError, match="phase wall time"):
        validate_runtime_telemetry(
            {"wall_seconds": 100.0, "training_wall_seconds": 101.0},
            budget_seconds=100.0,
        )


def test_scaling_preflight_selects_smallest_near_elbow_without_formal_seeds() -> None:
    config = load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml")
    result = run_confirmation_scaling_preflight(config, fixture_runner=_preflight_near_elbow_runner)
    assert result["valid"] is True
    assert result["selected_worker_count"] == 4
    assert result["stop_reason"] == "worker_8_point_requires_explicit_declaration"


def test_scaling_preflight_does_not_run_worker_eight_without_gain() -> None:
    config = load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml")
    result = run_confirmation_scaling_preflight(
        config,
        fixture_runner=_preflight_no_worker_8_runner,
        worker_points=(1, 2, 4, 8),
    )
    assert result["selected_worker_count"] == 2
    assert result["stop_reason"] == "worker_8_point_not_justified_by_worker_4_gain"


def test_scaling_preflight_terminates_a_hanging_worker_point() -> None:
    config = load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml")
    existing_children = {child.pid for child in multiprocessing.active_children()}
    result = run_confirmation_scaling_preflight(
        config,
        fixture_runner=_preflight_hanging_runner,
        hard_cap_seconds=1.5,
    )
    assert result["valid"] is False
    assert [item["worker_count"] for item in result["measurements"]] == [1]
    assert result["selected_worker_count"] is None
    assert result["stop_reason"] == "preflight_worker_point_timed_out"
    assert result["failed_worker_count"] == 2
    remaining_children = {child.pid for child in multiprocessing.active_children()}
    assert remaining_children <= existing_children


def test_formal_campaign_runs_all_phases_under_one_ledger(tmp_path: Path) -> None:
    root = tmp_path / "formal-campaign"
    result = run_confirmation_formal_campaign(
        load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml"),
        artifact_directory=root,
        reducer_commit="a" * 40,
        source_runner=_formal_fixture_source_runner,
        source_verifier=_formal_fixture_source_verifier,
        contract_runner=_formal_fixture_contract_runner,
        checker_runner=_formal_fixture_checker_runner,
    )
    assert result["status"] == "completed"
    assert result["gate"] == "oi-baseline-confirmed"
    assert (root / "source" / "source-summary.json").is_file()
    assert (root / "derived-contract" / "confirmation-contract.json").is_file()
    assert (root / "independent-check" / "verification-report.json").is_file()
    ledger = [
        json.loads(line)
        for line in (root / "campaign-ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    completed_phases = [event["phase"] for event in ledger if event["event"] == "phase_completed"]
    assert completed_phases == [
        "source-runner",
        "source-verification",
        "derived-contract",
        "independent-check",
    ]


def test_formal_campaign_resume_does_not_reset_consumed_budget(tmp_path: Path) -> None:
    config = load_confirmation_config(REPOSITORY_ROOT / "configs/oi-baseline-confirmation-v1.toml")
    root = tmp_path / "formal-resume"
    root.mkdir()
    records = [
        {
            "schema_version": "oi-baseline-confirmation-ledger-v1",
            "event": "campaign_started",
            "campaign_id": root.name,
            "experiment_id": config.experiment_id,
            "config_hash": confirmation_config_hash(config),
            "campaign_wall_seconds": config.campaign_wall_seconds,
            "finalization_reserve_seconds": config.finalization_reserve_seconds,
            "consumed_wall_seconds": 0.0,
        },
        {
            "schema_version": "oi-baseline-confirmation-ledger-v1",
            "event": "phase_failed",
            "phase": "source-runner",
            "phase_wall_seconds": 1790.0,
            "consumed_wall_seconds": 1790.0,
            "error": "fixture interruption",
        },
    ]
    root.joinpath("campaign-ledger.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    result = run_confirmation_formal_campaign(
        config,
        artifact_directory=root,
        reducer_commit="a" * 40,
        source_runner=_formal_fixture_source_runner,
        source_verifier=_formal_fixture_source_verifier,
        contract_runner=_formal_fixture_contract_runner,
        checker_runner=_formal_fixture_checker_runner,
        resume=True,
    )
    assert result["status"] == "performance-blocked"
    assert result["gate"] == "performance-blocked"
    assert result["consumed_wall_seconds"] >= 1790.0
    assert not (root / "source").exists()


def test_campaign_rejects_worker_override_before_creating_artifacts(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    with pytest.raises(ValueError, match="worker_count override"):
        run_confirmation_campaign(_config(), artifact_directory=root, max_workers=2)
    assert not root.exists()


def test_campaign_verifies_raw_source_before_publishing_attestation(tmp_path: Path) -> None:
    root = tmp_path / "campaign"
    summary = run_confirmation_campaign(
        _config(),
        artifact_directory=root,
        max_workers=1,
        shard_runner=run_confirmation_shard,
    )
    assert summary["gate"] == "oi-baseline-rejected"
    attestation = json.loads(
        (root / "source" / "source-verification.json").read_text(encoding="utf-8")
    )
    assert attestation["valid"] is True


def test_contract_rejects_source_mutation_during_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "campaign"
    summary = run_confirmation_campaign(
        _config(),
        artifact_directory=root,
        max_workers=1,
        shard_runner=run_confirmation_shard,
    )
    assert summary["gate"] == "oi-baseline-rejected"
    campaign_manifest_path = root / "campaign-manifest.json"
    campaign_manifest = json.loads(campaign_manifest_path.read_text(encoding="utf-8"))
    campaign_manifest["source_provenance"] = {"commit": "a" * 40, "dirty": False}
    campaign_manifest_path.write_text(json.dumps(campaign_manifest) + "\n", encoding="utf-8")

    original_replay = confirmation_contract.replay_confirmation_lineages

    def mutate_before_replay(
        source_directory: str | Path, *, max_workers: int = 1
    ) -> dict[str, Any]:
        Path(source_directory).joinpath("mutation-marker").write_text("changed\n", encoding="utf-8")
        return original_replay(source_directory, max_workers=max_workers)

    monkeypatch.setattr(confirmation_contract, "replay_confirmation_lineages", mutate_before_replay)
    with pytest.raises(ValueError, match="source changed while recomputing"):
        confirmation_contract.recompute_confirmation_contract(
            root / "source", reducer_commit="b" * 40
        )


def test_one_shard_is_atomically_published_and_replayed(tmp_path: Path) -> None:
    config = _config()
    destination = tmp_path / "shards" / "td0_zero"
    request = ConfirmationShardRequest(
        config=config,
        candidate_id="td0_zero",
        training_seed=config["training_seeds"][0],
        destination=str(destination),
        deadline_seconds=120.0,
    )
    run_confirmation_shard(request)
    report = verify_confirmation_shard(destination)
    assert report["valid"] is True
    proof = replay_confirmation_lineage(destination)
    assert proof["verified"] is True
    with pytest.raises(ValueError, match="already exists"):
        run_confirmation_shard(request)


def test_shard_resume_reuses_partial_checkpoint_and_budget(tmp_path: Path) -> None:
    config = _config()
    destination = tmp_path / "resumed"
    interrupted = ConfirmationShardRequest(
        config=config,
        candidate_id="td0_zero",
        training_seed=config["training_seeds"][0],
        destination=str(destination),
        deadline_seconds=0.0001,
    )
    with pytest.raises(ValueError):
        run_confirmation_shard(interrupted)
    partial = next(tmp_path.glob(".resumed.attempt-*"))
    partial_manifest = json.loads(
        partial.joinpath("partial-shard-manifest.json").read_text(encoding="utf-8")
    )
    assert partial_manifest["status"] == "interrupted"
    resumed = ConfirmationShardRequest(
        config=config,
        candidate_id="td0_zero",
        training_seed=config["training_seeds"][0],
        destination=str(destination),
        deadline_seconds=120.0,
        resume_from=str(partial),
    )
    run_confirmation_shard(resumed)
    assert verify_confirmation_shard(destination)["valid"] is True


def test_shard_verifier_replays_checkpoint_contract(tmp_path: Path) -> None:
    config = _config()
    destination = tmp_path / "tampered"
    run_confirmation_shard(
        ConfirmationShardRequest(
            config=config,
            candidate_id="td0_zero",
            training_seed=config["training_seeds"][0],
            destination=str(destination),
            deadline_seconds=120.0,
        )
    )
    manifest = json.loads(destination.joinpath("shard-manifest.json").read_text(encoding="utf-8"))
    metadata_path = destination / manifest["checkpoints"]["episode_40"]["metadata_path"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["checkpoint_hash"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata) + "\n", encoding="utf-8")
    report = verify_confirmation_shard(destination)
    assert report["valid"] is False
