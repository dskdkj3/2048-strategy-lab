from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from strategy2048.experiments.artifacts import ArtifactError, canonical_json
from strategy2048.experiments.calibration import (
    CALIBRATION_SCHEMA_PATH,
    CalibrationConfigError,
    compute_tuning_context_fingerprint,
    derive_screen_decision,
    encode_afterstate_u64,
    load_calibration_config,
    recompute_calibration_summary,
    resolve_calibration_config,
    run_algorithm_calibration,
    verify_calibration_artifact,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def _tiny_config(*, experiment_id: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "schema_version": "algorithm-calibration-v1",
        "experiment_id": experiment_id,
        "output_root": "artifacts",
        "shared_wall_seconds": 600,
        "finalization_reserve_seconds": 10.0,
        "screen_wall_seconds": 270,
        "round_robin_training_chunk": 10,
        "training_seeds": ["calibration-train-a-v1", "calibration-train-b-v1"],
        "selection_evaluation_root_seed": "calibration-selection-v1",
        "audit_evaluation_root_seed": "calibration-audit-v1",
        "screen_target_episode": 40,
        "confirm_target_episode": 200,
        "screen_evaluation_episodes": 20,
        "audit_evaluation_episodes": 50,
        "exploration_sample_stride": 1,
        "exploration_overhead_limit": 0.02,
        "incumbent_candidate_id": "td0_zero",
        "parent_calibration_id": "discovery-pilot-v1",
        "candidate_generation_rule": "fixed-log-grid-v1",
        "tuning_context_fingerprint": "0" * 64,
        "max_steps_per_episode": 1,
        "learner": {
            "alpha": 0.1,
            "gamma": 1.0,
            "symmetry": False,
            "value_cardinality": 4,
            "tuples": [[0, 1]],
        },
        "candidates": [
            {
                "id": "td0_zero",
                "initialization": "zero",
                "optimistic_total_value": 0.0,
            },
            {
                "id": "td0_oi_300",
                "initialization": "optimistic",
                "optimistic_total_value": 300.0,
            },
            {
                "id": "td0_oi_1000",
                "initialization": "optimistic",
                "optimistic_total_value": 1000.0,
            },
            {
                "id": "td0_oi_3000",
                "initialization": "optimistic",
                "optimistic_total_value": 3000.0,
            },
            {
                "id": "td0_oi_10000",
                "initialization": "optimistic",
                "optimistic_total_value": 10000.0,
            },
        ],
    }
    config["tuning_context_fingerprint"] = compute_tuning_context_fingerprint(config)
    return config


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


class AdvancingClock:
    def __init__(self, step: float = 1.0) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


def _selection_records(config: dict[str, Any]) -> list[dict[str, Any]]:
    scores = {
        "td0_zero": (100, 100),
        "td0_oi_300": (80, 80),
        "td0_oi_1000": (105, 100),
        "td0_oi_3000": (110, 90),
        "td0_oi_10000": (95, 95),
    }
    records: list[dict[str, Any]] = []
    for candidate_id, per_seed in scores.items():
        for seed_index, training_seed in enumerate(config["training_seeds"]):
            for episode_id in range(20):
                records.append(
                    {
                        "candidate_id": candidate_id,
                        "training_seed": training_seed,
                        "suite": "selection",
                        "checkpoint_episode": 40,
                        "evaluation_episode_id": episode_id,
                        "official_score": per_seed[seed_index],
                    }
                )
    return records


def test_production_config_is_the_fixed_600_second_contract() -> None:
    config = load_calibration_config(REPOSITORY_ROOT / "configs/algorithm-calibration-v1.toml")
    resolved = config.to_json()
    jsonschema.validate(resolved, _read_json(CALIBRATION_SCHEMA_PATH))

    assert resolved["shared_wall_seconds"] == 600
    assert resolved["screen_wall_seconds"] == 270
    assert resolved["screen_target_episode"] == 40
    assert resolved["confirm_target_episode"] == 200
    assert resolved["screen_evaluation_episodes"] == 20
    assert resolved["audit_evaluation_episodes"] == 50
    assert [item["optimistic_total_value"] for item in resolved["candidates"]] == [
        0.0,
        300.0,
        1000.0,
        3000.0,
        10000.0,
    ]


def test_config_rejects_legacy_value_seed_collision_and_matrix_drift() -> None:
    legacy = _tiny_config(experiment_id="legacy")
    legacy["candidates"][1]["optimistic_value"] = 300.0
    with pytest.raises(CalibrationConfigError, match="optimistic_total_value"):
        resolve_calibration_config(legacy)

    collision = _tiny_config(experiment_id="collision")
    collision["audit_evaluation_root_seed"] = collision["training_seeds"][0]
    collision["tuning_context_fingerprint"] = compute_tuning_context_fingerprint(collision)
    with pytest.raises(CalibrationConfigError, match="pairwise distinct"):
        resolve_calibration_config(collision)

    drift = _tiny_config(experiment_id="drift")
    drift["candidates"][1]["optimistic_total_value"] = 301.0
    drift["tuning_context_fingerprint"] = compute_tuning_context_fingerprint(drift)
    with pytest.raises(CalibrationConfigError, match="must be 300"):
        resolve_calibration_config(drift)

    stale_fingerprint = _tiny_config(experiment_id="stale-fingerprint")
    stale_fingerprint["learner"]["alpha"] = 0.2
    with pytest.raises(CalibrationConfigError, match="tuning_context_fingerprint"):
        resolve_calibration_config(stale_fingerprint)


def test_afterstate_encoding_is_stable_and_fails_closed_above_four_bits() -> None:
    board = tuple(range(16))
    encoded = encode_afterstate_u64(board)
    assert encoded == int("fedcba9876543210", 16)
    assert encode_afterstate_u64((0,) * 15 + (1,)) != encode_afterstate_u64((0,) * 16)
    with pytest.raises(ArtifactError, match="outside 0..15"):
        encode_afterstate_u64((16,) + (0,) * 15)


def test_screen_decision_uses_selection_only_and_prioritizes_worst_seed() -> None:
    raw = _tiny_config(experiment_id="selection")
    config = resolve_calibration_config(raw)
    records = _selection_records(raw)
    records.append(
        {
            "candidate_id": "td0_oi_300",
            "training_seed": raw["training_seeds"][0],
            "suite": "audit",
            "checkpoint_episode": 200,
            "evaluation_episode_id": 0,
            "official_score": 1_000_000,
        }
    )

    decision = derive_screen_decision(config, records)

    assert decision["survivor_candidate_id"] == "td0_oi_1000"
    by_id = {item["candidate_id"]: item for item in decision["candidate_results"]}
    assert by_id["td0_oi_300"]["eliminated"] is True
    assert by_id["td0_oi_3000"]["eliminated"] is False
    assert decision["selection_input_record_count"] == 200


def test_tiny_run_is_raw_derived_and_independently_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "calibration"
    summary = run_algorithm_calibration(
        _tiny_config(experiment_id="tiny-calibration"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )

    assert summary["stop_reason"] == "completed"
    assert summary["screen_complete"] is True
    assert summary["gate"] in {
        "oi-candidate-recommended",
        "zero-retained",
        "inconclusive",
    }
    assert summary["evidence_boundary"]["statistical_significance_claimed"] is False
    assert summary["exploration"]["selection_uses_exploration_metric"] is False
    assert canonical_json(summary) == canonical_json(recompute_calibration_summary(root))
    report = verify_calibration_artifact(root)
    assert report == {
        "schema_version": "algorithm-calibration-verification-v1",
        "valid": True,
        "gate": summary["gate"],
        "errors": [],
        "artifact_directory": str(root),
    }


def test_interrupted_run_resumes_same_artifact_and_same_budget(tmp_path: Path) -> None:
    root = tmp_path / "resume-calibration"
    fired = False

    def interrupt_once(phase: str) -> None:
        nonlocal fired
        if phase == "env_step" and not fired:
            fired = True
            signal.raise_signal(signal.SIGINT)

    interrupted = run_algorithm_calibration(
        _tiny_config(experiment_id="resume-calibration"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=interrupt_once,
    )
    assert interrupted["stop_reason"] == "interrupted"
    assert verify_calibration_artifact(root)["valid"] is True

    completed = run_algorithm_calibration(
        _tiny_config(experiment_id="resume-calibration"),
        resume_from=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )
    assert completed["stop_reason"] == "completed"
    assert verify_calibration_artifact(root)["valid"] is True

    uninterrupted_root = tmp_path / "uninterrupted-calibration"
    run_algorithm_calibration(
        _tiny_config(experiment_id="resume-calibration"),
        artifact_directory=uninterrupted_root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )
    resumed_coverage = sorted(root.glob("runs/*/*/coverage/latest.json"))
    uninterrupted_coverage = sorted(uninterrupted_root.glob("runs/*/*/coverage/latest.json"))
    assert [path.relative_to(root) for path in resumed_coverage] == [
        path.relative_to(uninterrupted_root) for path in uninterrupted_coverage
    ]
    for resumed_path, uninterrupted_path in zip(
        resumed_coverage, uninterrupted_coverage, strict=True
    ):
        resumed_value = _read_json(resumed_path)
        uninterrupted_value = _read_json(uninterrupted_path)
        for field in (
            "observed_steps",
            "sampled_observations",
            "distinct_sampled_afterstates",
            "first_visit_ratio",
            "action_distribution",
            "coverage_sha256",
            "sampled_afterstates",
        ):
            assert resumed_value[field] == uninterrupted_value[field]


def test_screen_deadline_yields_valid_performance_blocked_artifact(tmp_path: Path) -> None:
    root = tmp_path / "budget-calibration"
    summary = run_algorithm_calibration(
        _tiny_config(experiment_id="budget-calibration"),
        artifact_directory=root,
        clock=AdvancingClock(),
        process_clock=lambda: 0.0,
    )

    assert summary["stop_reason"] == "budget_exhausted"
    assert summary["gate"] == "performance-blocked"
    assert float(summary["consumed_wall_seconds"]) < 600.0
    assert verify_calibration_artifact(root)["valid"] is True


def test_verifier_rejects_selection_seed_relabeling(tmp_path: Path) -> None:
    root = tmp_path / "corrupt-calibration"
    run_algorithm_calibration(
        _tiny_config(experiment_id="corrupt-calibration"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )
    path = next(root.glob("runs/*/*/evaluation/selection/40/episodes.jsonl"))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    records[0]["evaluation_root_seed"] = "calibration-audit-v1"
    path.write_text("\n".join(canonical_json(item) for item in records) + "\n", encoding="utf-8")

    report = verify_calibration_artifact(root)
    assert report["valid"] is False
    assert report["gate"] == "contract-failed"
    assert "seed suite mismatch" in report["errors"][0]
