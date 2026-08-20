from __future__ import annotations

import json
import shutil
import signal
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from strategy2048.experiments.artifacts import ArtifactError, canonical_json
from strategy2048.experiments.discovery import (
    DISCOVERY_GATES,
    DISCOVERY_SCHEMA_PATH,
    DiscoveryConfigError,
    _derive_next_step_decision,
    classify_discovery_result,
    load_discovery_config,
    recompute_discovery_summary,
    resolve_discovery_config,
    run_discovery_pilot,
    verify_discovery_artifact,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def _tiny_config(*, experiment_id: str) -> dict[str, Any]:
    return {
        "schema_version": "discovery-pilot-v1",
        "experiment_id": experiment_id,
        "output_root": "artifacts",
        "shared_wall_seconds": 900,
        "finalization_reserve_seconds": 10.0,
        "round_robin_training_chunk": 10,
        "training_seeds": ["train-a-v1", "train-b-v1"],
        "evaluation_root_seed": "eval-root-v1",
        "checkpoint_episodes": [0, 50, 200],
        "evaluation_episodes_per_checkpoint": 1,
        "diagnostic_score_milestone": 1,
        "diagnostic_tile_milestone": 128,
        "max_training_episodes_per_run": 200,
        "max_steps_per_episode": 1,
        "learner": {
            "alpha": 0.1,
            "gamma": 1.0,
            "symmetry": False,
            "value_cardinality": 4,
            "tuples": [[0, 1]],
        },
        "arms": [
            {
                "id": "td0_zero",
                "initialization": "zero",
                "optimistic_total_value": 0.0,
            },
            {
                "id": "td0_optimistic",
                "initialization": "optimistic",
                "optimistic_total_value": 10.0,
            },
        ],
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def completed_artifact(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("discovery-complete") / "pilot"
    summary = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-complete"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )
    assert summary["stop_reason"] == "completed"
    return root


class AdvancingClock:
    def __init__(self, step: float) -> None:
        self.value = 0.0
        self.step = step

    def __call__(self) -> float:
        current = self.value
        self.value += self.step
        return current


class InterruptingClock:
    def __init__(self, interrupt_at: int, step: float = 0.001) -> None:
        self.calls = 0
        self.interrupt_at = interrupt_at
        self.step = step
        self.value = 0.0
        self.interrupted = False

    def __call__(self) -> float:
        self.calls += 1
        if self.calls == self.interrupt_at and not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        current = self.value
        self.value += self.step
        return current


class CooperativeSignal:
    def __init__(self, phase: str, *, occurrence: int = 1) -> None:
        self.phase = phase
        self.occurrence = occurrence
        self.seen = 0
        self.fired = False

    def __call__(self, phase: str) -> None:
        if phase != self.phase or self.fired:
            return
        self.seen += 1
        if self.seen == self.occurrence:
            self.fired = True
            signal.raise_signal(signal.SIGINT)


class SlowPhase:
    def __init__(self, clock: AdvancingClock, phase: str, jump: float) -> None:
        self.clock = clock
        self.phase = phase
        self.jump = jump
        self.fired = False

    def __call__(self, phase: str) -> None:
        if phase == self.phase and not self.fired:
            self.fired = True
            self.clock.value += self.jump


def test_production_config_resolves_to_canonical_versioned_contract() -> None:
    config = load_discovery_config(REPOSITORY_ROOT / "configs/discovery-pilot-v1.toml")
    resolved = config.to_json()
    schema = _read_json(DISCOVERY_SCHEMA_PATH)

    jsonschema.validate(resolved, schema)
    assert resolved["shared_wall_seconds"] == 900
    assert resolved["finalization_reserve_seconds"] == 10.0
    assert resolved["checkpoint_episodes"] == [0, 50, 200]
    assert resolved["max_training_episodes_per_run"] == 200
    assert resolved["diagnostic_score_milestone"] == 5000
    assert resolved["diagnostic_tile_milestone"] == 256
    assert resolved["training_seeds"] == [
        "discovery-train-a-v1",
        "discovery-train-b-v1",
    ]
    assert [arm["id"] for arm in resolved["arms"]] == [
        "td0_zero",
        "td0_optimistic",
    ]
    assert resolved["arms"][1]["optimistic_total_value"] == 10_000.0


def test_config_fails_closed_on_legacy_oi_or_seed_collisions() -> None:
    legacy = _tiny_config(experiment_id="legacy")
    legacy["arms"][1]["optimistic_value"] = 10.0
    with pytest.raises(DiscoveryConfigError, match="optimistic_total_value"):
        resolve_discovery_config(legacy)

    collision = _tiny_config(experiment_id="collision")
    collision["evaluation_root_seed"] = collision["training_seeds"][0]
    with pytest.raises(DiscoveryConfigError, match="separate"):
        resolve_discovery_config(collision)

    extra_seed = _tiny_config(experiment_id="extra-seed")
    extra_seed["training_seeds"].append("train-c-v1")
    with pytest.raises(DiscoveryConfigError, match="schema validation"):
        resolve_discovery_config(extra_seed)


@pytest.mark.parametrize("field", ["alpha", "gamma"])
def test_config_rejects_non_finite_learner_parameters(field: str) -> None:
    invalid = _tiny_config(experiment_id=f"nan-{field}")
    invalid["learner"][field] = float("nan")

    with pytest.raises(DiscoveryConfigError, match=f"learner.{field}"):
        resolve_discovery_config(invalid)


def test_runner_uses_milestone_and_episode_round_robin(completed_artifact: Path) -> None:
    checkpoints = [
        record
        for record in _read_jsonl(completed_artifact / "checkpoints.jsonl")
        if record["kind"] == "milestone"
    ]
    assert [record["checkpoint_episode"] for record in checkpoints] == [
        *([0] * 4),
        *([50] * 4),
        *([200] * 4),
    ]

    progress = _read_jsonl(completed_artifact / "progress.jsonl")
    training = [record for record in progress if record["event"] == "training_episode_completed"]
    expected_first_round = [
        *([("td0_zero", "train-a-v1")] * 10),
        *([("td0_optimistic", "train-a-v1")] * 10),
        *([("td0_zero", "train-b-v1")] * 10),
        *([("td0_optimistic", "train-b-v1")] * 10),
    ]
    assert [(record["arm_id"], record["training_seed"]) for record in training[:40]] == (
        expected_first_round
    )

    evaluation = [
        record for record in progress if record["event"] == "evaluation_episode_completed"
    ]
    assert [
        (record["checkpoint_episode"], record["arm_id"], record["training_seed"])
        for record in evaluation
    ] == [
        (checkpoint, arm, seed)
        for checkpoint in (0, 50, 200)
        for arm, seed in (
            ("td0_zero", "train-a-v1"),
            ("td0_optimistic", "train-a-v1"),
            ("td0_zero", "train-b-v1"),
            ("td0_optimistic", "train-b-v1"),
        )
    ]


def test_frozen_checkpoint_records_and_summary_are_recomputable(
    completed_artifact: Path,
) -> None:
    root_manifest = _read_json(completed_artifact / "knowledge-manifest.json")
    assert root_manifest["initialization"]["optimistic_total_value"] == 10.0
    assert root_manifest["initialization"]["active_feature_count"] == 1
    assert root_manifest["initialization"]["initial_feature_value"] == 10.0

    evaluation_records: list[dict[str, Any]] = []
    for path in sorted(completed_artifact.glob("runs/*/*/evaluation/*/episodes.jsonl")):
        evaluation_records.extend(_read_jsonl(path))

    assert len(evaluation_records) == 12
    assert all(record["evaluation_root_seed"] == "eval-root-v1" for record in evaluation_records)
    assert all(record["purpose"] == "discovery-eval" for record in evaluation_records)
    assert all(record["frozen_state_unchanged"] is True for record in evaluation_records)
    assert all(
        record["clone_state_hash_before"] == record["clone_state_hash_after"]
        for record in evaluation_records
    )
    assert all(
        record["training_state_hash_before"] == record["training_state_hash_after"]
        for record in evaluation_records
    )
    stored = _read_json(completed_artifact / "pilot-summary.json")
    recomputed = recompute_discovery_summary(completed_artifact)
    assert canonical_json(stored) == canonical_json(recomputed)
    assert stored["raw_record_counts"] == {
        "training_episodes": 800,
        "training_metrics": 12,
        "evaluation_episodes": 12,
        "checkpoints": 12,
        "progress": 826,
    }
    for run in stored["runs"].values():
        metrics = run["training_metrics"]
        assert metrics["wall_seconds"]["feature_value_lookup"] > 0
        assert metrics["wall_seconds"]["td_update"] > 0
        assert metrics["wall_seconds"]["checkpoint"] > 0
        assert metrics["rates"]["end_to_end_env_steps_per_second"] > 0
        assert run["milestone_efficiency"]["score"]["target"] == 1
        assert run["milestone_efficiency"]["score"]["status"] == "attained"
        assert run["milestone_efficiency"]["tile"]["status"] == "not-attained"
    decision = stored["next_step_decision"]
    assert decision["decision"] == "continue-algorithm"
    assert decision["assumed_hot_path_speedup"] == 2.0
    assert decision["native_core_gain_threshold"] == 0.3


def test_shared_deadline_writes_partial_checkpoint_and_budget_gate(tmp_path: Path) -> None:
    root = tmp_path / "budgeted"
    clock = AdvancingClock(step=10.0)
    summary = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-budget"),
        artifact_directory=root,
        clock=clock,
        process_clock=lambda: 0.0,
    )

    assert summary["stop_reason"] == "contract_failed"
    assert summary["gate"] == "contract-failed"
    assert summary["consumed_wall_seconds"] > 900.0
    progress = _read_jsonl(root / "progress.jsonl")
    assert sum(record["event"] == "pilot_started" for record in progress) == 1
    exhausted = [record for record in progress if record["event"] == "budget_exhausted"]
    assert exhausted
    assert exhausted[-1]["phase"] in {
        "training",
        "evaluation",
        "checkpoint",
        "checkpoint_durability",
        "resume_checkpoint",
        "finalization",
    }
    assert list(root.glob("runs/*/*/partial-checkpoints/**/*.json")) or list(
        root.glob("runs/*/*/evaluation/*/partials.jsonl")
    )
    stopped = next(record for record in reversed(progress) if record["event"] == "pilot_stopped")
    assert (
        stopped["budget_accounting"] == "measured-work-plus-finalization-plus-final-write-reserve"
    )
    assert stopped["planned_finalization_reserve_seconds"] == 10.0
    assert stopped["consumed_wall_seconds"] == (
        stopped["prior_consumed_wall_seconds"]
        + stopped["measured_segment_wall_seconds"]
        + stopped["measured_finalization_wall_seconds"]
        + stopped["charged_final_write_reserve_seconds"]
    )
    assert stopped["hard_deadline_overrun_seconds"] == (stopped["consumed_wall_seconds"] - 900.0)
    assert verify_discovery_artifact(root)["valid"] is True


def test_resume_with_no_remaining_budget_does_not_duplicate_partial_metrics(
    tmp_path: Path,
) -> None:
    root = tmp_path / "zero-remaining-resume"
    clock = AdvancingClock(step=1.0)
    first = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-zero-remaining-resume"),
        artifact_directory=root,
        clock=clock,
        process_clock=lambda: 0.0,
    )
    metrics_before = {
        path: path.read_text(encoding="utf-8") for path in root.glob("runs/*/*/metrics.jsonl")
    }

    resumed = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-zero-remaining-resume"),
        resume_from=root,
        clock=clock,
        process_clock=lambda: 0.0,
    )

    assert first["stop_reason"] == "budget_exhausted"
    assert resumed["stop_reason"] == "budget_exhausted"
    assert {
        path: path.read_text(encoding="utf-8") for path in root.glob("runs/*/*/metrics.jsonl")
    } == metrics_before
    report = verify_discovery_artifact(root)
    assert report["valid"] is True


def test_verify_is_read_only_and_tampering_fails_closed(completed_artifact: Path) -> None:
    before = {
        path.relative_to(completed_artifact): path.read_bytes()
        for path in completed_artifact.rglob("*")
        if path.is_file()
    }
    report = verify_discovery_artifact(completed_artifact)
    after = {
        path.relative_to(completed_artifact): path.read_bytes()
        for path in completed_artifact.rglob("*")
        if path.is_file()
    }
    assert report["valid"] is True
    assert report["read_only"] is True
    assert after == before

    summary_path = completed_artifact / "pilot-summary.json"
    summary = _read_json(summary_path)
    summary["raw_record_counts"]["evaluation_episodes"] += 1
    summary_path.write_text(canonical_json(summary) + "\n", encoding="utf-8")
    tampered_before = summary_path.read_bytes()
    report = verify_discovery_artifact(completed_artifact)

    assert report["valid"] is False
    assert report["gate"] == "contract-failed"
    assert any("not recomputable" in error for error in report["errors"])
    assert summary_path.read_bytes() == tampered_before


@pytest.mark.parametrize(
    "phase",
    [
        "env_step",
        "observe",
        "training_jsonl",
        "training_progress",
        "checkpoint",
        "finalization",
    ],
)
def test_real_interrupt_is_verifiable_and_resume_matches_uninterrupted_state(
    tmp_path: Path, phase: str
) -> None:
    config = _tiny_config(experiment_id=f"tiny-interrupted-{phase}")
    interrupted_root = tmp_path / f"interrupted-{phase}"
    uninterrupted_root = tmp_path / f"uninterrupted-{phase}"
    clock = AdvancingClock(step=0.001)
    interrupted = run_discovery_pilot(
        config,
        artifact_directory=interrupted_root,
        clock=clock,
        process_clock=lambda: 0.0,
        phase_hook=CooperativeSignal(phase),
    )

    assert interrupted["stop_reason"] == "interrupted"
    assert interrupted["consumed_wall_seconds"] < 900.0
    assert verify_discovery_artifact(interrupted_root)["valid"] is True
    assert list(interrupted_root.glob("runs/*/*/resume-checkpoint.json"))
    assert any(
        record["event"] == "interrupted"
        for record in _read_jsonl(interrupted_root / "progress.jsonl")
    )

    resumed = run_discovery_pilot(
        config,
        resume_from=interrupted_root,
        clock=clock,
        process_clock=lambda: 0.0,
    )
    uninterrupted = run_discovery_pilot(
        config,
        artifact_directory=uninterrupted_root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )

    assert resumed["stop_reason"] == "completed"
    assert uninterrupted["stop_reason"] == "completed"
    assert resumed["consumed_wall_seconds"] > interrupted["consumed_wall_seconds"]
    assert resumed["consumed_wall_seconds"] <= 900.0
    resumed_final = {
        (record["arm_id"], record["training_seed"]): (
            record["learner_state_hash"],
            record["table_hash"],
            record["environment"],
        )
        for record in _read_jsonl(interrupted_root / "checkpoints.jsonl")
        if record["kind"] == "milestone" and record["checkpoint_episode"] == 200
    }
    uninterrupted_final = {
        (record["arm_id"], record["training_seed"]): (
            record["learner_state_hash"],
            record["table_hash"],
            record["environment"],
        )
        for record in _read_jsonl(uninterrupted_root / "checkpoints.jsonl")
        if record["kind"] == "milestone" and record["checkpoint_episode"] == 200
    }
    assert resumed_final == uninterrupted_final
    assert verify_discovery_artifact(interrupted_root)["valid"] is True


def test_mid_chunk_interrupt_resume_preserves_round_robin_raw_sequence(
    tmp_path: Path,
) -> None:
    config = _tiny_config(experiment_id="tiny-mid-chunk-interrupted")
    interrupted_root = tmp_path / "mid-chunk-interrupted"
    uninterrupted_root = tmp_path / "mid-chunk-uninterrupted"

    interrupted = run_discovery_pilot(
        config,
        artifact_directory=interrupted_root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=CooperativeSignal("training_progress", occurrence=26),
    )

    assert interrupted["stop_reason"] == "interrupted"
    interrupted_progress = _read_jsonl(interrupted_root / "progress.jsonl")
    interrupted_stop = next(
        record for record in reversed(interrupted_progress) if record["event"] == "pilot_stopped"
    )
    assert interrupted_stop["scheduler_cursor"] == {
        "phase": "training",
        "checkpoint_episode": 50,
        "arm_id": "td0_zero",
        "training_seed": "train-b-v1",
        "completed_training_episodes": 6,
        "global_env_step": 6,
    }
    assert verify_discovery_artifact(interrupted_root)["valid"] is True

    resumed = run_discovery_pilot(
        config,
        resume_from=interrupted_root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )
    uninterrupted = run_discovery_pilot(
        config,
        artifact_directory=uninterrupted_root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )

    assert resumed["stop_reason"] == "completed"
    assert uninterrupted["stop_reason"] == "completed"

    def training_sequence(root: Path) -> list[tuple[str, str, int]]:
        return [
            (record["arm_id"], record["training_seed"], record["episode_id"])
            for record in _read_jsonl(root / "progress.jsonl")
            if record["event"] == "training_episode_completed"
        ]

    assert training_sequence(interrupted_root) == training_sequence(uninterrupted_root)
    resumed_final = {
        (record["arm_id"], record["training_seed"]): (
            record["learner_state_hash"],
            record["table_hash"],
            record["environment"],
        )
        for record in _read_jsonl(interrupted_root / "checkpoints.jsonl")
        if record["kind"] == "milestone" and record["checkpoint_episode"] == 200
    }
    uninterrupted_final = {
        (record["arm_id"], record["training_seed"]): (
            record["learner_state_hash"],
            record["table_hash"],
            record["environment"],
        )
        for record in _read_jsonl(uninterrupted_root / "checkpoints.jsonl")
        if record["kind"] == "milestone" and record["checkpoint_episode"] == 200
    }
    assert resumed_final == uninterrupted_final
    for run_key, resumed_run in resumed["runs"].items():
        resumed_metrics = resumed_run["training_metrics"]
        uninterrupted_metrics = uninterrupted["runs"][run_key]["training_metrics"]
        assert resumed_metrics["counters"] == uninterrupted_metrics["counters"]
        assert resumed_metrics["counters"]["env_steps"] == 200
        assert resumed_metrics["counters"]["games"] == 200
        assert resumed_metrics["counters"]["updates"] == 200
    assert verify_discovery_artifact(interrupted_root)["valid"] is True


def test_verify_rejects_resume_checkpoint_metrics_rollback(tmp_path: Path) -> None:
    root = tmp_path / "resume-metrics-rollback"
    run_discovery_pilot(
        _tiny_config(experiment_id="tiny-resume-metrics-rollback"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=CooperativeSignal("training_progress", occurrence=26),
    )
    pointer_path = root / "runs/td0_zero/train-a-v1/resume-checkpoint.json"
    pointer = _read_json(pointer_path)
    pointer["metrics"]["counters"]["env_steps"] -= 1
    pointer_path.write_text(canonical_json(pointer) + "\n", encoding="utf-8")

    report = verify_discovery_artifact(root)

    assert report["valid"] is False
    assert any("metrics env_steps" in error for error in report["errors"])


@pytest.mark.parametrize(
    "phase", ["evaluation_episode", "checkpoint", "finalization", "finalization_progress"]
)
def test_cooperative_sigint_stops_at_durable_phase_boundary(tmp_path: Path, phase: str) -> None:
    root = tmp_path / f"sigint-{phase}"
    previous_handler = signal.getsignal(signal.SIGINT)
    summary = run_discovery_pilot(
        _tiny_config(experiment_id=f"tiny-sigint-{phase}"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=CooperativeSignal(phase),
    )

    assert summary["stop_reason"] == "interrupted"
    assert verify_discovery_artifact(root)["valid"] is True
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_raw_keyboard_interrupt_fails_closed_and_restores_signal_handler(
    tmp_path: Path,
) -> None:
    root = tmp_path / "raw-keyboard-interrupt"
    previous_handler = signal.getsignal(signal.SIGINT)

    def raw_interrupt(phase: str) -> None:
        if phase == "env_step":
            raise KeyboardInterrupt

    summary = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-raw-keyboard-interrupt"),
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=raw_interrupt,
    )

    assert summary["stop_reason"] == "contract_failed"
    assert any("unsafe KeyboardInterrupt" in error for error in summary["contract_errors"])
    assert verify_discovery_artifact(root)["valid"] is True
    assert signal.getsignal(signal.SIGINT) is previous_handler


def test_slow_checkpoint_and_finalization_fail_closed_without_deadline_clipping(
    tmp_path: Path,
) -> None:
    checkpoint_clock = AdvancingClock(step=0.0)
    checkpoint_root = tmp_path / "slow-checkpoint"
    checkpoint_summary = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-slow-checkpoint"),
        artifact_directory=checkpoint_root,
        clock=checkpoint_clock,
        process_clock=lambda: 0.0,
        phase_hook=SlowPhase(checkpoint_clock, "checkpoint", 901.0),
    )
    assert checkpoint_summary["stop_reason"] == "contract_failed"
    assert checkpoint_summary["consumed_wall_seconds"] >= 901.0
    assert verify_discovery_artifact(checkpoint_root)["valid"] is True

    finalization_clock = AdvancingClock(step=0.0)
    finalization_root = tmp_path / "slow-finalization"
    finalization_summary = run_discovery_pilot(
        _tiny_config(experiment_id="tiny-slow-finalization"),
        artifact_directory=finalization_root,
        clock=finalization_clock,
        process_clock=lambda: 0.0,
        phase_hook=SlowPhase(finalization_clock, "finalization", 901.0),
    )
    assert finalization_summary["stop_reason"] == "contract_failed"
    assert finalization_summary["consumed_wall_seconds"] >= 901.0
    assert finalization_summary["measured_finalization_wall_seconds"] >= 901.0
    assert verify_discovery_artifact(finalization_root)["valid"] is True


def test_mid_episode_deadline_partial_has_durable_metrics_and_resume_state(
    tmp_path: Path,
) -> None:
    root = tmp_path / "mid-episode-deadline"
    clock = AdvancingClock(step=0.0)
    config = _tiny_config(experiment_id="tiny-mid-episode-deadline")
    config["max_steps_per_episode"] = 10
    summary = run_discovery_pilot(
        config,
        artifact_directory=root,
        clock=clock,
        process_clock=lambda: 0.0,
        phase_hook=SlowPhase(clock, "env_step", 891.0),
    )

    assert summary["stop_reason"] == "budget_exhausted"
    assert verify_discovery_artifact(root)["valid"] is True
    pointer = _read_json(root / "runs/td0_zero/train-a-v1/resume-checkpoint.json")
    assert pointer["resume_in_episode"] is True
    assert pointer["resume_episode"]["env_steps_before"] == 0
    assert pointer["metrics"]["counters"]["env_steps"] == 1
    assert (
        pointer["metrics"]["counters"]["action_value_calls"]
        == pointer["counters"]["action_value_calls"]
    )
    assert pointer["metrics"]["counters"]["updates"] == pointer["counters"]["updates"]

    resumed = run_discovery_pilot(
        config,
        resume_from=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
    )

    assert resumed["stop_reason"] == "budget_exhausted"
    assert resumed["contract_errors"] == []
    assert verify_discovery_artifact(root)["valid"] is True


def test_verify_rejects_checkpoint_paths_outside_artifact_root(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "copied"
    shutil.copytree(completed_artifact, copied)
    checkpoint_path = copied / "checkpoints.jsonl"
    checkpoints = _read_jsonl(checkpoint_path)
    checkpoints[0]["checkpoint_directory"] = "../../outside-checkpoint"
    checkpoint_path.write_text(
        "".join(canonical_json(record) + "\n" for record in checkpoints), encoding="utf-8"
    )

    report = verify_discovery_artifact(copied)
    assert report["valid"] is False
    assert any("artifact path escapes" in error for error in report["errors"])


def test_resume_preflight_rejects_checkpoint_escape_before_restore(tmp_path: Path) -> None:
    root = tmp_path / "resume-escape"
    config = _tiny_config(experiment_id="tiny-resume-escape")
    summary = run_discovery_pilot(
        config,
        artifact_directory=root,
        clock=lambda: 0.0,
        process_clock=lambda: 0.0,
        phase_hook=CooperativeSignal("env_step"),
    )
    assert summary["stop_reason"] == "interrupted"

    checkpoint_path = root / "checkpoints.jsonl"
    checkpoints = _read_jsonl(checkpoint_path)
    checkpoints[0]["checkpoint_directory"] = "../../outside-checkpoint"
    checkpoint_path.write_text(
        "".join(canonical_json(record) + "\n" for record in checkpoints), encoding="utf-8"
    )

    with pytest.raises(ArtifactError, match="preflight verification"):
        run_discovery_pilot(
            config,
            resume_from=root,
            clock=lambda: 0.0,
            process_clock=lambda: 0.0,
        )


def test_verify_rejects_checkpoint_symlink_even_when_target_is_inside_artifact(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "symlinked-checkpoint"
    shutil.copytree(completed_artifact, copied)
    checkpoint_path = copied / "checkpoints.jsonl"
    checkpoints = _read_jsonl(checkpoint_path)
    metadata_path = copied / checkpoints[0]["metadata_path"]
    symlink_path = metadata_path.with_name("symlinked-metadata.json")
    symlink_path.symlink_to(metadata_path.name)
    checkpoints[0]["metadata_path"] = str(symlink_path.relative_to(copied))
    checkpoint_path.write_text(
        "".join(canonical_json(record) + "\n" for record in checkpoints), encoding="utf-8"
    )

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("contains a symlink" in error for error in report["errors"])


def test_verify_rejects_symlinked_run_parent_even_when_file_is_inside_artifact(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "symlinked-run-parent"
    shutil.copytree(completed_artifact, copied)
    run_path = copied / "runs/td0_zero/train-a-v1"
    external_target = tmp_path / "external-run"
    shutil.copytree(run_path, external_target)
    preserved_path = copied / "runs/td0_zero/train-a-v1-real"
    run_path.rename(preserved_path)
    run_path.symlink_to(external_target, target_is_directory=True)

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("contains a symlink" in error for error in report["errors"])


def test_verify_rejects_non_discovery_knowledge_manifest(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "external-manifest"
    shutil.copytree(completed_artifact, copied)
    manifest_path = copied / "knowledge-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["experiment_kind"] = "external"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("experiment_kind=discovery" in error for error in report["errors"])


def test_verify_rejects_manifest_values_that_do_not_match_resolved_config(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "mismatched-manifest"
    shutil.copytree(completed_artifact, copied)
    manifest_path = copied / "knowledge-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["initialization"]["optimistic_total_value"] = 999.0
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("does not match resolved Discovery config" in error for error in report["errors"])


def test_verify_requires_complete_checkpoint_matrix_for_completed_artifact(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "missing-checkpoint"
    shutil.copytree(completed_artifact, copied)
    checkpoint_path = copied / "checkpoints.jsonl"
    checkpoints = _read_jsonl(checkpoint_path)
    checkpoints = [record for record in checkpoints if record["checkpoint_episode"] != 200]
    checkpoint_path.write_text(
        "".join(canonical_json(record) + "\n" for record in checkpoints), encoding="utf-8"
    )

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("checkpoint 200 was not saved" in error for error in report["errors"])


def test_verify_rejects_duplicate_checkpoint_matrix_key(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "duplicate-checkpoint"
    shutil.copytree(completed_artifact, copied)
    checkpoint_path = copied / "checkpoints.jsonl"
    checkpoints = _read_jsonl(checkpoint_path)
    first = checkpoints[0]
    duplicate_key = (first["arm_id"], first["training_seed"], first["checkpoint_episode"])
    checkpoints[1] = dict(first)
    checkpoint_path.write_text(
        "".join(canonical_json(record) + "\n" for record in checkpoints), encoding="utf-8"
    )

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("duplicate milestone checkpoint" in error for error in report["errors"])
    assert any("checkpoint matrix mismatch" in error for error in report["errors"])
    assert duplicate_key == ("td0_zero", "train-a-v1", 0)


def test_recompute_requires_common_evaluation_episode_ids(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "unpaired-evaluation"
    shutil.copytree(completed_artifact, copied)
    path = copied / "runs/td0_zero/train-a-v1/evaluation/0/episodes.jsonl"
    records = _read_jsonl(path)
    records[0]["evaluation_episode_id"] = 1
    path.write_text("".join(canonical_json(record) + "\n" for record in records), encoding="utf-8")

    summary = recompute_discovery_summary(copied)

    assert summary["minimum_comparable"] is False
    assert summary["gate"] == "performance-blocked"


def test_verify_rejects_evaluation_without_its_milestone_checkpoint(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "evaluation-without-checkpoint"
    shutil.copytree(completed_artifact, copied)
    path = copied / "runs/td0_zero/train-a-v1/evaluation/0/episodes.jsonl"
    records = _read_jsonl(path)
    records[0]["checkpoint_global_env_step"] = -1
    path.write_text("".join(canonical_json(record) + "\n" for record in records), encoding="utf-8")

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("checkpoint step mismatch" in error for error in report["errors"])


def test_verify_rejects_manifest_stop_reason_mismatch(
    completed_artifact: Path, tmp_path: Path
) -> None:
    copied = tmp_path / "manifest-stop-mismatch"
    shutil.copytree(completed_artifact, copied)
    manifest_path = copied / "run-manifest.json"
    manifest = _read_json(manifest_path)
    manifest["stop_reason"] = "budget_exhausted"
    manifest_path.write_text(canonical_json(manifest) + "\n", encoding="utf-8")

    report = verify_discovery_artifact(copied)

    assert report["valid"] is False
    assert any("stop reason does not match" in error for error in report["errors"])


@pytest.mark.parametrize(
    ("errors", "minimum", "signal", "expected"),
    [
        (["hash mismatch"], True, True, "contract-failed"),
        ([], False, True, "performance-blocked"),
        ([], True, True, "pipeline-valid-signal-visible"),
        ([], True, False, "pipeline-valid-inconclusive"),
    ],
)
def test_result_gate_precedence(
    errors: list[str], minimum: bool, signal: bool, expected: str
) -> None:
    assert set(DISCOVERY_GATES) == {
        "pipeline-valid-signal-visible",
        "pipeline-valid-inconclusive",
        "performance-blocked",
        "contract-failed",
    }
    assert (
        classify_discovery_result(
            contract_errors=errors,
            minimum_comparable=minimum,
            consistent_signal=signal,
        )
        == expected
    )


def test_next_step_decision_uses_profile_and_amdahl_gate() -> None:
    native = _derive_next_step_decision(
        "performance-blocked",
        {
            "run": {
                "training_metrics": {
                    "wall_seconds": {
                        "end_to_end": 100.0,
                        "action_selection": 50.0,
                        "learning": 25.0,
                        "rules": 5.0,
                        "checkpoint": 5.0,
                        "artifact_logging": 5.0,
                    }
                }
            }
        },
    )
    assert native["decision"] == "native-core-child"
    assert native["estimated_overall_gain"] >= 0.3
    assert native["rules_only_rewrite_recommended"] is False

    python_first = _derive_next_step_decision(
        "performance-blocked",
        {
            "run": {
                "training_metrics": {
                    "wall_seconds": {
                        "end_to_end": 100.0,
                        "action_selection": 20.0,
                        "learning": 10.0,
                        "rules": 10.0,
                        "checkpoint": 20.0,
                        "artifact_logging": 10.0,
                    }
                }
            }
        },
    )
    assert python_first["decision"] == "python-optimization"

    stopped = _derive_next_step_decision("contract-failed", {})
    assert stopped["decision"] == "stop-route"
    assert stopped["evidence_sufficient"] is False
