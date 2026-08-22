from __future__ import annotations

import json
from pathlib import Path

from strategy2048 import cli


def test_discovery_cli_routes_pilot_and_explicit_resume(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "pilot.toml"
    config_path.write_text("experiment_id = 'cli-test'\n", encoding="utf-8")
    calls: list[tuple[dict[str, object], str | None]] = []

    def fake_run(config: dict[str, object], *, resume_from: str | Path | None = None):
        calls.append((config, None if resume_from is None else str(resume_from)))
        return {"gate": "pipeline-valid-inconclusive", "stop_reason": "completed"}

    monkeypatch.setattr(cli, "run_discovery_pilot", fake_run)
    assert (
        cli.main(
            [
                "discovery",
                "pilot",
                "--config",
                str(config_path),
                "--resume",
                str(tmp_path / "artifact"),
            ]
        )
        == 0
    )
    assert calls == [({"experiment_id": "cli-test"}, str(tmp_path / "artifact"))]


def test_discovery_cli_resume_without_path_uses_config_output_root(
    monkeypatch, tmp_path: Path
) -> None:
    config_path = tmp_path / "pilot.toml"
    config_path.write_text(
        "experiment_id = 'cli-test'\noutput_root = 'saved-artifacts'\n", encoding="utf-8"
    )
    calls: list[str | None] = []

    def fake_run(config: dict[str, object], *, resume_from: str | Path | None = None):
        del config
        calls.append(None if resume_from is None else str(resume_from))
        return {"gate": "pipeline-valid-inconclusive", "stop_reason": "completed"}

    monkeypatch.setattr(cli, "run_discovery_pilot", fake_run)
    assert cli.main(["discovery", "pilot", "--config", str(config_path), "--resume"]) == 0
    assert calls == ["saved-artifacts/cli-test"]


def test_discovery_cli_verify_returns_nonzero_for_contract_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "verify_discovery_artifact",
        lambda path: {"valid": False, "gate": "contract-failed", "path": path},
    )

    assert cli.main(["discovery", "verify", "artifact-dir"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["gate"] == "contract-failed"


def test_calibration_cli_routes_run_and_explicit_resume(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "calibration.toml"
    config_path.write_text("experiment_id = 'cli-calibration'\n", encoding="utf-8")
    calls: list[tuple[dict[str, object], str | None]] = []

    def fake_run(config: dict[str, object], *, resume_from: str | Path | None = None):
        calls.append((config, None if resume_from is None else str(resume_from)))
        return {"gate": "inconclusive", "stop_reason": "completed"}

    monkeypatch.setattr(cli, "run_algorithm_calibration", fake_run)
    assert (
        cli.main(
            [
                "calibration",
                "run",
                "--config",
                str(config_path),
                "--resume",
                str(tmp_path / "artifact"),
            ]
        )
        == 0
    )
    assert calls == [({"experiment_id": "cli-calibration"}, str(tmp_path / "artifact"))]


def test_calibration_cli_verify_returns_nonzero_for_contract_failure(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        cli,
        "verify_calibration_artifact",
        lambda path: {"valid": False, "gate": "contract-failed", "path": path},
    )

    assert cli.main(["calibration", "verify", "artifact-dir"]) == 1
    output = json.loads(capsys.readouterr().out)
    assert output["gate"] == "contract-failed"
