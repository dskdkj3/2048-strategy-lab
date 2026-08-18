from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from strategy2048.adapters.tdl import TDLAdapter, TDLAdapterError, TDLWorkload


def _git_repository(path: Path) -> str:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "test"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    (path / "README").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "add", "README"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def _fake_binary(path: Path, revision: str) -> None:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "print('TDL2048+ by Hung Guei')\n"
        f"print('Develop Rev.{revision[:7]} (GCC fixture C++202002 @ fixture)')\n"
        "print('summary 7ms 1234.50ops')\n"
        "print('total:  avg=5017 max=13064 tile=1024 win=5.00%')\n",
        encoding="utf-8",
    )
    path.chmod(path.stat().st_mode | 0o111)


def test_adapter_verifies_provenance_and_parses_native_text(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _git_repository(source)
    binary = tmp_path / "tdl-fixture"
    _fake_binary(binary, commit)
    adapter = TDLAdapter(expected_commit=commit)

    report = adapter.run(
        source,
        binary,
        TDLWorkload(seed="fixture", network="4x6patt", evaluation=1),
    )

    assert report.rules_lineage == "tdl_native_rules"
    assert report.reproducibility_class == "deterministic"
    assert report.parser_version == "tdl-text-v1"
    assert report.metric_semantics == "tdl_eval_moves"
    assert report.result["average_score"] == 5017
    assert report.result["maximum_tile"] == 1024
    assert report.result["win_rate"] == 0.05
    assert report.result["stdout_sha256"]


def test_adapter_passes_parallel_thread_count_to_tdl(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _git_repository(source)
    binary = tmp_path / "tdl-fixture"
    _fake_binary(binary, commit)

    report = TDLAdapter(expected_commit=commit).run(
        source,
        binary,
        TDLWorkload(seed="fixture", threads=4, network="4x6patt", train=1),
    )

    command = report.result["command"]
    assert "-p" in command
    assert command[command.index("-p") + 1] == "4"


def test_adapter_fails_closed_on_dirty_source(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _git_repository(source)
    binary = tmp_path / "tdl-fixture"
    _fake_binary(binary, commit)
    (source / "dirty").write_text("untracked\n", encoding="utf-8")

    with pytest.raises(TDLAdapterError, match="dirty"):
        TDLAdapter(expected_commit=commit).verify(
            source,
            binary,
            TDLWorkload(seed=1, network="4x6patt", evaluation=1),
        )


def test_parser_accepts_zero_millisecond_infinite_ops_report() -> None:
    result = TDLAdapter._parse_output(
        "Develop Rev.a99f620 (GCC fixture)\n"
        "summary 0ms infops\n"
        "total:  avg=1572 max=1572 tile=128 win=0.00%\n"
    )

    assert result["reported_ops_per_second"] is None
    assert result["reported_ops_text"] == "inf"


def test_adapter_rejects_non_executable_binary(tmp_path) -> None:
    source = tmp_path / "source"
    commit = _git_repository(source)
    binary = tmp_path / "tdl-fixture"
    binary.write_text("fixture\n", encoding="utf-8")
    binary.chmod(os.stat(binary).st_mode & ~0o111)

    with pytest.raises(TDLAdapterError, match="not executable"):
        TDLAdapter(expected_commit=commit).verify(
            source,
            binary,
            TDLWorkload(seed=1, network="4x6patt", evaluation=1),
        )
