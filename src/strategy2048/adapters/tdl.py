"""Thin, fail-closed adapter for a user-provided TDL2048 binary.

Nothing in this module downloads, patches, vendors, or imports TDL source.
"""

from __future__ import annotations

import hashlib
import platform
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FIXED_TDL_COMMIT = "a99f620aec0d30a75943a4c9646743f1f53b0197"
PARSER_VERSION = "tdl-text-v1"


class TDLAdapterError(RuntimeError):
    """TDL provenance, invocation, or parser contract failed."""


@dataclass(frozen=True, slots=True)
class TDLWorkload:
    seed: int | str
    threads: int = 1
    network: str = "default"
    train: int = 0
    evaluation: int = 0
    search: str = "1p"

    def to_json(self) -> dict[str, Any]:
        if self.threads <= 0 or self.train < 0 or self.evaluation < 0:
            raise ValueError("invalid TDL workload bounds")
        if not self.train and not self.evaluation:
            raise ValueError("TDL workload must train or evaluate at least one episode")
        if not self.network:
            raise ValueError("TDL network name must be explicit")
        if self.search != "1p":
            raise ValueError("Scientific MVP supports only the native 1-ply TDL path")
        return {
            "seed": self.seed,
            "threads": self.threads,
            "network": self.network,
            "train": self.train,
            "eval": self.evaluation,
            "search": self.search,
        }


@dataclass(frozen=True, slots=True)
class TDLReport:
    schema_version: str
    rules_lineage: str
    reproducibility_class: str
    metric_semantics: str
    source: dict[str, Any]
    binary: dict[str, Any]
    toolchain: dict[str, Any]
    workload: dict[str, Any]
    result: dict[str, Any]
    parser_version: str = PARSER_VERSION

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "rules_lineage": self.rules_lineage,
            "reproducibility_class": self.reproducibility_class,
            "metric_semantics": self.metric_semantics,
            "source": self.source,
            "binary": self.binary,
            "toolchain": self.toolchain,
            "workload": self.workload,
            "result": self.result,
            "parser_version": self.parser_version,
        }


class TDLAdapter:
    def __init__(self, *, expected_commit: str = FIXED_TDL_COMMIT) -> None:
        self.expected_commit = expected_commit

    def _source_identity(self, source: Path) -> dict[str, Any]:
        if not source.is_dir():
            raise TDLAdapterError(f"TDL source directory does not exist: {source}")
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=source, check=True, capture_output=True, text=True
            ).stdout.strip()
            dirty_output = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=source,
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError) as exc:
            raise TDLAdapterError(f"cannot inspect TDL source: {source}") from exc
        if commit != self.expected_commit:
            raise TDLAdapterError(
                f"TDL commit mismatch: expected {self.expected_commit}, got {commit}"
            )
        if dirty_output.strip():
            raise TDLAdapterError("TDL source worktree is dirty")
        return {"path": str(source), "commit": commit, "dirty": False}

    def _binary_identity(self, binary: Path) -> dict[str, Any]:
        if not binary.is_file() or not binary.stat().st_mode & 0o111:
            raise TDLAdapterError(f"TDL binary is missing or not executable: {binary}")
        digest = hashlib.sha256(binary.read_bytes()).hexdigest()
        return {"path": str(binary.resolve()), "sha256": digest, "size": binary.stat().st_size}

    def verify(self, source: str | Path, binary: str | Path, workload: TDLWorkload) -> TDLReport:
        source_identity = self._source_identity(Path(source))
        binary_identity = self._binary_identity(Path(binary))
        return TDLReport(
            schema_version="external-baseline-v1",
            rules_lineage="tdl_native_rules",
            reproducibility_class="deterministic" if workload.threads == 1 else "nondeterministic",
            metric_semantics=(
                "tdl_training_and_eval_moves"
                if workload.train and workload.evaluation
                else "tdl_training_moves"
                if workload.train
                else "tdl_eval_moves"
            ),
            source=source_identity,
            binary=binary_identity,
            toolchain={"python": platform.python_version(), "platform": platform.platform()},
            workload=workload.to_json(),
            result={"verified_only": True},
        )

    def run(self, source: str | Path, binary: str | Path, workload: TDLWorkload) -> TDLReport:
        report = self.verify(source, binary, workload)
        command = [str(Path(binary).resolve()), "-n", workload.network]
        if workload.threads > 1:
            command.extend(("-p", str(workload.threads)))
        if workload.train:
            command.extend(("-t", f"{workload.threads}x{workload.train}"))
        if workload.evaluation:
            command.extend(("-e", f"1x{workload.evaluation}"))
        command.extend(("-s", str(workload.seed)))
        try:
            with tempfile.TemporaryDirectory(prefix="strategy2048-tdl-") as temporary:
                completed = subprocess.run(
                    command,
                    cwd=temporary,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            raise TDLAdapterError("TDL process invocation failed") from exc
        result = self._parse_output(completed.stdout)
        if not self.expected_commit.startswith(str(result["reported_revision"])):
            raise TDLAdapterError("TDL banner revision does not match the verified source commit")
        result["command"] = command
        return TDLReport(
            schema_version=report.schema_version,
            rules_lineage=report.rules_lineage,
            reproducibility_class=report.reproducibility_class,
            metric_semantics=report.metric_semantics,
            source=report.source,
            binary=report.binary,
            toolchain=report.toolchain,
            workload=report.workload,
            result=result,
        )

    @staticmethod
    def _parse_output(output: str) -> dict[str, Any]:
        revision = re.search(r"Develop Rev\.([0-9a-f]+) \(([^\n]+)\)", output)
        summary = re.search(r"summary\s+(\d+)ms\s+([A-Za-z0-9.+-]+)ops", output)
        total_matches = re.findall(
            r"total:\s+avg=(\d+)\s+max=(\d+)\s+tile=(\d+)\s+win=([0-9.]+)%",
            output,
        )
        if revision is None or summary is None or not total_matches:
            raise TDLAdapterError("TDL output schema/parser mismatch")
        average, maximum, tile, win_percent = total_matches[-1]
        reported_ops = summary.group(2)
        return {
            "reported_revision": revision.group(1),
            "reported_toolchain": revision.group(2),
            "summary_milliseconds": int(summary.group(1)),
            "reported_ops_per_second": None if reported_ops == "inf" else float(reported_ops),
            "reported_ops_text": reported_ops,
            "average_score": int(average),
            "maximum_score": int(maximum),
            "maximum_tile": int(tile),
            "win_rate": float(win_percent) / 100.0,
            "stdout_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
        }
