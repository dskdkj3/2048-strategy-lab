"""Stable JSON/JSONL artifact contracts and the Discovery knowledge firewall."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ARTIFACT_SCHEMA_VERSION = "artifact-v1"


class ArtifactError(ValueError):
    """Malformed or incompatible artifact."""


class KnowledgeBoundaryError(ArtifactError):
    """A Discovery manifest contains a forbidden information source."""


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def config_hash(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(config).encode("utf-8")).hexdigest()


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _git_metadata(cwd: Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"commit": revision, "dirty": dirty}


@dataclass(slots=True)
class KnowledgeManifest:
    """Machine-readable declaration of all knowledge entering a run."""

    experiment_kind: str = "discovery"
    observation: dict[str, Any] = field(default_factory=lambda: {"source": "official_board"})
    reward: dict[str, Any] = field(default_factory=lambda: {"source": "official_score_delta"})
    features: dict[str, Any] = field(default_factory=lambda: {"source": "configured_tuple_set"})
    initialization: dict[str, Any] = field(default_factory=lambda: {"source": "zero"})
    curriculum: dict[str, Any] = field(default_factory=lambda: {"source": "none"})
    checkpoint: dict[str, Any] = field(default_factory=lambda: {"source": "learner_state_only"})
    demonstrations: dict[str, Any] = field(default_factory=lambda: {"source": "none"})
    search: dict[str, Any] = field(default_factory=lambda: {"source": "none"})
    tablebase: dict[str, Any] = field(default_factory=lambda: {"source": "none"})
    detectors: dict[str, Any] = field(default_factory=lambda: {"source": "none"})

    FORBIDDEN_FIELDS = (
        "external_checkpoint",
        "pretrained",
        "human",
        "heuristic",
        "tablebase",
        "demonstration",
        "curriculum",
        "pattern",
        "detector",
    )

    def to_json(self) -> dict[str, Any]:
        value = {
            "schema_version": "knowledge-manifest-v1",
            "experiment_kind": self.experiment_kind,
            "observation": self.observation,
            "reward": self.reward,
            "features": self.features,
            "initialization": self.initialization,
            "curriculum": self.curriculum,
            "checkpoint": self.checkpoint,
            "demonstrations": self.demonstrations,
            "search": self.search,
            "tablebase": self.tablebase,
            "detectors": self.detectors,
        }
        self.validate()
        return value

    def validate(self) -> None:
        if self.experiment_kind not in {"baseline", "discovery", "hybrid", "external"}:
            raise KnowledgeBoundaryError(f"unknown experiment kind: {self.experiment_kind}")
        if self.experiment_kind != "discovery":
            return
        fields = {
            "observation": self.observation,
            "reward": self.reward,
            "features": self.features,
            "initialization": self.initialization,
            "curriculum": self.curriculum,
            "checkpoint": self.checkpoint,
            "demonstrations": self.demonstrations,
            "search": self.search,
            "tablebase": self.tablebase,
            "detectors": self.detectors,
        }
        forbidden: list[str] = []
        for field_name, value in fields.items():
            text = canonical_json(value).lower()
            if any(token in text for token in self.FORBIDDEN_FIELDS):
                forbidden.append(field_name)
        if self.initialization.get("source") not in {"zero", "optimistic"}:
            forbidden.append("initialization")
        if forbidden:
            raise KnowledgeBoundaryError(
                "Discovery manifest contains forbidden sources: "
                + ", ".join(sorted(set(forbidden)))
            )

    @classmethod
    def from_json(cls, value: object) -> KnowledgeManifest:
        if not isinstance(value, dict):
            raise ArtifactError("knowledge manifest must be an object")
        if value.get("schema_version") != "knowledge-manifest-v1":
            raise ArtifactError("unsupported knowledge manifest schema")
        kwargs = {
            field_name: dict(value[field_name])
            for field_name in (
                "observation",
                "reward",
                "features",
                "initialization",
                "curriculum",
                "checkpoint",
                "demonstrations",
                "search",
                "tablebase",
                "detectors",
            )
            if field_name in value
        }
        manifest = cls(experiment_kind=str(value["experiment_kind"]), **kwargs)
        manifest.validate()
        return manifest


class ArtifactStore:
    """Create and append to one immutable experiment directory."""

    def __init__(
        self,
        root: str | Path,
        resolved_config: Mapping[str, Any],
        *,
        repo_root: str | Path | None = None,
    ):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise ArtifactError(f"artifact directory is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self.resolved_config = dict(resolved_config)
        self.config_hash = config_hash(self.resolved_config)
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.started_at = time.time()

    @property
    def experiment_id(self) -> str:
        return self.root.name

    def write_json(self, name: str, value: Mapping[str, Any]) -> Path:
        path = self.root / name
        _atomic_write(path, canonical_json(dict(value)) + "\n")
        return path

    def append_jsonl(self, name: str, record: Mapping[str, Any]) -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(dict(record)) + "\n")

    def initialize(
        self,
        *,
        knowledge_manifest: KnowledgeManifest,
        seed: int | str,
        budget: Mapping[str, Any] | None = None,
    ) -> None:
        knowledge = knowledge_manifest.to_json()
        self.write_json("resolved-config.json", self.resolved_config)
        self.write_json(
            "run-manifest.json",
            {
                "schema_version": "run-manifest-v1",
                "experiment_id": self.experiment_id,
                "config_hash": self.config_hash,
                "seed": seed,
                "source": _git_metadata(self.repo_root),
                "host": {
                    "platform": platform.platform(),
                    "python": platform.python_version(),
                    "processor": platform.processor(),
                },
                "budget": dict(budget or {}),
                "stop_reason": None,
                "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            },
        )
        self.write_json("knowledge-manifest.json", knowledge)

    def finalize(self, *, stop_reason: str, summary: Mapping[str, Any]) -> None:
        self.write_json("summary.json", {"schema_version": "summary-v1", **dict(summary)})
        manifest_path = self.root / "run-manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["stop_reason"] = stop_reason
        manifest["finished_at"] = time.time()
        self.write_json("run-manifest.json", manifest)

    def failure(self, error: BaseException) -> None:
        self.write_json(
            "failure.json",
            {
                "schema_version": "failure-v1",
                "error_type": type(error).__name__,
                "message": str(error),
            },
        )
