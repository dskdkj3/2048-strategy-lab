from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from strategy2048.experiments.artifacts import (
    ArtifactStore,
    KnowledgeBoundaryError,
    KnowledgeManifest,
)

REPOSITORY_ROOT = Path(__file__).parents[2]


def _schema(name: str) -> dict[str, object]:
    return json.loads((REPOSITORY_ROOT / "schemas" / name).read_text(encoding="utf-8"))


def test_discovery_manifest_allows_zero_and_optimistic_initialization() -> None:
    for source in ("zero", "optimistic"):
        manifest = KnowledgeManifest(initialization={"source": source, "value": 1.0})
        value = manifest.to_json()
        jsonschema.validate(value, _schema("knowledge-manifest.v1.schema.json"))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("features", {"source": "corner_heuristic"}),
        ("checkpoint", {"source": "external_checkpoint"}),
        ("curriculum", {"source": "human_pattern_curriculum"}),
        ("tablebase", {"source": "tablebase"}),
    ],
)
def test_discovery_manifest_rejects_forbidden_knowledge(field: str, value: dict[str, str]) -> None:
    manifest = KnowledgeManifest()
    setattr(manifest, field, value)

    with pytest.raises(KnowledgeBoundaryError):
        manifest.validate()


def test_artifact_store_writes_versioned_manifests(tmp_path) -> None:
    store = ArtifactStore(
        tmp_path / "experiment", {"algorithm": "random"}, repo_root=REPOSITORY_ROOT
    )
    store.initialize(
        knowledge_manifest=KnowledgeManifest(experiment_kind="baseline"),
        seed="artifact-seed",
        budget={"episodes": 2},
    )
    store.finalize(stop_reason="completed", summary={"episodes": 2})

    run_manifest = json.loads((store.root / "run-manifest.json").read_text(encoding="utf-8"))
    knowledge_manifest = json.loads(
        (store.root / "knowledge-manifest.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(run_manifest, _schema("run-manifest.v1.schema.json"))
    jsonschema.validate(knowledge_manifest, _schema("knowledge-manifest.v1.schema.json"))
    assert run_manifest["stop_reason"] == "completed"
    assert len(run_manifest["config_hash"]) == 64
