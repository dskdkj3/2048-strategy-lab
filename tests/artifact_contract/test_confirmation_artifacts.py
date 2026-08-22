from __future__ import annotations

from pathlib import Path

import pytest

from strategy2048.experiments.artifacts import ArtifactError
from strategy2048.experiments.confirmation_contract import (
    artifact_tree_sha256,
    verify_confirmation_source,
)


def test_confirmation_tree_hash_rejects_symlink(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    source.joinpath("record.json").write_text("{}\n", encoding="utf-8")
    first = artifact_tree_sha256(source)
    source.joinpath("record.json").write_text('{"changed":true}\n', encoding="utf-8")
    assert artifact_tree_sha256(source) != first
    source.joinpath("link").symlink_to("record.json")
    with pytest.raises(ArtifactError, match="symlink"):
        artifact_tree_sha256(source)


def test_confirmation_source_verifier_fails_closed_on_missing_controls(tmp_path: Path) -> None:
    report = verify_confirmation_source(tmp_path / "missing")
    assert report["valid"] is False
    assert report["errors"]
