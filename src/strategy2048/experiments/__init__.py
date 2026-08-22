"""Experiment contracts, artifact persistence, and evaluation runners."""

from typing import Any

from strategy2048.experiments.artifacts import (
    ArtifactError,
    ArtifactStore,
    KnowledgeBoundaryError,
    KnowledgeManifest,
    canonical_json,
    config_hash,
)

_DISCOVERY_EXPORTS = {
    "DISCOVERY_GATES",
    "DiscoveryConfigError",
    "DiscoveryPilotConfig",
    "classify_discovery_result",
    "load_discovery_config",
    "recompute_discovery_summary",
    "resolve_discovery_config",
    "run_discovery_pilot",
    "verify_discovery_artifact",
}

_CONFIRMATION_EXPORTS = {
    "CONFIRMATION_GATES",
    "ConfirmationConfig",
    "ConfirmationConfigError",
    "ConfirmationShardRequest",
    "compute_confirmation_gate",
    "load_confirmation_config",
    "reduce_confirmation_gate",
    "resolve_confirmation_config",
    "run_confirmation_campaign",
    "run_confirmation_formal_campaign",
    "run_confirmation_scaling_preflight",
    "run_confirmation_shard",
    "run_scaling_preflight",
    "verify_confirmation_shard",
}


def __getattr__(name: str) -> Any:
    """Load runner exports lazily so the TD learner can import artifacts safely."""

    if name in _DISCOVERY_EXPORTS:
        from strategy2048.experiments import discovery

        return getattr(discovery, name)
    if name in _CONFIRMATION_EXPORTS:
        from strategy2048.experiments import confirmation

        return getattr(confirmation, name)
    raise AttributeError(name)


__all__ = [
    "ArtifactError",
    "ArtifactStore",
    "KnowledgeBoundaryError",
    "KnowledgeManifest",
    "canonical_json",
    "classify_discovery_result",
    "config_hash",
    "DISCOVERY_GATES",
    "DiscoveryConfigError",
    "DiscoveryPilotConfig",
    "load_discovery_config",
    "recompute_discovery_summary",
    "resolve_discovery_config",
    "run_discovery_pilot",
    "verify_discovery_artifact",
    "CONFIRMATION_GATES",
    "ConfirmationConfig",
    "ConfirmationConfigError",
    "ConfirmationShardRequest",
    "compute_confirmation_gate",
    "load_confirmation_config",
    "reduce_confirmation_gate",
    "resolve_confirmation_config",
    "run_confirmation_campaign",
    "run_confirmation_formal_campaign",
    "run_confirmation_scaling_preflight",
    "run_confirmation_shard",
    "run_scaling_preflight",
    "verify_confirmation_shard",
]
