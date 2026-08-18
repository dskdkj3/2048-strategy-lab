"""Shared agent protocol."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from strategy2048.engine.oracle import Observation, StepResult
from strategy2048.experiments.artifacts import KnowledgeManifest
from strategy2048.rules.core import Action


class EvaluationMode(StrEnum):
    TRAIN = "train"
    EVALUATE = "evaluate"


class Agent(Protocol):
    agent_type: str

    def act(
        self, observation: Observation, mode: EvaluationMode = EvaluationMode.EVALUATE
    ) -> Action: ...

    def observe(self, transition: StepResult, next_observation: Observation) -> None: ...

    def knowledge_manifest(self) -> KnowledgeManifest: ...
