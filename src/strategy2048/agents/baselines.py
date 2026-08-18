"""Small baseline agents with no human-pattern features."""

from __future__ import annotations

from dataclasses import dataclass, field

from strategy2048.agents.protocol import EvaluationMode
from strategy2048.engine.oracle import Observation, StepResult
from strategy2048.experiments.artifacts import KnowledgeManifest
from strategy2048.rng.stream import ScientificRNG, rng_for
from strategy2048.rules.core import Action, move_without_spawn


@dataclass(slots=True)
class RandomAgent:
    root_seed: int | str = 0
    agent_id: str = "random"
    agent_type: str = "baseline"
    rng: ScientificRNG = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = rng_for(self.root_seed, "agent-init", self.agent_id, 0)

    def act(
        self, observation: Observation, mode: EvaluationMode = EvaluationMode.EVALUATE
    ) -> Action:
        del mode
        if not observation.legal_actions:
            raise RuntimeError("random agent received an observation with no legal action")
        return observation.legal_actions[self.rng.randbelow(len(observation.legal_actions))]

    def observe(self, transition: StepResult, next_observation: Observation) -> None:
        del transition, next_observation

    def knowledge_manifest(self) -> KnowledgeManifest:
        return KnowledgeManifest(
            experiment_kind="baseline",
            observation={"source": "official_board", "fields": ["board", "legal_actions"]},
            reward={"source": "none_for_random_policy"},
            features={"source": "none"},
            initialization={"source": "random_action_stream"},
        )


@dataclass(slots=True)
class ScoreGreedyAgent:
    tie_break: str = "action-order"
    agent_type: str = "baseline"

    def act(
        self, observation: Observation, mode: EvaluationMode = EvaluationMode.EVALUATE
    ) -> Action:
        del mode
        if not observation.legal_actions:
            raise RuntimeError("score-greedy agent received an observation with no legal action")
        scored = [
            (move_without_spawn(observation.board, action).score_delta, -int(action), action)
            for action in observation.legal_actions
        ]
        return max(scored, key=lambda item: (item[0], item[1]))[2]

    def observe(self, transition: StepResult, next_observation: Observation) -> None:
        del transition, next_observation

    def knowledge_manifest(self) -> KnowledgeManifest:
        return KnowledgeManifest(
            experiment_kind="baseline",
            observation={"source": "official_board", "fields": ["board", "legal_actions"]},
            reward={"source": "official_immediate_merge_score"},
            features={"source": "none"},
            initialization={"source": "none"},
            search={"source": "none"},
        )
