"""Deterministic evaluation and aggregate episode statistics."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from strategy2048.agents.protocol import Agent, EvaluationMode
from strategy2048.engine.oracle import Observation, OracleEnv, StepResult
from strategy2048.experiments.artifacts import ArtifactStore, KnowledgeManifest
from strategy2048.experiments.metrics import Metrics
from strategy2048.learning.td import LearningCounters, TDLearner, TimerFactory
from strategy2048.rules.core import Action


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    episode_id: int
    score: int
    max_tile: int
    steps: int
    terminated: bool
    truncated: bool
    wall_seconds: float
    reason: str

    def to_json(self) -> dict[str, object]:
        return {
            "schema_version": "episode-v1",
            "episode_id": self.episode_id,
            "official_score": self.score,
            "max_tile": self.max_tile,
            "steps": self.steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "wall_seconds": self.wall_seconds,
            "stop_reason": self.reason,
        }


@dataclass(slots=True)
class FrozenPolicyAgent:
    """Read-only policy view over a TD learner.

    Evaluation must not mutate the learner tables or the training counters.
    ``TDLearner`` therefore exposes a non-counting scorer and this wrapper
    makes ``observe`` an explicit no-op for the generic evaluation protocol.
    """

    learner: TDLearner
    agent_type: str = "discovery-frozen"

    def act(
        self, observation: Observation, mode: EvaluationMode = EvaluationMode.EVALUATE
    ) -> Action:
        del mode
        return self.learner.choose_action_read_only(observation)

    def act_read_only(
        self,
        observation: Observation,
        *,
        feature_timer: TimerFactory | None = None,
    ) -> Action:
        """Score one action choice without changing tables or learner counters."""

        return self.learner.choose_action_read_only(observation, feature_timer=feature_timer)

    def observe(self, transition: StepResult, next_observation: Observation) -> None:
        del transition, next_observation

    def knowledge_manifest(self) -> KnowledgeManifest:
        return self.learner.knowledge_manifest()

    @property
    def counters(self) -> LearningCounters:
        return self.learner.counters

    def state_hash(self) -> str:
        return self.learner.state_hash()

    def table_hash(self) -> str:
        return self.learner.table_hash()


class FrozenEvaluationError(RuntimeError):
    """Frozen evaluation changed state that was required to remain read-only."""

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result
        super().__init__("frozen evaluation mutated learner state or counters")


def _record_td_counter_delta(metrics: Metrics, before: dict[str, int], agent: object) -> None:
    counters = getattr(agent, "counters", None)
    if counters is None:
        return
    current = counters.to_json()
    for name in ("action_value_calls", "tuple_lookups", "updates", "tuple_updates"):
        value = current.get(name)
        if isinstance(value, int):
            metrics.increment(name, value - before.get(name, 0))


def _td_counter_snapshot(agent: object) -> dict[str, int]:
    counters = getattr(agent, "counters", None)
    if counters is None:
        return {}
    value = counters.to_json()
    return {
        name: int(raw)
        for name, raw in value.items()
        if name in {"action_value_calls", "tuple_lookups", "updates", "tuple_updates"}
        and isinstance(raw, int)
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def aggregate_episodes(episodes: list[EpisodeSummary]) -> dict[str, object]:
    scores = sorted(float(item.score) for item in episodes)
    tiles = [item.max_tile for item in episodes]
    return {
        "count": len(episodes),
        "score": {
            "mean": statistics.fmean(scores) if scores else 0.0,
            "median": statistics.median(scores) if scores else 0.0,
            "p25": _percentile(scores, 0.25),
            "p75": _percentile(scores, 0.75),
            "p95": _percentile(scores, 0.95),
        },
        "tile_reach_rate": {
            str(tile): sum(max_tile >= tile for max_tile in tiles) / len(tiles) if tiles else 0.0
            for tile in (128, 256, 512, 1024, 2048, 4096)
        },
        "max_tile_mean": statistics.fmean(tiles) if tiles else 0.0,
    }


def evaluate(
    agent_factory: Callable[[int], Agent],
    *,
    episodes: int,
    root_seed: int | str,
    purpose: str = "eval-env",
    max_steps: int | None = None,
    artifact_store: ArtifactStore | None = None,
) -> dict[str, object]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    metrics = Metrics()
    summaries: list[EpisodeSummary] = []
    started_all = time.perf_counter()
    with metrics.timer("evaluation"):
        for episode_id in range(episodes):
            agent = agent_factory(episode_id)
            env = OracleEnv(
                root_seed=root_seed,
                environment_id=f"eval-{episode_id}",
                max_steps=max_steps,
            )
            observation = env.reset(episode_id=episode_id, purpose=purpose)
            started = time.perf_counter()
            while not observation.terminated and not observation.truncated:
                counter_before = _td_counter_snapshot(agent)
                with metrics.timer("action_selection"):
                    action = agent.act(observation, EvaluationMode.EVALUATE)
                metrics.increment("action_decisions")
                with metrics.timer("rules"):
                    result = env.step(action)
                metrics.increment("env_steps")
                with metrics.timer("learning"):
                    observation = result.observation
                    agent.observe(result, observation)
                _record_td_counter_delta(metrics, counter_before, agent)
            elapsed = time.perf_counter() - started
            reason = "truncated" if observation.truncated else "terminated"
            summaries.append(
                EpisodeSummary(
                    episode_id=episode_id,
                    score=observation.score,
                    max_tile=max(0 if cell == 0 else 1 << cell for cell in observation.board),
                    steps=observation.step_id,
                    terminated=observation.terminated,
                    truncated=observation.truncated,
                    wall_seconds=elapsed,
                    reason=reason,
                )
            )
            metrics.increment("games")
    metrics.add_wall("end_to_end", time.perf_counter() - started_all)
    summary = {
        "schema_version": "evaluation-summary-v1",
        **aggregate_episodes(summaries),
        "metrics": metrics.snapshot(),
    }
    if artifact_store is not None:
        for item in summaries:
            artifact_store.append_jsonl("episodes.jsonl", item.to_json())
        artifact_store.append_jsonl("metrics.jsonl", metrics.snapshot())
        artifact_store.finalize(stop_reason="completed", summary=summary)
    return summary


def evaluate_frozen(
    agent: FrozenPolicyAgent,
    *,
    episodes: int,
    root_seed: int | str,
    purpose: str = "eval-env",
    environment_id: str = "frozen-eval",
    max_steps: int | None = None,
    episode_ids: tuple[int, ...] | None = None,
    clock: Callable[[], float] = time.monotonic,
    deadline: float | None = None,
) -> dict[str, Any]:
    """Evaluate a checkpoint clone without changing learner state.

    Evaluation counters are owned by the returned metrics object.  The
    learner's counters and table hash are captured before and after the run so
    callers can make the frozen-state contract explicit in their artifact.
    """

    if episodes <= 0:
        raise ValueError("episodes must be positive")
    ids = tuple(range(episodes)) if episode_ids is None else episode_ids
    if (
        len(ids) != episodes
        or len(set(ids)) != len(ids)
        or any(type(item) is not int or item < 0 for item in ids)
    ):
        raise ValueError("episode_ids must contain unique non-negative ids")
    before_hash = agent.state_hash()
    before_table_hash = agent.table_hash()
    before_counters = agent.counters.to_json()
    metrics = Metrics()
    summaries: list[EpisodeSummary] = []
    partial_episode: int | None = None
    stop_reason = "completed"
    started_all = time.perf_counter()
    with metrics.timer("evaluation"):
        for episode_id in ids:
            env = OracleEnv(
                root_seed=root_seed,
                environment_id=environment_id,
                max_steps=max_steps,
            )
            observation = env.reset(episode_id=episode_id, purpose=purpose)
            started = time.perf_counter()
            complete = True
            while not observation.terminated and not observation.truncated:
                if deadline is not None and clock() >= deadline:
                    complete = False
                    partial_episode = episode_id
                    stop_reason = "budget_exhausted"
                    break
                action_values = (
                    len(observation.legal_actions)
                    * agent.learner.value_function.active_feature_count
                )
                with metrics.timer("action_selection"):
                    action = agent.act_read_only(
                        observation,
                        feature_timer=lambda: metrics.timer("feature_value_lookup"),
                    )
                metrics.increment("action_decisions")
                metrics.increment("action_value_calls", len(observation.legal_actions))
                metrics.increment("tuple_lookups", action_values)
                with metrics.timer("rules"):
                    step_result = env.step(action)
                metrics.increment("env_steps")
                observation = step_result.observation
            if not complete:
                break
            elapsed = time.perf_counter() - started
            summaries.append(
                EpisodeSummary(
                    episode_id=episode_id,
                    score=observation.score,
                    max_tile=max(0 if cell == 0 else 1 << cell for cell in observation.board),
                    steps=observation.step_id,
                    terminated=observation.terminated,
                    truncated=observation.truncated,
                    wall_seconds=elapsed,
                    reason="truncated" if observation.truncated else "terminated",
                )
            )
            metrics.increment("games")
    metrics.add_wall("end_to_end", time.perf_counter() - started_all)
    after_hash = agent.state_hash()
    after_table_hash = agent.table_hash()
    after_counters = agent.counters.to_json()
    metrics_record = metrics.snapshot()
    result = {
        "schema_version": "frozen-evaluation-v1",
        "requested_episodes": episodes,
        "completed_episodes": len(summaries),
        **aggregate_episodes(summaries),
        "episodes": [item.to_json() for item in summaries],
        "metrics": metrics_record,
        "evaluation_counters": dict(metrics.counters),
        "state_hash_before": before_hash,
        "state_hash_after": after_hash,
        "table_hash_before": before_table_hash,
        "table_hash_after": after_table_hash,
        "counters_before": before_counters,
        "counters_after": after_counters,
        "state_unchanged": (
            before_hash == after_hash
            and before_table_hash == after_table_hash
            and before_counters == after_counters
        ),
        "stop_reason": stop_reason,
        "partial_episode": partial_episode,
    }
    if not result["state_unchanged"]:
        raise FrozenEvaluationError(result)
    return result
