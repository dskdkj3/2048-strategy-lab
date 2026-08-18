"""Deterministic evaluation and aggregate episode statistics."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass

from strategy2048.agents.protocol import Agent, EvaluationMode
from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.artifacts import ArtifactStore
from strategy2048.experiments.metrics import Metrics


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
            action = agent.act(observation, EvaluationMode.EVALUATE)
            with metrics.timer("rules"):
                result = env.step(action)
            metrics.increment("env_steps")
            observation = result.observation
            agent.observe(result, observation)
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
