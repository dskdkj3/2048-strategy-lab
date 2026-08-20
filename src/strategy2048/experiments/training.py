"""Bounded training runner for the reference TD(0) learner."""

from __future__ import annotations

import time
from typing import Any

from strategy2048.agents.protocol import EvaluationMode
from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.artifacts import ArtifactStore
from strategy2048.experiments.evaluation import EpisodeSummary, aggregate_episodes
from strategy2048.experiments.metrics import Metrics
from strategy2048.learning.td import TD1PAgent


def train_td(
    agent: TD1PAgent,
    *,
    episodes: int,
    root_seed: int | str,
    artifact_store: ArtifactStore | None = None,
    checkpoint_every: int | None = None,
    max_steps: int | None = None,
    start_episode: int = 0,
) -> dict[str, Any]:
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    metrics = Metrics()
    summaries: list[EpisodeSummary] = []
    started_all = time.perf_counter()
    for episode_offset in range(episodes):
        episode_id = start_episode + episode_offset
        env = OracleEnv(
            root_seed=root_seed, environment_id=f"train-{episode_id}", max_steps=max_steps
        )
        observation = env.reset(episode_id=episode_id, purpose="train-env")
        started = time.perf_counter()
        while not observation.terminated and not observation.truncated:
            action = agent.act(observation, EvaluationMode.TRAIN)
            with metrics.timer("rules"):
                result = env.step(action)
            metrics.increment("env_steps")
            with metrics.timer("learning"):
                agent.observe(result, result.observation)
            metrics.increment(
                "updates", agent.counters.updates - metrics.counters.get("updates", 0)
            )
            observation = result.observation
        elapsed = time.perf_counter() - started
        summary = EpisodeSummary(
            episode_id=episode_id,
            score=observation.score,
            max_tile=max(0 if cell == 0 else 1 << cell for cell in observation.board),
            steps=observation.step_id,
            terminated=observation.terminated,
            truncated=observation.truncated,
            wall_seconds=elapsed,
            reason="truncated" if observation.truncated else "terminated",
        )
        summaries.append(summary)
        metrics.increment("games")
        if artifact_store is not None:
            artifact_store.append_jsonl("episodes.jsonl", summary.to_json())
            if checkpoint_every and (episode_id + 1) % checkpoint_every == 0:
                with metrics.timer("checkpoint"):
                    agent.save_checkpoint(
                        artifact_store.root / "checkpoints",
                        episode_id + 1,
                        config_hash=artifact_store.config_hash,
                        environment_snapshot=env.snapshot(),
                    )
    metrics.add_wall("end_to_end", time.perf_counter() - started_all)
    metrics_record = metrics.snapshot()
    if artifact_store is not None:
        artifact_store.append_jsonl("metrics.jsonl", metrics_record)
        artifact_store.finalize(
            stop_reason="completed",
            summary={
                "kind": "training",
                "episodes": aggregate_episodes(summaries),
                "metrics": metrics_record,
                "learner_state_hash": agent.learner.state_hash(),
            },
        )
    return {
        "schema_version": "training-summary-v1",
        "episodes": aggregate_episodes(summaries),
        "metrics": metrics_record,
        "learner_state_hash": agent.learner.state_hash(),
    }
