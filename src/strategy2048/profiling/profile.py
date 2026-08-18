"""cProfile wrapper plus separated internal counters."""

from __future__ import annotations

import cProfile
import io
import json
import pstats
import socket
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from strategy2048.agents.protocol import Agent, EvaluationMode
from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.metrics import Metrics


def benchmark(
    agent_factory: Callable[[int], Agent],
    *,
    episodes: int,
    root_seed: int | str,
    max_steps: int | None = None,
) -> dict[str, Any]:
    metrics = Metrics()
    started_all = time.perf_counter()
    for episode_id in range(episodes):
        env = OracleEnv(
            root_seed=root_seed, environment_id=f"benchmark-{episode_id}", max_steps=max_steps
        )
        observation = env.reset(episode_id=episode_id, purpose="eval-env")
        agent = agent_factory(episode_id)
        while not observation.terminated and not observation.truncated:
            action = agent.act(observation, EvaluationMode.EVALUATE)
            with metrics.timer("rules"):
                result = env.step(action)
            observation = result.observation
            metrics.increment("env_steps")
        metrics.increment("games")
    metrics.add_wall("end_to_end", time.perf_counter() - started_all)
    return {
        "schema_version": "profile-summary-v1",
        "host": socket.gethostname(),
        "episodes": episodes,
        "metrics": metrics.snapshot(),
    }


def profile_training(
    workload: Callable[[], dict[str, Any]],
    *,
    output_dir: str | Path,
    name: str = "training",
) -> dict[str, Any]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    profile_path = destination / f"{name}.prof"
    summary_path = destination / "summary.json"
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    try:
        workload_result = workload()
    finally:
        profiler.disable()
    elapsed = time.perf_counter() - started
    profiler.dump_stats(profile_path)
    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    result = {
        "schema_version": "profile-summary-v1",
        "name": name,
        "wall_seconds": elapsed,
        "workload": workload_result,
        "native_core_gate": {
            "recommendation": "continue-python-reference",
            "reason": "profile is evidence for a later decision; this MVP does not add a native core",
        },
        "top_functions": stream.getvalue(),
    }
    summary_path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result
