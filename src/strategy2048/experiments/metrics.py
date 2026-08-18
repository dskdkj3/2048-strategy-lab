"""Separated counters for rules, learning, boundary, and durability costs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass(slots=True)
class Metrics:
    counters: dict[str, int] = field(default_factory=dict)
    wall_seconds: dict[str, float] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + amount

    def add_wall(self, name: str, amount: float) -> None:
        self.wall_seconds[name] = self.wall_seconds.get(name, 0.0) + amount

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_wall(name, time.perf_counter() - started)

    def snapshot(self) -> dict[str, object]:
        rules_seconds = self.wall_seconds.get("rules", 0.0)
        learning_seconds = self.wall_seconds.get("learning", 0.0)
        end_to_end_seconds = self.wall_seconds.get("end_to_end", 0.0)
        env_steps = self.counters.get("env_steps", 0)
        updates = self.counters.get("updates", 0)
        games = self.counters.get("games", 0)
        return {
            "schema_version": "metric-v1",
            "counters": dict(sorted(self.counters.items())),
            "wall_seconds": dict(sorted(self.wall_seconds.items())),
            "rates": {
                "env_steps_per_second": env_steps / rules_seconds if rules_seconds else 0.0,
                "updates_per_second": updates / learning_seconds if learning_seconds else 0.0,
                "games_per_hour": games * 3600.0 / end_to_end_seconds
                if end_to_end_seconds
                else 0.0,
            },
            "search": {"nodes": 0, "wall_seconds": 0.0, "implemented": False},
        }
