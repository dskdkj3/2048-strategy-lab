"""Separated counters for rules, learning, boundary, and durability costs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

PHASE_TIMERS: tuple[str, ...] = (
    "rules",
    "action_selection",
    "feature_value_lookup",
    "td_update",
    "learning",
    "evaluation",
    "checkpoint",
    "artifact_logging",
    "verification",
    "end_to_end",
)


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
        td_update_seconds = self.wall_seconds.get("td_update", 0.0) or learning_seconds
        feature_lookup_seconds = self.wall_seconds.get("feature_value_lookup", 0.0)
        action_selection_seconds = self.wall_seconds.get("action_selection", 0.0)
        end_to_end_seconds = self.wall_seconds.get("end_to_end", 0.0)
        env_steps = self.counters.get("env_steps", 0)
        updates = self.counters.get("updates", 0)
        games = self.counters.get("games", 0)
        action_value_calls = self.counters.get("action_value_calls", 0)
        tuple_lookups = self.counters.get("tuple_lookups", 0)
        wall_seconds = {name: max(0.0, self.wall_seconds.get(name, 0.0)) for name in PHASE_TIMERS}
        wall_seconds.update(
            {name: value for name, value in self.wall_seconds.items() if name not in wall_seconds}
        )
        return {
            "schema_version": "metric-v1",
            "counters": dict(sorted(self.counters.items())),
            "wall_seconds": dict(sorted(wall_seconds.items())),
            "rates": {
                # Kept for Scientific MVP compatibility.  New consumers must
                # use the explicitly named rate below; this denominator is
                # the rules timer only and is not end-to-end training speed.
                "env_steps_per_second": env_steps / rules_seconds if rules_seconds else 0.0,
                "rules_env_steps_per_second": env_steps / rules_seconds if rules_seconds else 0.0,
                "end_to_end_env_steps_per_second": env_steps / end_to_end_seconds
                if end_to_end_seconds
                else 0.0,
                "action_value_calls_per_second": action_value_calls / action_selection_seconds
                if action_selection_seconds
                else 0.0,
                "tuple_lookups_per_second": tuple_lookups / feature_lookup_seconds
                if feature_lookup_seconds
                else 0.0,
                "updates_per_second": updates / td_update_seconds if td_update_seconds else 0.0,
                "games_per_hour": games * 3600.0 / end_to_end_seconds
                if end_to_end_seconds
                else 0.0,
            },
            "rate_semantics": {
                "rules_env_steps_per_second": {
                    "numerator": "counters.env_steps",
                    "denominator": "wall_seconds.rules",
                    "scope": "rules phase",
                },
                "end_to_end_env_steps_per_second": {
                    "numerator": "counters.env_steps",
                    "denominator": "wall_seconds.end_to_end",
                    "scope": "complete run",
                },
                "action_value_calls_per_second": {
                    "numerator": "counters.action_value_calls",
                    "denominator": "wall_seconds.action_selection",
                    "scope": "action-selection phase",
                },
                "tuple_lookups_per_second": {
                    "numerator": "counters.tuple_lookups",
                    "denominator": "wall_seconds.feature_value_lookup",
                    "scope": "feature/value lookup phase",
                },
                "updates_per_second": {
                    "numerator": "counters.updates",
                    "denominator": "wall_seconds.td_update",
                    "fallback_denominator": "wall_seconds.learning",
                    "scope": "TD update phase",
                },
                "games_per_hour": {
                    "numerator": "counters.games * 3600",
                    "denominator": "wall_seconds.end_to_end",
                    "scope": "complete run",
                },
            },
            "compatibility": {
                "env_steps_per_second": {
                    "value": env_steps / rules_seconds if rules_seconds else 0.0,
                    "semantics": "rules_timer_denominator",
                    "deprecated": True,
                }
            },
            "search": {"nodes": 0, "wall_seconds": 0.0, "implemented": False},
        }
