"""Replay records are JSONL-friendly and preserve chance events explicitly."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strategy2048.engine.oracle import EngineSnapshot, OracleEnv, StepResult
from strategy2048.rules.core import Action, ChanceEvent

REPLAY_SCHEMA_VERSION = "replay-v1"


class ReplayDivergence(AssertionError):
    """Raised with the first mismatching replay field."""

    def __init__(self, step_index: int, field: str, expected: object, actual: object) -> None:
        self.step_index = step_index
        self.field = field
        self.expected = expected
        self.actual = actual
        super().__init__(
            f"replay divergence at step {step_index}, {field}: expected {expected!r}, got {actual!r}"
        )


@dataclass(frozen=True, slots=True)
class ReplayFrame:
    action: Action
    chance_event: ChanceEvent | None
    result: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "action": self.action.name_lower,
            "chance_event": None if self.chance_event is None else self.chance_event.to_json(),
            "result": self.result,
        }

    @classmethod
    def from_json(cls, value: object) -> ReplayFrame:
        if not isinstance(value, dict):
            raise ValueError("replay frame must be an object")
        event = value.get("chance_event")
        return cls(
            action=Action.parse(value["action"]),
            chance_event=None if event is None else ChanceEvent.from_json(event),
            result=dict(value["result"]),
        )


@dataclass(slots=True)
class ReplayLog:
    initial_snapshot: EngineSnapshot
    frames: list[ReplayFrame] = field(default_factory=list)

    def to_json_lines(self) -> list[str]:
        lines = [
            json.dumps(
                {
                    "schema_version": REPLAY_SCHEMA_VERSION,
                    "kind": "initial",
                    "snapshot": self.initial_snapshot.to_json(),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        ]
        lines.extend(
            json.dumps(
                {"schema_version": REPLAY_SCHEMA_VERSION, "kind": "step", **frame.to_json()},
                sort_keys=True,
                separators=(",", ":"),
            )
            for frame in self.frames
        )
        return lines

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("\n".join(self.to_json_lines()) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path: str | Path) -> ReplayLog:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        if not lines:
            raise ValueError("replay is empty")
        records = [json.loads(line) for line in lines]
        initial = records[0]
        if (
            initial.get("schema_version") != REPLAY_SCHEMA_VERSION
            or initial.get("kind") != "initial"
        ):
            raise ValueError("replay does not start with a replay-v1 initial record")
        log = cls(EngineSnapshot.from_json(initial["snapshot"]))
        for record in records[1:]:
            if (
                record.get("schema_version") != REPLAY_SCHEMA_VERSION
                or record.get("kind") != "step"
            ):
                raise ValueError("replay contains an unknown record")
            log.frames.append(ReplayFrame.from_json(record))
        return log


class ReplayRecorder:
    def __init__(self, env: OracleEnv) -> None:
        self.log = ReplayLog(env.snapshot())

    def record(self, result: StepResult) -> None:
        # A resolved spawn is not necessarily an injected replay event.  When
        # the environment sampled it from its RNG, replay must call ``step``
        # without an event so that the same RNG draws are consumed.  An
        # explicitly injected event leaves the RNG counter unchanged, which is
        # enough to preserve that distinction without expanding the frame
        # schema.
        replay_event = (
            result.spawn
            if result.spawn is not None and result.rng_counter_after == result.rng_counter_before
            else None
        )
        self.log.frames.append(ReplayFrame(result.action, replay_event, result.to_json()))


def _compare_result(step_index: int, expected: dict[str, Any], actual: StepResult) -> None:
    actual_json = actual.to_json()
    fields = (
        "before",
        "afterstate",
        "board",
        "action",
        "valid",
        "score_delta",
        "total_score",
        "spawn",
        "won",
        "terminated",
        "truncated",
        "episode_id",
        "step_id",
        "rng_counter_before",
        "rng_counter_after",
    )
    for field_name in fields:
        if expected.get(field_name) != actual_json.get(field_name):
            raise ReplayDivergence(
                step_index,
                field_name,
                expected.get(field_name),
                actual_json.get(field_name),
            )


def verify_replay(env: OracleEnv, log: ReplayLog) -> None:
    """Restore the initial state and fail at the first divergent frame."""

    env.restore(log.initial_snapshot)
    for index, frame in enumerate(log.frames):
        result = env.step(frame.action, frame.chance_event)
        _compare_result(index, frame.result, result)


def replay_from_actions(
    env: OracleEnv,
    actions: Iterable[Action | int | str],
    chance_events: Iterable[ChanceEvent | None] | None = None,
) -> ReplayLog:
    recorder = ReplayRecorder(env)
    events = iter(chance_events) if chance_events is not None else None
    for action in actions:
        event = next(events) if events is not None else None
        recorder.record(env.step(action, event))
    return recorder.log
