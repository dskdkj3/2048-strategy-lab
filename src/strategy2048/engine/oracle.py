"""Single-environment implementation backed by the official rules oracle."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from strategy2048.rng.stream import RNGSnapshot, ScientificRNG, rng_for
from strategy2048.rules.core import (
    Action,
    Board,
    ChanceEvent,
    empty_cells,
    is_terminated,
    legal_actions,
    max_tile_value,
    move_without_spawn,
    validate_board,
)


@dataclass(frozen=True, slots=True)
class Observation:
    board: Board
    score: int
    legal_actions: tuple[Action, ...]
    won: bool
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int

    def to_json(self) -> dict[str, Any]:
        return {
            "board": list(self.board),
            "score": self.score,
            "legal_actions": [action.name_lower for action in self.legal_actions],
            "won": self.won,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    before: Board
    afterstate: Board
    board: Board
    action: Action
    valid: bool
    score_delta: int
    total_score: int
    spawn: ChanceEvent | None
    won: bool
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int
    rng_counter_before: int
    rng_counter_after: int

    @property
    def observation(self) -> Observation:
        return Observation(
            board=self.board,
            score=self.total_score,
            legal_actions=legal_actions(self.board),
            won=self.won,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id=self.episode_id,
            step_id=self.step_id,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "before": list(self.before),
            "afterstate": list(self.afterstate),
            "board": list(self.board),
            "action": self.action.name_lower,
            "valid": self.valid,
            "score_delta": self.score_delta,
            "total_score": self.total_score,
            "spawn": None if self.spawn is None else self.spawn.to_json(),
            "won": self.won,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "rng_counter_before": self.rng_counter_before,
            "rng_counter_after": self.rng_counter_after,
        }


@dataclass(frozen=True, slots=True)
class EngineSnapshot:
    schema_version: str
    board: Board
    score: int
    won: bool
    terminated: bool
    truncated: bool
    episode_id: int
    step_id: int
    rng: RNGSnapshot

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "board": list(self.board),
            "score": self.score,
            "won": self.won,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "episode_id": self.episode_id,
            "step_id": self.step_id,
            "rng": self.rng.to_json(),
        }

    @classmethod
    def from_json(cls, value: object) -> EngineSnapshot:
        if not isinstance(value, dict):
            raise ValueError("engine snapshot must be an object")
        required = {
            "schema_version",
            "board",
            "score",
            "won",
            "terminated",
            "truncated",
            "episode_id",
            "step_id",
            "rng",
        }
        if set(value) != required:
            raise ValueError("engine snapshot must contain the complete field set")
        if not isinstance(value["schema_version"], str):
            raise ValueError("engine snapshot schema_version must be a string")
        for field_name in ("won", "terminated", "truncated"):
            if type(value[field_name]) is not bool:
                raise ValueError(f"engine snapshot {field_name} must be a boolean")
        for field_name in ("score", "episode_id", "step_id"):
            if type(value[field_name]) is not int:
                raise ValueError(f"engine snapshot {field_name} must be an integer")
        board = value["board"]
        if not isinstance(board, (list, tuple)):
            raise ValueError("engine snapshot board must be an array")
        if any(type(cell) is not int for cell in board):
            raise ValueError("engine snapshot board cells must be integers")
        score = value["score"]
        episode_id = value["episode_id"]
        step_id = value["step_id"]
        if score < 0 or episode_id < -1 or step_id < 0:
            raise ValueError("engine snapshot counters are out of range")
        return cls(
            schema_version=value["schema_version"],
            board=validate_board(board),
            score=score,
            won=value["won"],
            terminated=value["terminated"],
            truncated=value["truncated"],
            episode_id=episode_id,
            step_id=step_id,
            rng=RNGSnapshot.from_json(value["rng"]),
        )


class OracleEnv:
    """A deterministic, headless 4x4 environment.

    ``reset`` performs two independent official spawns. ``step`` increments
    the logical step counter for every attempted action, but invalid actions
    do not consume the RNG and never spawn a tile.
    """

    SNAPSHOT_SCHEMA_VERSION = "engine-snapshot-v1"

    def __init__(
        self,
        *,
        root_seed: int | str = 0,
        environment_id: str = "env-0",
        win_tile: int = 2048,
        max_steps: int | None = None,
    ) -> None:
        if win_tile <= 0 or win_tile & (win_tile - 1):
            raise ValueError("win_tile must be a positive power of two")
        self.root_seed = root_seed
        self.environment_id = environment_id
        self.win_exponent = win_tile.bit_length() - 1
        self.max_steps = max_steps
        self.episode_id = -1
        self.step_id = 0
        self.board: Board = (0,) * 16
        self.score = 0
        self.won = False
        self.terminated = False
        self.truncated = False
        self.rng = rng_for(root_seed, "train-env", environment_id, 0)

    def _make_rng(self, episode_id: int, purpose: str = "train-env") -> ScientificRNG:
        return rng_for(self.root_seed, purpose, self.environment_id, episode_id)

    def reset(
        self,
        *,
        episode_id: int | None = None,
        purpose: str = "train-env",
        chance_events: Iterable[ChanceEvent | None] | None = None,
    ) -> Observation:
        if episode_id is None:
            episode_id = self.episode_id + 1
        if episode_id < 0:
            raise ValueError("episode_id must be non-negative")
        events = None if chance_events is None else tuple(chance_events)
        if events is not None and len(events) != 2:
            raise ValueError("reset requires exactly two chance events")
        candidate_rng = self._make_rng(episode_id, purpose)
        candidate_board: Board = (0,) * 16
        for event in events if events is not None else (None, None):
            candidate_board, _ = self._spawn_on_board(candidate_board, event, candidate_rng)
        candidate_won = max_tile_value(candidate_board) >= (1 << self.win_exponent)
        candidate_terminated = is_terminated(candidate_board)

        # Commit only after both chance events and all derived state have been
        # validated.  A malformed reset request therefore cannot partially
        # replace an already-running episode or advance its RNG.
        self.episode_id = episode_id
        self.step_id = 0
        self.board = candidate_board
        self.score = 0
        self.won = candidate_won
        self.truncated = False
        self.terminated = candidate_terminated
        self.rng = candidate_rng
        return self.observation()

    def _spawn_on_board(
        self,
        board: Board,
        event: ChanceEvent | None,
        rng: ScientificRNG | None = None,
    ) -> tuple[Board, ChanceEvent]:
        cells = empty_cells(board)
        if not cells:
            raise ValueError("cannot spawn on a full board")
        sampler = self.rng if rng is None else rng
        if event is None:
            event = ChanceEvent(
                empty_rank=sampler.randbelow(len(cells)),
                tile_exponent=2 if sampler.randbelow(10) == 9 else 1,
            )
        if not isinstance(event, ChanceEvent):
            raise ValueError("chance event must be a ChanceEvent")
        event.validate()
        if event.empty_rank >= len(cells):
            raise ValueError(
                f"chance empty_rank {event.empty_rank} is invalid for {len(cells)} empty cells"
            )
        index = cells[event.empty_rank]
        if board[index] != 0:
            raise ValueError("chance event selected a non-empty cell")
        mutable = list(board)
        mutable[index] = event.tile_exponent
        return tuple(mutable), event

    def _spawn(self, event: ChanceEvent | None) -> ChanceEvent:
        self.board, resolved = self._spawn_on_board(self.board, event)
        return resolved

    def observation(self) -> Observation:
        return Observation(
            board=self.board,
            score=self.score,
            legal_actions=legal_actions(self.board),
            won=self.won,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id=self.episode_id,
            step_id=self.step_id,
        )

    def step(
        self, action: Action | int | str, chance_event: ChanceEvent | None = None
    ) -> StepResult:
        if self.terminated or self.truncated:
            raise RuntimeError("cannot step a finished environment; call reset")
        parsed_action = Action.parse(action)
        before = self.board
        rng_before = self.rng.counter
        move = move_without_spawn(before, parsed_action)
        spawn: ChanceEvent | None = None
        if move.changed:
            next_board = move.afterstate
            if empty_cells(next_board):
                next_board, spawn = self._spawn_on_board(next_board, chance_event)
            elif chance_event is not None:
                raise ValueError("chance event supplied when the afterstate has no empty cell")
            self.board = next_board
            self.score += move.score_delta
        elif chance_event is not None:
            raise ValueError("invalid actions must not receive a chance event")
        else:
            self.board = move.afterstate
        self.step_id += 1
        if max_tile_value(self.board) >= (1 << self.win_exponent):
            self.won = True
        self.terminated = is_terminated(self.board)
        self.truncated = self.max_steps is not None and self.step_id >= self.max_steps
        return StepResult(
            before=before,
            afterstate=move.afterstate,
            board=self.board,
            action=parsed_action,
            valid=move.changed,
            score_delta=move.score_delta if move.changed else 0,
            total_score=self.score,
            spawn=spawn,
            won=self.won,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id=self.episode_id,
            step_id=self.step_id,
            rng_counter_before=rng_before,
            rng_counter_after=self.rng.counter,
        )

    def legal_actions(self) -> tuple[Action, ...]:
        return legal_actions(self.board)

    def snapshot(self) -> EngineSnapshot:
        return EngineSnapshot(
            schema_version=self.SNAPSHOT_SCHEMA_VERSION,
            board=self.board,
            score=self.score,
            won=self.won,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_id=self.episode_id,
            step_id=self.step_id,
            rng=self.rng.snapshot(),
        )

    def restore(self, snapshot: EngineSnapshot | dict[str, Any]) -> None:
        if isinstance(snapshot, dict):
            snapshot = EngineSnapshot.from_json(snapshot)
        elif isinstance(snapshot, EngineSnapshot):
            snapshot = EngineSnapshot.from_json(snapshot.to_json())
        else:
            raise ValueError("engine snapshot must be an EngineSnapshot or object")
        if snapshot.schema_version != self.SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported engine snapshot: {snapshot.schema_version}")
        candidate_board = validate_board(snapshot.board)
        candidate_rng = ScientificRNG(
            snapshot.rng.seed,
            purpose="snapshot-restore",
            lineage=snapshot.rng.lineage,
        )
        candidate_rng.restore(snapshot.rng)
        if snapshot.score < 0 or snapshot.episode_id < -1 or snapshot.step_id < 0:
            raise ValueError("engine snapshot counters must be non-negative")
        if snapshot.terminated != is_terminated(candidate_board):
            raise ValueError("engine snapshot termination flag does not match the board")
        reached_win_tile = max_tile_value(candidate_board) >= (1 << self.win_exponent)
        if snapshot.won != reached_win_tile:
            raise ValueError("engine snapshot won flag does not match the board")

        self.board = candidate_board
        self.score = snapshot.score
        self.won = snapshot.won
        self.terminated = snapshot.terminated
        self.truncated = snapshot.truncated
        self.episode_id = snapshot.episode_id
        self.step_id = snapshot.step_id
        self.rng = candidate_rng
