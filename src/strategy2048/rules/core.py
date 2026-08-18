"""A small, explicit implementation of the official 4x4 2048 rules.

Boards use row-major tile exponents: ``0`` is empty and ``n`` represents
``2**n``.  This representation is intentionally not a bitboard and has no
four-bit exponent limit, which keeps the oracle useful for differential tests.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import IntEnum

Board = tuple[int, ...]
BOARD_SIZE = 4
BOARD_CELLS = BOARD_SIZE * BOARD_SIZE


class Action(IntEnum):
    """Versioned action encoding used by the engine and replay schema."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3

    @property
    def name_lower(self) -> str:
        return self.name.lower()

    @classmethod
    def parse(cls, value: Action | int | str) -> Action:
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            try:
                return cls(value)
            except ValueError as exc:
                raise ValueError(f"unknown action integer: {value}") from exc
        normalized = value.strip().lower()
        aliases = {"u": cls.UP, "r": cls.RIGHT, "d": cls.DOWN, "l": cls.LEFT}
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls[normalized.upper()]
        except KeyError as exc:
            raise ValueError(f"unknown action name: {value!r}") from exc


ACTION_ORDER: tuple[Action, ...] = (Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT)


@dataclass(frozen=True, slots=True)
class ChanceEvent:
    """An injected spawn event, expressed as a rank among current empty cells."""

    empty_rank: int
    tile_exponent: int

    def validate(self) -> None:
        if self.empty_rank < 0:
            raise ValueError("empty_rank must be non-negative")
        if self.tile_exponent not in (1, 2):
            raise ValueError("tile_exponent must be 1 (tile 2) or 2 (tile 4)")

    def to_json(self) -> dict[str, int]:
        self.validate()
        return {"empty_rank": self.empty_rank, "tile_exponent": self.tile_exponent}

    @classmethod
    def from_json(cls, value: object) -> ChanceEvent:
        if not isinstance(value, dict):
            raise ValueError("chance event must be an object")
        try:
            event = cls(int(value["empty_rank"]), int(value["tile_exponent"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid chance event") from exc
        event.validate()
        return event


@dataclass(frozen=True, slots=True)
class MergeEvent:
    """A merge observed while processing one line."""

    output_index: int
    source_exponent: int
    result_exponent: int
    score_delta: int


@dataclass(frozen=True, slots=True)
class MoveResult:
    before: Board
    afterstate: Board
    action: Action
    changed: bool
    score_delta: int
    merges: tuple[MergeEvent, ...]


def validate_board(board: Sequence[int]) -> Board:
    if len(board) != BOARD_CELLS:
        raise ValueError(f"board must contain {BOARD_CELLS} cells, got {len(board)}")
    normalized = tuple(int(cell) for cell in board)
    if any(cell < 0 for cell in normalized):
        raise ValueError("board exponents must be non-negative")
    return normalized


def board_from_values(values: Iterable[int]) -> Board:
    """Convert tile values (0, 2, 4, ...) into exponent representation."""

    exponents: list[int] = []
    for value in values:
        value = int(value)
        if value == 0:
            exponents.append(0)
            continue
        if value < 0 or value & (value - 1):
            raise ValueError(f"tile value must be zero or a power of two: {value}")
        exponents.append(value.bit_length() - 1)
    return validate_board(exponents)


def board_to_values(board: Sequence[int]) -> tuple[int, ...]:
    return tuple(0 if exponent == 0 else 1 << exponent for exponent in validate_board(board))


def empty_cells(board: Sequence[int]) -> tuple[int, ...]:
    normalized = validate_board(board)
    return tuple(index for index, exponent in enumerate(normalized) if exponent == 0)


def _compress_merge_line(
    line: Sequence[int],
) -> tuple[tuple[int, ...], int, tuple[MergeEvent, ...]]:
    nonzero = [int(value) for value in line if value]
    output: list[int] = []
    merges: list[MergeEvent] = []
    score = 0
    cursor = 0
    while cursor < len(nonzero):
        value = nonzero[cursor]
        if cursor + 1 < len(nonzero) and nonzero[cursor + 1] == value:
            result = value + 1
            delta = 1 << result
            merges.append(MergeEvent(len(output), value, result, delta))
            output.append(result)
            score += delta
            cursor += 2
        else:
            output.append(value)
            cursor += 1
    output.extend([0] * (len(line) - len(output)))
    return tuple(output), score, tuple(merges)


def _line_indices(action: Action, line_number: int) -> tuple[int, ...]:
    if action is Action.LEFT:
        return tuple(line_number * BOARD_SIZE + offset for offset in range(BOARD_SIZE))
    if action is Action.RIGHT:
        return tuple(line_number * BOARD_SIZE + offset for offset in range(BOARD_SIZE - 1, -1, -1))
    if action is Action.UP:
        return tuple(line_number + BOARD_SIZE * offset for offset in range(BOARD_SIZE))
    return tuple(line_number + BOARD_SIZE * offset for offset in range(BOARD_SIZE - 1, -1, -1))


def move_without_spawn(board: Sequence[int], action: Action | int | str) -> MoveResult:
    """Apply one move without chance spawning.

    Every source tile is consumed at most once by the adjacent-equal merge
    loop.  The function is pure and is therefore safe to use as an afterstate
    feature extractor and as the official differential-test oracle.
    """

    before = validate_board(board)
    parsed_action = Action.parse(action)
    after = list(before)
    total_score = 0
    all_merges: list[MergeEvent] = []
    for line_number in range(BOARD_SIZE):
        indices = _line_indices(parsed_action, line_number)
        line = tuple(before[index] for index in indices)
        merged, score, merges = _compress_merge_line(line)
        total_score += score
        all_merges.extend(merges)
        for index, value in zip(indices, merged, strict=True):
            after[index] = value
    afterstate = tuple(after)
    return MoveResult(
        before=before,
        afterstate=afterstate,
        action=parsed_action,
        changed=afterstate != before,
        score_delta=total_score,
        merges=tuple(all_merges),
    )


def legal_actions(board: Sequence[int]) -> tuple[Action, ...]:
    normalized = validate_board(board)
    return tuple(
        action for action in ACTION_ORDER if move_without_spawn(normalized, action).changed
    )


def is_terminated(board: Sequence[int]) -> bool:
    return not legal_actions(board)


def max_tile_value(board: Sequence[int]) -> int:
    normalized = validate_board(board)
    maximum = max(normalized, default=0)
    return 0 if maximum == 0 else 1 << maximum
