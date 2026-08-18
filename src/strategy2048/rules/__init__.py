"""Official 2048 rules oracle."""

from strategy2048.rules.core import (
    ACTION_ORDER,
    Action,
    Board,
    ChanceEvent,
    MergeEvent,
    MoveResult,
    board_from_values,
    board_to_values,
    empty_cells,
    is_terminated,
    legal_actions,
    max_tile_value,
    move_without_spawn,
    validate_board,
)

__all__ = [
    "ACTION_ORDER",
    "Action",
    "Board",
    "ChanceEvent",
    "MergeEvent",
    "MoveResult",
    "board_from_values",
    "board_to_values",
    "empty_cells",
    "is_terminated",
    "legal_actions",
    "max_tile_value",
    "move_without_spawn",
    "validate_board",
]
