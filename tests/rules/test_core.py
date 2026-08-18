from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from strategy2048.rules.core import (
    Action,
    board_from_values,
    board_to_values,
    legal_actions,
    move_without_spawn,
)


def _board_with_first_row(values: list[int]) -> tuple[int, ...]:
    return board_from_values([*values, *([0] * 12)])


@pytest.mark.parametrize(
    ("values", "expected", "score"),
    [
        ([2, 2, 2, 2], [4, 4, 0, 0], 8),
        ([2, 2, 4, 0], [4, 4, 0, 0], 4),
        ([4, 4, 8, 8], [8, 16, 0, 0], 24),
        ([2, 0, 2, 2], [4, 2, 0, 0], 4),
    ],
)
def test_left_merge_vectors(values: list[int], expected: list[int], score: int) -> None:
    result = move_without_spawn(_board_with_first_row(values), Action.LEFT)

    assert board_to_values(result.afterstate)[:4] == tuple(expected)
    assert result.score_delta == score
    assert result.changed


def test_directional_move_and_legal_actions() -> None:
    board = board_from_values([2, 0, 0, 0, 2, 0, 0, 0, *([0] * 8)])

    down = move_without_spawn(board, Action.DOWN)

    assert board_to_values(down.afterstate)[12] == 4
    assert down.score_delta == 4
    assert Action.DOWN in legal_actions(board)
    assert Action.LEFT not in legal_actions(board)


@given(st.lists(st.integers(min_value=0, max_value=10), min_size=16, max_size=16))
def test_move_preserves_tile_sum(exponents: list[int]) -> None:
    board = tuple(exponents)
    before_sum = sum(board_to_values(board))

    for action in Action:
        result = move_without_spawn(board, action)
        assert sum(board_to_values(result.afterstate)) == before_sum
