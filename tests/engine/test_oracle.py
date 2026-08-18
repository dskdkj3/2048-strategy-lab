from __future__ import annotations

import pytest

from strategy2048.engine.oracle import OracleEnv
from strategy2048.rules.core import Action, ChanceEvent, board_from_values, board_to_values


def test_reset_accepts_two_independent_injected_spawns() -> None:
    env = OracleEnv(root_seed="reset-vector")

    observation = env.reset(
        episode_id=0,
        chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 2)),
    )

    assert board_to_values(observation.board)[:3] == (2, 4, 0)
    assert observation.score == 0
    assert not observation.terminated


def test_invalid_action_does_not_consume_rng_or_spawn() -> None:
    env = OracleEnv(root_seed="invalid-vector")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 2)))
    env.board = board_from_values([2, 4, 0, 0, *([0] * 12)])
    before = env.snapshot()

    result = env.step(Action.LEFT)

    assert not result.valid
    assert result.spawn is None
    assert result.board == before.board
    assert result.rng_counter_before == result.rng_counter_after


def test_invalid_injected_event_is_atomic() -> None:
    env = OracleEnv(root_seed="atomic-vector")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 2)))
    env.board = board_from_values([2, 4, 0, 0, *([0] * 12)])
    before = env.snapshot()

    with pytest.raises(ValueError, match="invalid actions"):
        env.step(Action.LEFT, ChanceEvent(0, 1))

    after = env.snapshot()
    assert after.board == before.board
    assert after.step_id == before.step_id
    assert after.rng.counter == before.rng.counter


def test_valid_move_scores_and_spawns_after_afterstate() -> None:
    env = OracleEnv(root_seed="score-vector")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 1)))
    env.board = board_from_values([2, 2, 0, 0, *([0] * 12)])

    result = env.step(Action.LEFT, ChanceEvent(0, 2))

    assert result.valid
    assert result.score_delta == 4
    assert result.total_score == 4
    assert board_to_values(result.afterstate)[:4] == (4, 0, 0, 0)
    assert board_to_values(result.board)[:4] == (4, 4, 0, 0)


def test_win_is_not_termination() -> None:
    env = OracleEnv(root_seed="win-vector")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 1)))
    env.board = board_from_values([1024, 1024, 0, 0, *([0] * 12)])

    result = env.step(Action.LEFT, ChanceEvent(0, 1))

    assert result.won
    assert not result.terminated
    assert max(board_to_values(result.board)) == 2048


def test_no_move_board_terminates_without_rng_progress() -> None:
    values = [2, 4, 2, 4, 4, 2, 4, 2, 2, 4, 2, 4, 4, 2, 4, 2]
    env = OracleEnv(root_seed="termination-vector")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 1)))
    env.board = board_from_values(values)
    before_counter = env.rng.counter

    result = env.step(Action.LEFT)

    assert not result.valid
    assert result.terminated
    assert result.rng_counter_after == before_counter
