from __future__ import annotations

import pytest

from strategy2048.engine.oracle import OracleEnv
from strategy2048.replay.log import ReplayDivergence, ReplayLog, ReplayRecorder, verify_replay
from strategy2048.rules.core import Action, ChanceEvent


def _recorded_log() -> ReplayLog:
    env = OracleEnv(root_seed="different-root", environment_id="different-env")
    env.reset(episode_id=0, chance_events=(ChanceEvent(0, 1), ChanceEvent(0, 1)))
    recorder = ReplayRecorder(env)
    recorder.record(env.step(Action.LEFT, ChanceEvent(0, 1)))
    recorder.record(env.step(Action.DOWN, ChanceEvent(0, 2)))
    return recorder.log


def test_replay_round_trip_and_verification(tmp_path) -> None:
    path = tmp_path / "run.jsonl"
    log = _recorded_log()
    log.write(path)
    loaded = ReplayLog.read(path)
    env = OracleEnv(root_seed="different-root", environment_id="different-env")

    verify_replay(env, loaded)

    assert len(loaded.frames) == 2


def test_replay_round_trip_preserves_rng_generated_spawns(tmp_path) -> None:
    env = OracleEnv(root_seed="random-replay", environment_id="random-env")
    env.reset(episode_id=0)
    recorder = ReplayRecorder(env)
    recorder.record(env.step(Action.LEFT))
    recorder.record(env.step(Action.DOWN))

    path = tmp_path / "random-run.jsonl"
    recorder.log.write(path)
    loaded = ReplayLog.read(path)

    assert all(frame.chance_event is None for frame in loaded.frames)
    verify_replay(OracleEnv(root_seed="other-root", environment_id="other-env"), loaded)


def test_replay_reports_first_divergent_field() -> None:
    log = _recorded_log()
    log.frames[0].result["score_delta"] = 999
    env = OracleEnv(root_seed="replay-root", environment_id="replay")

    with pytest.raises(ReplayDivergence) as error:
        verify_replay(env, log)

    assert error.value.step_index == 0
    assert error.value.field == "score_delta"
