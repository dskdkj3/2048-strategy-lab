# OI-1000 multi-seed confirmation

This protocol is the fresh-seed follow-up to
[`algorithm-calibration-v1`](algorithm-calibration.md). It compares only
`td0_zero` and `td0_oi_1000`; the old calibration matrix, source digest, and
CLI remain unchanged.

## Evidence contract

The resolved config declares eight fresh training seeds before any result is
produced. They are consumed in four fixed two-seed cohorts. The training roots,
the paired audit root, and the historical calibration roots are all required to
be disjoint. Historical seeds are persisted in `legacy_seed_denylist` and can
never count toward the fresh gate.

Each `(candidate, training_seed)` pair is a private shard. A shard owns its
learner, RNG lineage, checkpoints, training JSONL, and paired audit records.
Workers publish a completed shard by atomic directory rename. Only the root
coordinator writes campaign progress, cohort records, and source summary; the
reducer sorts by cohort, seed, and candidate rather than by worker completion
order.

The source layout is:

```text
campaign/
  campaign-manifest.json
  campaign-progress.jsonl
  source/
    resolved-config.json
    cohorts/0001.json
    shards/<training-seed>/<candidate>/
      shard-manifest.json
      resolved-config.json
      run/training-episodes.jsonl
      run/checkpoints/episode-40/
      run/checkpoints/episode-200/
      run/evaluation/episodes.jsonl
    source-summary.json
  derived-contract/       # separate post-run sibling
  independent-check/      # separate post-run sibling
```

The scientific digest excludes wall/process CPU, RSS, timestamps, PIDs,
temporary paths, and worker completion order. Those observations remain in a
validated runtime telemetry projection and must satisfy finite, non-negative,
resource-bound, and budget-accounting checks.

## Gate and stopping rules

For each completed fresh seed:

```text
score_gain = (oi_mean_score - zero_mean_score) / zero_mean_score
tile_reach_delta = oi_256_reach_rate - zero_256_reach_rate
```

After at least four seeds, the reducer emits `oi-baseline-confirmed` only when
the median score gain is at least `+15%`, at least `75%` of seeds are positive,
no seed has `score_gain <= -20%`, and the median 256-tile reach delta is
non-negative. It emits `oi-baseline-rejected` when the median score gain is at
most zero, at most `25%` of seeds are positive, or at least two seeds have a
severe regression. Otherwise it adds exactly one two-seed cohort. A mixed result
after eight seeds is `inconclusive`.

Means, bootstrap intervals, worst seed, tile distributions, environment steps,
and CPU/wall time are diagnostics only. This is an engineering robustness gate,
not a statistical-significance claim.

## Budget and authority boundary

The campaign wall budget is `1800 s`, including source generation, derived
strong replay, and the independent checker. Resume must reuse the same seed,
checkpoint, RNG lineage, counters, and consumed budget; it cannot reset the
clock or replace a failed seed. A timeout produces `performance-blocked`, and a
contract or hash/resource failure produces `contract-failed`.

The internal Python API `run_confirmation_formal_campaign(...)` owns that
single ledger across source generation, source verification, derived replay,
and the independent checker. Continuing an existing campaign requires the
explicit `resume=True` argument. An interrupted phase is conservatively charged
against its previously allocated wall-time, so restarting the Python process
cannot recreate a fresh `1800 s` budget.

The bounded scaling preflight has its own `240 s` cap and must be approved
separately from implementation and ordinary tests. The formal campaign also
requires a separate approval after code delivery and preflight. This module
does not add a public command, wrapper, PATH entry, deployment, activation, or
campus/GPU route.

Injected preflight fixture runners and formal phase runners must be importable,
`spawn`-pickleable callables. Every synchronous runner executes in a child
process; on deadline the coordinator performs terminate/kill/join cleanup before
returning. Tiny watchdog fixtures in the ordinary test suite only prove this
control path and are not the separately approved scaling preflight.

## Verification

`confirmation_contract.py` independently recomputes the source gate from raw
shards, replays every completed episode-40→200 lineage for the derived bundle,
and runs the same work again for the checker. Stored `verified` flags are not
trusted. The source and destination are siblings, symlinks are rejected, and an
existing destination is never overwritten.
