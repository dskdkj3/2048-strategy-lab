# Discovery baseline pilot

The Discovery pilot is a bounded diagnostic protocol for the reference
afterstate n-tuple TD(0) learner. It compares exactly two arms:

- `td0_zero`: zero-initialized feature tables;
- `td0_optimistic`: optimistic initialization using
  `optimistic_total_value`, divided across active tuple/symmetry features.

The shipped `discovery-pilot-v1` configuration fixes two training seeds, an
independent evaluation seed root, checkpoints at episodes `0`, `50`, and
`200`, and one shared **900-second wall-clock budget**. The 900 seconds cover
all four runs, training, frozen evaluation, checkpoint writes, logging, and
summary finalization. They are not a per-arm or per-seed allowance. Protocol
v1 reserves the final 10 seconds for durable checkpoints, stop records, and
summary finalization; that reserve is charged to the same 900-second budget
even when finalization completes faster.

## Commands

Run the approved protocol from a clean product checkout:

```bash
uv run strategy2048 discovery pilot --config configs/discovery-pilot-v1.toml
```

The output directory is `artifacts/discovery-pilot-v1/` (or the configured
`output_root`/`experiment_id` pair). It contains the resolved config, manifests,
raw JSONL records, milestone and partial checkpoints, a diagnostic summary, and
a verification report. Generated artifacts remain ignored by Git.

If an operator deliberately stops an incomplete run, resume it explicitly in
place:

```bash
uv run strategy2048 discovery pilot \
  --config configs/discovery-pilot-v1.toml \
  --resume artifacts/discovery-pilot-v1
```

Resume consumes the saved learner tables, environment snapshot, RNG state, and
scheduler cursor. It validates the resolved-config hash and checkpoint hashes
with the complete read-only artifact verifier before mutating a learner, and
computes the new deadline as:

```text
remaining = max(0, 900 - recorded_consumed_wall_seconds)
```

It cannot grant another 900 seconds. A completed or contract-failed artifact
is immutable and cannot be resumed. Resume is never automatic.

Post-hoc verification is read-only and separately timed:

```bash
uv run strategy2048 discovery verify artifacts/discovery-pilot-v1
```

Verification rechecks the config, KnowledgeManifest firewall, checkpoint pairs
and hashes, frozen-state invariants, pairing keys, scheduler prefix, and the
summary's recomputability from raw records. It does not resume training or add
evaluation episodes.

## What the pilot can say

The pilot is diagnostic only. It records training episodes, environment steps,
TD updates, wall-clock and CPU time, phase timers, official frozen-evaluation
score, max tile, and tile reach rates. OI comparisons use common completed
evaluation episode IDs for each training seed. A small pilot does not support
statistical significance, a claim that OI is generally stronger, or a claim
that the agent discovered a named human strategy.

Protocol v1 also fixes diagnostic milestones at official score `5000` and tile
`256`. For every arm/seed, the summary reports the first evaluated checkpoint
that reaches each target together with the training episode, cumulative env
steps, and training wall time. A partial run reports an unattained target as
`inconclusive`; only a completed matrix may report `not-attained`.

The summary emits one of four gates:

- `pipeline-valid-signal-visible`: all contracts pass and a direction is
  visible with the same sign across the two training seeds;
- `pipeline-valid-inconclusive`: contracts pass, but the direction is mixed or
  the sample is too small/noisy;
- `performance-blocked`: the shared budget did not produce the minimum paired
  checkpoint records;
- `contract-failed`: config, hash, pairing, checkpoint, resume, frozen-state,
  or aggregation validation failed.

These gates are routing decisions for the next experiment, not strength
claims. They may justify a larger approved baseline, a narrow Python
optimization, or a separate native-core investigation, but the pilot itself
does not implement native code.

The raw-derived `next_step_decision` makes that routing explicit. It reports
`continue-algorithm`, `python-optimization`, `native-core-child`, or
`stop-route`, together with the measured learner hot-path, rules, and
durability shares. A native-core recommendation is only possible when the
pilot is performance-blocked and Amdahl's law estimates that a conservative
2× learner-hot-path speedup would improve overall training throughput by at
least 30%. The report never recommends a rules-only rewrite.

## Discovery firewall and evaluation boundary

Training and evaluation use the official rules oracle and separate purpose-
derived RNG streams. Frozen evaluation restores a verified checkpoint into a
clone and scores actions through the non-counting read-only path; neither the
clone nor the live training learner may change tables or counters.

Discovery runs do not use human pattern/search detectors, handcrafted corner or
monotonic heuristics, tablebases, demonstrations, external checkpoints, or
contaminated curriculum. The pilot is CPU-only on the approved `desktop`
target; it does not use `campus2`, a GPU sweep, or a background run.

The OI setting `optimistic_total_value = 10000.0` is a short-pilot diagnostic
value. With the default eight tuples and D4 symmetry it yields 64 active
features and an initial per-feature value of `156.25`; it is not a paper-scale
reproduction or a generally optimal hyperparameter. The legacy
`optimistic_value` key is rejected rather than silently interpreted.
