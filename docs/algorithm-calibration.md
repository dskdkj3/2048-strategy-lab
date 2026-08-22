# Algorithm calibration v1

`algorithm-calibration-v1` answers one narrow question: under a short CPU
budget, is a small optimistic initialization (`optimistic_total_value`) more
useful than keeping the TD learner at zero initialization?

The whole matrix has one shared **600-second wall-clock limit**. The limit
includes training, frozen evaluation, checkpoints, exploration diagnostics,
logging, and final summary writes. New-run initialization and resume preflight/
restore are charged from the moment the runner is entered. It is not 600
seconds per candidate, and resume never starts a fresh 600-second allowance.

## Two-stage experiment

The screen stage compares five values: `0`, `300`, `1000`, `3000`, and
`10000`. Each value uses the same two training seeds and the same learner
configuration. Every run is evaluated at episodes `0` and `40` with 20 games
from the selection seed suite. Scheduling rotates through candidates in
10-episode chunks so that one candidate cannot consume the useful early part
of the budget by itself.

A non-zero value is removed only when it scores at least 10% below zero for
both training seeds. If every non-zero value is removed, the run stops early
and retains zero. Otherwise the most robust non-zero candidate is selected by
its worse seed, then by its paired mean. Exploration coverage is deliberately
excluded from this ranking.

The confirmation stage continues only zero and the selected non-zero
candidate from episode 40 to episode 200. It then evaluates 50 games per run
using a separate audit seed suite that was not used to choose the survivor.
The non-zero candidate is recommended only when:

- its audit paired mean is at least 5% above zero; and
- neither training-seed lineage is below zero.

Two training seeds are an engineering direction check, not statistical
significance.

## Commands

Run the fixed protocol:

```bash
uv run strategy2048 calibration run \
  --config configs/algorithm-calibration-v1.toml
```

The default artifact is `artifacts/algorithm-calibration-v1/`. An interrupted
or budget-exhausted run can be resumed explicitly in the same directory:

```bash
uv run strategy2048 calibration run \
  --config configs/algorithm-calibration-v1.toml \
  --resume artifacts/algorithm-calibration-v1
```

Resume restores the learner, environment RNG, counters, coverage state, and
completed evaluations. It subtracts recorded wall time from the original 600
seconds; it does not grant a fresh budget. A completed or contract-failed
artifact cannot be resumed.

Run the independent, read-only verifier with:

```bash
uv run strategy2048 calibration verify artifacts/algorithm-calibration-v1
```

The verifier revalidates the resolved config and seed separation, restores
every milestone checkpoint, checks table/state hashes and RNG lineage, checks
that frozen evaluation changed neither live nor cloned learner state, derives
the screen survivor from selection raw records, derives the final gate from
audit raw records, verifies coverage hashes, checks contiguous training episode,
environment-step, counter, and checkpoint relationships, and recomputes the
summary.

## Derived contract v2

The v1 run directory remains the immutable scientific source. A post-run Python
library API can read that source and create a separate sibling
`algorithm-calibration-contract-v2` bundle. It never writes a sidecar into the
source directory and refuses to overwrite an existing destination.

The derived contract records a content digest of every source file, the source
run commit, the clean reducer commit, the projection schema version, and a full
raw-derived projection. The projection includes per-seed absolute score
distributions, max-tile and tile-reach rates, episodes, environment steps,
updates, wall/CPU cost, paired comparisons, and learning-efficiency fields.
Episode-40 own-score gain uses selection checkpoint 0 and 40 from the same
suite. Episode-200 reporting uses paired audit advantage and the resource delta
from episode 40 to 200; it does not subtract across the selection and audit
suites.

For every run that reached confirmation, the strong verifier restores the
exact episode-40 checkpoint and deterministically replays episodes 40 through
199. It compares each persisted score, tile, step/counter transition, learner
hash, and RNG lineage, then compares the complete episode-200 checkpoint. This
can take minutes on a formal artifact. It is independent verification after the
run, not additional training, extra samples, or an extension of the 600-second
scientific budget.

Existing valid v1 artifacts remain valid without a derived bundle. A derived
bundle is stronger portable evidence, not a rewrite or retroactive upgrade of
the source. A `contract-failed` source cannot be turned into a valid bundle.

## Exploration diagnostic

Training samples the chosen afterstate at a fixed stride. A sampled 4×4 board
is packed as 16 four-bit exponents (`afterstate-u64-v1`). Exponents above 15
are rejected rather than truncated, so the encoding cannot silently collide.
Each run records:

- observed and sampled steps;
- distinct sampled afterstates and first-visit ratio;
- action distribution;
- coverage content hash;
- instrumentation wall time.

If instrumentation exceeds 2% of observed training wall time, the metric is
marked `too-expensive`. It still cannot promote a candidate whose official
audit score is worse.

## Result gates and limits

- `oi-candidate-recommended`: the holdout audit passes both promotion rules;
- `zero-retained`: every OI candidate is removed in screening, or the final OI
  candidate loses on both audit lineages;
- `inconclusive`: screening found a survivor but the complete audit is mixed,
  too small, or unfinished;
- `performance-blocked`: the 600-second budget did not produce the minimum
  complete screen comparison;
- `contract-failed`: config, seed, hash, stage, checkpoint, frozen-state,
  coverage, or summary validation failed.

The artifact records the incumbent, parent calibration ID, fixed candidate
generation rule, tuning-context fingerprint, promotion result, and a proposed
next candidate neighborhood. It never starts another run automatically. A
network, learner algorithm, training scale, or seed-suite revision creates a
new tuning context; the old winner may be a search center, but must not be
treated as permanently optimal.

This experiment does not run a `63L`, T-shape, or other named-pattern
detector. It can show a score/coverage direction, not prove that the learner
discovered a human strategy.
