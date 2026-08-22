# Experiment and artifact contract

Every run stores a canonical resolved configuration, SHA-256 config hash, run
manifest, KnowledgeManifest, JSONL episodes/metrics, optional replay, and
checkpoint pairs under one ignored artifact directory. JSON records carry a
schema version. Checkpoint metadata v2 combines the complete learner config,
counters, table and checkpoint hashes, plus an `EngineSnapshot` containing the
board and complete RNG state/lineage. NumPy arrays are restored with
`allow_pickle=False`; all metadata, array, learner and environment validation
finishes before the live learner is mutated. Imported v2 metadata requires the
complete learner configuration and RNG lineage field sets with strict JSON
types; malformed but coercible values are rejected.

The reproducibility class is explicit: the reference single-thread path is
`deterministic`; execution order changes in external multi-thread baselines are
not silently claimed deterministic. Evaluation and training use separate
purpose-derived RNG streams.

Discovery pilot artifacts add a versioned two-arm protocol around this base
contract. `strategy2048 discovery pilot --config ...` owns one shared
900-second wall-clock budget across training, frozen evaluation, checkpointing,
logging, and summary finalization. Protocol v1 charges a versioned 10-second
finalization reserve inside that same budget. `--resume <artifact-dir>` is explicit: it consumes the saved
learner/environment/RNG/scheduler state and subtracts the recorded consumed
wall time, never resetting the budget; the complete read-only verifier must
pass before any existing checkpoint is restored. `strategy2048 discovery verify
<artifact-dir>` is a separately timed, read-only recomputation gate; it cannot
resume training or add evaluation samples. See
[`discovery-baseline.md`](discovery-baseline.md) for the diagnostic-only result
gates, fixed score/tile milestones, next-step decision, and Discovery firewall
boundary.

Algorithm calibration artifacts add a separate
`algorithm-calibration-v1` contract around the same official oracle, TD
learner, checkpoint, and frozen-evaluation primitives. The protocol has one
shared 600-second limit, fixed `0/300/1000/3000/10000` candidates, a
selection-only episode-40 screen, and an independent episode-200 audit gate.
The raw-derived reducer and verifier keep the two evaluation suites separate,
persist the stage decision and experiment lineage, and record deterministic
afterstate coverage without using it for promotion. See
[`algorithm-calibration.md`](algorithm-calibration.md).

The calibration resolver has a mandatory canonical round-trip contract:

```text
raw TOML -> resolved JSON -> resolve again == identical resolved JSON
```

The tuning-context fingerprint is computed after semantic normalization. In
particular, omitted default tuples are expanded and candidate order is
canonicalized before hashing. A fingerprint over the surface TOML would let
the run start but make its own resolved artifact fail during final reduction.
