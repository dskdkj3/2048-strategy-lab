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
