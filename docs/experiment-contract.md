# Experiment and artifact contract

Every run stores a canonical resolved configuration, SHA-256 config hash, run
manifest, KnowledgeManifest, JSONL episodes/metrics, optional replay, and
checkpoint pairs under one ignored artifact directory. JSON records carry a
schema version. Learner checkpoints are compressed NumPy arrays plus JSON
metadata and are restored with `allow_pickle=False`.

The reproducibility class is explicit: the reference single-thread path is
`deterministic`; execution order changes in external multi-thread baselines are
not silently claimed deterministic. Evaluation and training use separate
purpose-derived RNG streams.
