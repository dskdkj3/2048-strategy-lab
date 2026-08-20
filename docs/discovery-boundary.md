# Discovery boundary

Discovery is a knowledge contract, not an experiment name. The manifest must
show that observation and reward come from the official environment, feature
tables are configured tuples, initialization is zero or explicitly optimistic,
and curriculum, demonstrations, tablebase, search heuristics, human-pattern
detectors, and external checkpoints are absent.

If a future experiment uses one of those sources it must be labeled `hybrid` or
another explicit non-discovery kind. Hiding a source in a renamed field is not
an acceptable bypass; `KnowledgeManifest.validate()` normalizes separator and
camel-case variants before rejecting the run. Imported manifests must also
contain exactly the complete v1 field set rather than relying on local defaults.
