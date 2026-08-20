# 2048 Strategy Lab — Scientific MVP

This repository is a correctness-first, reproducible reference implementation
for studying 4×4 2048 agents. It deliberately starts with a slow official-rule
oracle, deterministic replay, small baselines, and an afterstate n-tuple TD(0)
learner. It is a scientific baseline, not a promise of a strong 2048 player.

## Quick start

With Nix:

```bash
nix develop
uv sync --frozen
uv run pytest
uv run strategy2048 doctor
uv run strategy2048 evaluate --config configs/random-smoke.toml
uv run strategy2048 train --config configs/td0-zero-smoke.toml
uv run strategy2048 discovery pilot --config configs/discovery-pilot-v1.toml
uv run strategy2048 discovery verify artifacts/discovery-pilot-v1
```

Without Nix, Python 3.13 and `uv` are required. `uv sync` creates the same
locked virtual environment. The CLI writes generated data only below the
configured `artifacts/` directory, which is ignored by Git.

## Scientific boundaries

The official oracle uses row-major tile exponents, the original 2/4 spawn
probabilities (0.9/0.1), valid-move-only spawning, and one merge per tile per
move. Every run records a resolved config hash, seed lineage, metrics, episode
summaries, and a KnowledgeManifest. Discovery runs reject human patterns,
handcrafted heuristics, tablebases, demonstrations, pretrained checkpoints,
and contaminated curriculum.

Checkpoint metadata v2 stores the complete learner state together with an
engine snapshot and RNG lineage. Restore validates the whole pair before
changing the live learner and returns the environment snapshot for explicit
resume.

The TDL adapter is an external provenance boundary. TDL results are labeled
`tdl_native_rules` and are never silently compared as oracle-compatible.

The bounded Discovery protocol is documented in
[`docs/discovery-baseline.md`](docs/discovery-baseline.md). It is a diagnostic
zero/OI pilot with one shared 900-second budget, frozen checkpoint evaluation,
an in-budget finalization reserve, fixed diagnostic score/tile milestones,
explicit read-only verification, and explicit resume only from a fully
validated partial artifact. Its summary derives a bounded next-step routing
decision from learning and phase-timer evidence. It does not claim statistical significance or strategy
discovery and does not use human pattern/search/tablebase knowledge, `campus2`,
or GPUs.

## License and contributions

The project is available under `MIT OR Apache-2.0`; see `NOTICE`,
`THIRD_PARTY.md`, and the two license files. Do not add GPL code, unknown
checkpoints, private data, or large generated profiles. Contributions should
include focused tests and preserve deterministic replay/checkpoint contracts.
