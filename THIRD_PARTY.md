# Third-party boundary

The Scientific MVP is an independent implementation. The official behavior
was audited against `gabrielecirulli/2048` commit
`478b6ec346e3787f589e4af751378d06ded4cbbc`; no upstream source is copied into
this repository.

`numpy` and `jsonschema` are runtime dependencies. Development tools are
`pytest`, `hypothesis`, `ruff`, and `mypy`; their licenses are tracked by the
package lock rather than vendored source.

TDL2048 is an explicitly external, user-supplied adapter target. The adapter
does not clone, download, patch, vendor, or link TDL source and reports its
native rules lineage separately. GPL sources, unknown-license tables, external
checkpoints, and demonstrations are research-only inputs and must not be
placed under this repository.
