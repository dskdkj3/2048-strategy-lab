# Native-core gate (Scientific MVP)

The MVP intentionally retains the Python reference implementation. The
profiler separates rules, learning, boundary, durability, and end-to-end
timers, and writes a bounded profile artifact. A native Rust/C++/Numba core is
recommended only after repeated stable workloads show that a differential-test
covered rules or learner hot path dominates end-to-end time and that the
expected games/CPU-hour or wall-clock learning gain justifies FFI, build,
portability, RNG, replay, and maintenance costs.

## Current evidence

The bounded Scientific MVP smoke was run on the `desktop` host with the
Python reference batch and a fixed configuration/seed. It completed **274
environment steps** at approximately **10,704.69 `env_steps/s`** for this
three-episode workload. This is an observed smoke result, not a hard
performance threshold: it is too small to characterize long-run learning
throughput and is sensitive to host background load.

The fixed TDL external smoke is recorded separately. Its native text output
reported approximately **161,000 training/evaluation operations per second**
for the bounded run, with `metric_semantics=tdl_training_and_eval_moves` and
`rules_lineage=tdl_native_rules`. This is an external-baseline metric with a
different engine, RNG lineage, and workload; it must not be compared directly
with the Python `env_steps/s` number or presented as this project's throughput.

The current evidence does not establish that a replaceable Python rules or
learner hot path dominates end-to-end cost. The gate therefore remains
**continue-python-reference**: first improve algorithmic batching, learner
layout, and logging only when profiling identifies them as material costs,
then reconsider a native-core child using differential tests and replay
alignment.

Until that evidence exists, the decision is: **continue Python reference; first
optimize algorithmic batching, learner layout, or logging**. TDL native moves/s
are a separate external-baseline metric and must not be presented as this
project's environment throughput.
