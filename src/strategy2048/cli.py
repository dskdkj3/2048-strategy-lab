"""Command-line entry points for bounded scientific smoke runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

from strategy2048.adapters.tdl import TDLAdapter, TDLWorkload
from strategy2048.agents.baselines import RandomAgent, ScoreGreedyAgent
from strategy2048.engine.oracle import OracleEnv
from strategy2048.experiments.artifacts import (
    ArtifactError,
    ArtifactStore,
    KnowledgeManifest,
)
from strategy2048.experiments.discovery import (
    run_discovery_pilot,
    verify_discovery_artifact,
)
from strategy2048.experiments.evaluation import evaluate
from strategy2048.experiments.training import train_td
from strategy2048.learning.td import TD1PAgent, TDLearner, TupleValueFunction
from strategy2048.profiling.profile import benchmark, profile_training
from strategy2048.replay.log import ReplayLog, verify_replay


def _load_toml(path: str | Path) -> dict[str, Any]:
    import tomllib

    with Path(path).open("rb") as handle:
        value = tomllib.load(handle)
    return value


def _artifact_store(config: dict[str, Any], command: str) -> ArtifactStore:
    output_root = Path(config.get("output_root", "artifacts"))
    experiment_id = str(config.get("experiment_id", f"{command}-{os.getpid()}"))
    resolved = {"command": command, **config}
    return ArtifactStore(output_root / experiment_id, resolved)


def _agent_factory(name: str, config: dict[str, Any]) -> Callable[[int], Any]:
    if name == "random":
        return lambda episode_id: RandomAgent(config.get("seed", 0), f"random-{episode_id}")
    if name == "score_greedy":
        return lambda episode_id: ScoreGreedyAgent()
    raise ValueError(f"unsupported evaluation agent: {name}")


def _knowledge_for_agent(name: str, config: dict[str, Any]) -> KnowledgeManifest:
    if name == "random":
        return RandomAgent(config.get("seed", 0)).knowledge_manifest()
    if name == "score_greedy":
        return ScoreGreedyAgent().knowledge_manifest()
    raise ValueError(f"unknown baseline agent: {name}")


def _td_agent(config: dict[str, Any]) -> TD1PAgent:
    initialization = str(config.get("initialization", "zero"))
    if initialization not in {"zero", "optimistic"}:
        raise ValueError("initialization must be zero or optimistic")
    if "optimistic_value" in config:
        raise ValueError(
            "optimistic_value is obsolete; use optimistic_total_value so the initial value "
            "is interpreted as the total active-feature value"
        )
    optimistic_total_value = (
        float(config.get("optimistic_total_value", 0.0)) if initialization == "optimistic" else 0.0
    )
    if optimistic_total_value < 0:
        raise ValueError("optimistic_total_value must be non-negative")
    tuple_config = config.get("tuples")
    tuples = (
        tuple(tuple(int(index) for index in item) for item in tuple_config)
        if tuple_config
        else None
    )
    value_function = TupleValueFunction(
        tuples=tuples or TupleValueFunction().tuples,
        value_cardinality=int(config.get("value_cardinality", 16)),
        symmetry=bool(config.get("symmetry", True)),
        optimistic_total_value=optimistic_total_value,
    )
    return TD1PAgent(
        TDLearner(
            value_function=value_function,
            alpha=float(config.get("alpha", 0.1)),
            gamma=float(config.get("gamma", 1.0)),
            optimistic_initialization=value_function.initial_value,
            optimistic_total_value=optimistic_total_value,
        )
    )


def command_evaluate(args: argparse.Namespace) -> int:
    config = _load_toml(args.config)
    name = str(config.get("agent", "random"))
    store = _artifact_store(config, "evaluate")
    store.initialize(
        knowledge_manifest=_knowledge_for_agent(name, config),
        seed=config.get("seed", 0),
        budget={"episodes": config.get("episodes", 1), "max_steps": config.get("max_steps")},
    )
    summary = evaluate(
        _agent_factory(name, config),
        episodes=int(config.get("episodes", 1)),
        root_seed=config.get("seed", 0),
        max_steps=config.get("max_steps"),
        artifact_store=store,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_train(args: argparse.Namespace) -> int:
    config = _load_toml(args.config)
    agent = _td_agent(config)
    store = _artifact_store(config, "train")
    store.initialize(
        knowledge_manifest=agent.knowledge_manifest(),
        seed=config.get("seed", 0),
        budget={"episodes": config.get("episodes", 1), "max_steps": config.get("max_steps")},
    )
    summary = train_td(
        agent,
        episodes=int(config.get("episodes", 1)),
        root_seed=config.get("seed", 0),
        artifact_store=store,
        checkpoint_every=config.get("checkpoint_every"),
        max_steps=config.get("max_steps"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_benchmark(args: argparse.Namespace) -> int:
    config = _load_toml(args.config)
    name = str(config.get("agent", "random"))
    summary = benchmark(
        _agent_factory(name, config),
        episodes=int(config.get("episodes", 1)),
        root_seed=config.get("seed", 0),
        max_steps=config.get("max_steps"),
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def command_profile(args: argparse.Namespace) -> int:
    config = _load_toml(args.config)
    name = str(config.get("agent", "random"))
    result = profile_training(
        lambda: benchmark(
            _agent_factory(name, config),
            episodes=int(config.get("episodes", 1)),
            root_seed=config.get("seed", 0),
            max_steps=config.get("max_steps"),
        ),
        output_dir=config.get("profile_dir", "profiles"),
        name=str(config.get("profile_name", "reference")),
    )
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "top_functions"},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def command_discovery_pilot(args: argparse.Namespace) -> int:
    config = _load_toml(args.config)
    resume_from = args.resume
    if resume_from == "":
        resume_from = Path(config.get("output_root", "artifacts")) / str(
            config.get("experiment_id", "discovery-pilot-v1")
        )
    summary = run_discovery_pilot(
        config,
        resume_from=resume_from,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary.get("gate") != "contract-failed" else 1


def command_discovery_verify(args: argparse.Namespace) -> int:
    report = verify_discovery_artifact(args.artifact_directory)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("valid") is True else 1


def command_doctor(_: argparse.Namespace) -> int:
    checks = {
        "python": sys.version_info >= (3, 13) and sys.version_info < (3, 14),
        "numpy": _module_available("numpy"),
        "jsonschema": _module_available("jsonschema"),
        "artifact_boundary": all(suffix not in str(Path.cwd()) for suffix in (".npz", ".prof")),
    }
    print(json.dumps({"schema_version": "doctor-v1", "checks": checks}, indent=2, sort_keys=True))
    return 0 if all(checks.values()) else 1


def _module_available(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def command_manifest_validate(args: argparse.Namespace) -> int:
    value = json.loads(Path(args.path).read_text(encoding="utf-8"))
    KnowledgeManifest.from_json(value)
    print(json.dumps({"valid": True, "schema_version": value["schema_version"]}, sort_keys=True))
    return 0


def command_checkpoint_verify(args: argparse.Namespace) -> int:
    config = _load_toml(args.config) if args.config else {}
    agent = _td_agent(config)
    environment = agent.restore_checkpoint(
        args.directory, int(args.step), config_hash=str(args.config_hash or "")
    )
    print(
        json.dumps(
            {
                "valid": True,
                "state_hash": agent.learner.state_hash(),
                "environment_schema": environment.schema_version,
                "rng_schema": environment.rng.schema_version,
            },
            sort_keys=True,
        )
    )
    return 0


def command_replay_verify(args: argparse.Namespace) -> int:
    log = ReplayLog.read(args.path)
    snapshot = log.initial_snapshot
    env = OracleEnv(root_seed=snapshot.rng.seed, environment_id="replay")
    verify_replay(env, log)
    print(json.dumps({"valid": True, "frames": len(log.frames)}, sort_keys=True))
    return 0


def command_tdl(args: argparse.Namespace, *, run: bool) -> int:
    config = _load_toml(args.config) if args.config else {}
    workload = TDLWorkload(
        seed=config.get("seed", args.seed or 0),
        threads=int(config.get("threads", args.threads or 1)),
        network=str(config.get("network", "default")),
        train=int(config.get("train", 0)),
        evaluation=int(config.get("eval", 0)),
        search=str(config.get("search", "1p")),
    )
    report = (TDLAdapter().run if run else TDLAdapter().verify)(args.source, args.binary, workload)
    print(json.dumps(report.to_json(), indent=2, sort_keys=True))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="strategy2048")
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor = subparsers.add_parser("doctor")
    doctor.set_defaults(handler=command_doctor)
    command_handlers: tuple[tuple[str, Callable[[argparse.Namespace], int]], ...] = (
        ("evaluate", command_evaluate),
        ("train", command_train),
        ("benchmark", command_benchmark),
        ("profile", command_profile),
    )
    for command_name, command_handler in command_handlers:
        sub = subparsers.add_parser(command_name)
        sub.add_argument("--config", required=True)
        sub.set_defaults(handler=command_handler)
    discovery = subparsers.add_parser("discovery")
    discovery_sub = discovery.add_subparsers(dest="discovery_command", required=True)
    discovery_pilot = discovery_sub.add_parser(
        "pilot",
        help="run the diagnostic zero/OI pilot under one shared 900-second budget",
    )
    discovery_pilot.add_argument("--config", required=True)
    discovery_pilot.add_argument(
        "--resume",
        nargs="?",
        const="",
        metavar="ARTIFACT_DIR",
        help="explicitly resume a prior artifact; omit the path to use the config output root",
    )
    discovery_pilot.set_defaults(handler=command_discovery_pilot)
    discovery_verify = discovery_sub.add_parser(
        "verify",
        help="read-only verification of a Discovery artifact",
    )
    discovery_verify.add_argument("artifact_directory")
    discovery_verify.set_defaults(handler=command_discovery_verify)
    manifest = subparsers.add_parser("manifest")
    manifest_sub = manifest.add_subparsers(dest="manifest_command", required=True)
    manifest_validate = manifest_sub.add_parser("validate")
    manifest_validate.add_argument("path")
    manifest_validate.set_defaults(handler=command_manifest_validate)
    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint_sub = checkpoint.add_subparsers(dest="checkpoint_command", required=True)
    checkpoint_verify = checkpoint_sub.add_parser("verify")
    checkpoint_verify.add_argument("directory")
    checkpoint_verify.add_argument("step")
    checkpoint_verify.add_argument("--config")
    checkpoint_verify.add_argument("--config-hash")
    checkpoint_verify.set_defaults(handler=command_checkpoint_verify)
    replay = subparsers.add_parser("replay")
    replay_sub = replay.add_subparsers(dest="replay_command", required=True)
    replay_verify = replay_sub.add_parser("verify")
    replay_verify.add_argument("path")
    replay_verify.set_defaults(handler=command_replay_verify)
    external = subparsers.add_parser("external")
    external_sub = external.add_subparsers(dest="external_kind", required=True)
    tdl = external_sub.add_parser("tdl")
    tdl_sub = tdl.add_subparsers(dest="tdl_command", required=True)
    for name, run in (("verify", False), ("smoke", True)):
        command_parser = tdl_sub.add_parser(name)
        command_parser.add_argument("--source", required=True)
        command_parser.add_argument("--binary", required=True)
        command_parser.add_argument("--config")
        command_parser.add_argument("--seed")
        command_parser.add_argument("--threads", type=int)
        command_parser.set_defaults(handler=lambda args, run=run: command_tdl(args, run=run))
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (ArtifactError, ValueError, RuntimeError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
