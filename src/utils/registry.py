from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from .io import read_yaml


DEFAULT_ARTIFACT_REGISTRY = "configs/registry/artifacts.yaml"
DEFAULT_EXPERIMENT_REGISTRY = "configs/registry/experiments.yaml"
DEFAULT_STRATEGY_REGISTRY = "configs/registry/strategies.yaml"


def load_registry(path: str | Path) -> dict[str, Any]:
    return read_yaml(path)


def _mapping(registry: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = registry.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"registry `{source}` must define a `{key}` mapping")
    return value


def resolve_artifact(registry: dict[str, Any], artifact_key: str, *, source: str = DEFAULT_ARTIFACT_REGISTRY) -> dict[str, Any]:
    artifacts = _mapping(registry, "artifacts", source=source)
    if artifact_key not in artifacts:
        choices = ", ".join(sorted(str(name) for name in artifacts))
        raise ValueError(f"unknown artifact `{artifact_key}` in `{source}`; available artifacts: {choices}")
    artifact = artifacts[artifact_key]
    if not isinstance(artifact, dict):
        raise ValueError(f"artifact `{artifact_key}` in `{source}` must be a mapping")
    return artifact


def artifact_path(registry: dict[str, Any], artifact_key: str, *, source: str = DEFAULT_ARTIFACT_REGISTRY) -> str:
    artifact = resolve_artifact(registry, artifact_key, source=source)
    path = artifact.get("path")
    if not path:
        raise ValueError(f"artifact `{artifact_key}` in `{source}` must define `path`")
    return str(path)


def resolve_bundle(registry: dict[str, Any], bundle_key: str, *, source: str = DEFAULT_ARTIFACT_REGISTRY) -> dict[str, str]:
    bundles = _mapping(registry, "bundles", source=source)
    if bundle_key not in bundles:
        choices = ", ".join(sorted(str(name) for name in bundles))
        raise ValueError(f"unknown artifact bundle `{bundle_key}` in `{source}`; available bundles: {choices}")
    bundle = bundles[bundle_key]
    if not isinstance(bundle, dict):
        raise ValueError(f"artifact bundle `{bundle_key}` in `{source}` must be a mapping")
    return {str(name): artifact_path(registry, str(artifact_key), source=source) for name, artifact_key in bundle.items()}


def resolve_experiment(registry: dict[str, Any], experiment_key: str, *, source: str = DEFAULT_EXPERIMENT_REGISTRY) -> dict[str, Any]:
    experiments = _mapping(registry, "experiments", source=source)
    if experiment_key not in experiments:
        choices = ", ".join(sorted(str(name) for name in experiments))
        raise ValueError(f"unknown experiment `{experiment_key}` in `{source}`; available experiments: {choices}")
    experiment = experiments[experiment_key]
    if not isinstance(experiment, dict):
        raise ValueError(f"experiment `{experiment_key}` in `{source}` must be a mapping")
    return dict(experiment)


def resolve_strategy_run(registry: dict[str, Any], run_key: str, *, source: str = DEFAULT_STRATEGY_REGISTRY) -> dict[str, Any]:
    strategy_runs = _mapping(registry, "strategy_runs", source=source)
    if run_key not in strategy_runs:
        choices = ", ".join(sorted(str(name) for name in strategy_runs))
        raise ValueError(f"unknown strategy run `{run_key}` in `{source}`; available strategy runs: {choices}")
    strategy_run = strategy_runs[run_key]
    if not isinstance(strategy_run, dict):
        raise ValueError(f"strategy run `{run_key}` in `{source}` must be a mapping")
    return dict(strategy_run)


def parser_defaults(parser: argparse.ArgumentParser) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    for action in parser._actions:
        if action.dest and action.dest != "help":
            defaults[action.dest] = action.default
    return defaults


def _is_default(args: argparse.Namespace, defaults: dict[str, Any], key: str) -> bool:
    return getattr(args, key, None) == defaults.get(key)


def apply_experiment_defaults(args: argparse.Namespace, experiment_cfg: dict[str, Any], defaults: dict[str, Any]) -> None:
    for key in ["out_root", "run_name"]:
        if key in experiment_cfg and _is_default(args, defaults, key):
            setattr(args, key, experiment_cfg[key])
    for key, value in (experiment_cfg.get("args") or {}).items():
        dest = str(key).replace("-", "_")
        if hasattr(args, dest) and _is_default(args, defaults, dest):
            setattr(args, dest, value)
    if "artifact_bundle" in experiment_cfg and hasattr(args, "artifact_bundle") and _is_default(args, defaults, "artifact_bundle"):
        setattr(args, "artifact_bundle", experiment_cfg["artifact_bundle"])
