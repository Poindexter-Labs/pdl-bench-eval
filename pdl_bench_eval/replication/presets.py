"""Presets: the authority for what a replication run does.

A preset names, per task family, the prompt, run profile, scoring policy, pass
threshold, sandbox image and (for judged tasks) the judge panel; and, per model,
the provider family, route and model id. Budgets are read from the vendored
profile the preset names, never typed twice.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from . import PRESET_DIR, assets
from .agents import MODEL_FAMILIES, SCRIPTED, required_env

FAMILY_KEYS = {"kind", "prompt", "profile", "scoring", "pass_threshold", "sandbox_image", "evaluator_image"}
MODEL_KEYS = {"id", "provider", "endpoint", "model", "attempts"}
# The non-SWE packages spell the Azure variables differently; accept both.
ENV_ALIASES = {"AZURE_OPENAI_KEY": "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL": "AZURE_OPENAI_ENDPOINT"}


def apply_env_aliases() -> None:
    for alias, canonical in ENV_ALIASES.items():
        if os.environ.get(alias, "").strip() and not os.environ.get(canonical, "").strip():
            os.environ[canonical] = os.environ[alias].strip()


def available() -> list:
    return sorted(p.stem for p in PRESET_DIR.glob("*.json"))


def load(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if not path.is_file():
        if not re.fullmatch(r"[a-z0-9_-]+", name_or_path):
            raise ValueError("choose a preset from: " + ", ".join(available()))
        path = PRESET_DIR / f"{name_or_path}.json"
        if not path.is_file():
            raise ValueError(f"unknown preset {name_or_path!r}; choose from: " + ", ".join(available()))
    preset = json.loads(path.read_text())
    validate(preset)
    return preset


def validate(preset: dict) -> None:
    for key in ("id", "version", "basis", "families", "models"):
        if key not in preset:
            raise ValueError(f"preset is missing {key!r}")
    for name, fam in preset["families"].items():
        missing = FAMILY_KEYS - set(fam)
        if missing:
            raise ValueError(f"family {name!r} is missing {sorted(missing)}")
        if fam["kind"] not in ("executable", "qna"):
            raise ValueError(f"family {name!r}: kind must be executable or qna")
        assets.profile(fam["profile"])        # must exist among the vendored assets
        assets.prompt(fam["prompt"])
    for cfg in preset["models"]:
        validate_config(cfg)


def validate_config(cfg: dict) -> None:
    if cfg.get("provider") in SCRIPTED:
        return
    missing = MODEL_KEYS - set(cfg)
    if missing:
        raise ValueError(f"model configuration {cfg.get('id')!r} is missing {sorted(missing)}")
    if cfg["provider"] not in MODEL_FAMILIES:
        raise ValueError(f"model configuration {cfg['id']!r}: unknown provider family {cfg['provider']!r}")
    if type(cfg["attempts"]) is not int or not 1 <= cfg["attempts"] <= 8:
        raise ValueError("attempts must be 1–8")


def family_for(preset: dict, kind: str) -> dict:
    for fam in preset["families"].values():
        if fam["kind"] == kind:
            return fam
    raise ValueError(f"preset {preset['id']} has no {kind!r} family")


def resolve_models(preset: dict, model_ids=None, attempts=None, agent=None) -> list:
    """The configurations a run covers, each a copy with ``attempts`` applied."""
    if agent:
        if agent not in SCRIPTED:
            raise ValueError(f"--agent must be one of {SCRIPTED}")
        base = next((dict(b) for b in preset.get("baselines", []) if b["id"] == agent),
                    {"id": agent, "provider": agent, "attempts": 1})
        base["attempts"] = attempts or base.get("attempts", 1)
        return [base]
    configs = [dict(c) for c in preset["models"]]
    if model_ids:
        configs = [c for c in configs if c["id"] in model_ids]
        if len(configs) != len(set(model_ids)):
            raise ValueError("unknown model configuration id; see plan output")
    for cfg in configs:
        if attempts is not None:
            cfg["attempts"] = attempts
        validate_config(cfg)
    return configs


def execution_config(cfg: dict) -> dict:
    """The identity of a configuration for attempt keys: everything except the attempt count and notes."""
    return {k: v for k, v in cfg.items() if k not in ("attempts", "note", "recorded_model_parameters", "route_label")}


def budgets(fam: dict) -> dict:
    profile = assets.profile(fam["profile"])
    return {s: profile.get(s) or {} for s in ("agent_budget", "timing", "resources", "terminal_limits")}


def judges_for(fam: dict, override: str | None):
    """Judge members for a judged family: ``--judges``/``PDL_JUDGES`` if given, else the policy's reference panel."""
    from ..judges import parse_members, panel_from_policy
    from ..scoring import load_policy

    if fam["kind"] != "qna":
        return []
    spec = override or os.environ.get("PDL_JUDGES", "").strip()
    if spec:
        return parse_members(spec.split(","))
    return panel_from_policy(load_policy(fam["scoring"]))


def judge_env(members) -> set:
    keys = set()
    for m in members:
        if m.provider == "openai":
            keys.add("OPENAI_API_KEY")
            if not m.base_url:
                keys.add("OPENAI_BASE_URL")
        elif m.provider == "azure":
            keys.add("AZURE_OPENAI_API_KEY")
            if not m.base_url:
                keys.add("AZURE_OPENAI_ENDPOINT")
        elif m.provider == "anthropic":
            keys.add("ANTHROPIC_API_KEY")
        elif m.provider == "google":
            keys.add("GOOGLE_CLOUD_PROJECT|GEMINI_API_KEY")
    return keys


def required_keys(configs, judges_by_family: dict, *, solving=True) -> list:
    keys = set()
    if solving:
        for cfg in configs:
            keys |= required_env(cfg)
    for members in judges_by_family.values():
        keys |= judge_env(members)
    return sorted(keys)


def missing_keys(required) -> list:
    out = []
    for key in required:
        options = key.split("|")
        if not any(os.environ.get(o, "").strip() for o in options):
            out.append(key)
    return out


def plan_view(preset: dict, tasks: list, configs: list, judges_by_family: dict, parallel: int = 1) -> dict:
    families_used = sorted({t["family"] for t in tasks})
    view_families = {}
    for name, fam in preset["families"].items():
        if fam["kind"] not in families_used:
            continue
        view_families[name] = {**fam, "budgets": budgets(fam),
                               "judges": [m.label for m in judges_by_family.get(fam["kind"], [])]}
    return {
        "benchmark": preset["id"],
        "preset_version": preset["version"],
        "basis": preset["basis"],
        "tasks": len(tasks),
        "tasks_by_family": {f: sum(1 for t in tasks if t["family"] == f) for f in families_used},
        "attempts": sum(c["attempts"] for c in configs) * len(tasks),
        "models": configs,
        "families": view_families,
        "required_environment_variables": required_keys(configs, judges_by_family),
        "execution": (f"{'sequential' if parallel <= 1 else f'{parallel} concurrent episodes'}; run/resume is explicit; "
                      "technical failures remain ungraded; each attempt is a full agent episode in a Docker sandbox"),
    }
