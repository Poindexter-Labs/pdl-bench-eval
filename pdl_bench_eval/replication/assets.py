"""The versioned reference assets the agent loop reads: prompts, tool schema, contract, profiles, schemas.

All of them are byte copies from the PD RL Environment repository (see
``pdl_bench_eval/_vendor/SOURCE_MANIFEST.json``); nothing here is edited.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

ASSETS = Path(__file__).resolve().parent.parent / "_vendor" / "pd_rl_assets"


@lru_cache(maxsize=None)
def _yaml(rel: str) -> dict:
    return yaml.safe_load((ASSETS / rel).read_text())


@lru_cache(maxsize=None)
def _json(rel: str) -> dict:
    return json.loads((ASSETS / rel).read_text())


def prompt(prompt_id: str) -> dict:
    return _yaml(f"prompts/{prompt_id}.yaml")


def profile(profile_id: str) -> dict:
    return _yaml(f"profiles/{profile_id}.yaml")


def contract(contract_id: str) -> dict:
    return _yaml(f"contracts/{contract_id}.yaml")


def tools(tools_id: str) -> dict:
    return _yaml(f"contracts/{tools_id}.yaml")


def trajectory_schema() -> dict:
    return _json("schemas/trajectory_event.schema.json")


def record_schema() -> dict:
    return _json("schemas/episode_record.schema.json")


def available() -> dict:
    return {kind: sorted(p.stem.replace(".schema", "") for p in (ASSETS / kind).iterdir())
            for kind in ("prompts", "profiles", "contracts", "schemas")}
