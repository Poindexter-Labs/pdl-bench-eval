"""Agents a replication attempt can run: the reference model agent, or a scripted baseline.

Model agents are the vendored reference loop (``ModelAgent`` over a provider
adapter). The two scripted baselines are what the reference runner calibrates
every task with: ``gold`` applies the task's reference patch through the same
``write_patch`` tool a model would use and submits (expected reward 1.0);
``noop`` submits at once (expected reward 0.0 under ``scoring_v0.2``).
"""
from __future__ import annotations

from pathlib import Path

from .._vendor.pd_rl.agents import NoopAgent, ScriptedAgent, _AgentFault, _Submit  # noqa: F401 (re-exported)

SCRIPTED = ("gold", "noop")
MODEL_FAMILIES = ("openai", "anthropic", "gemini", "maas")


class GoldAgent(ScriptedAgent):
    """The reference patch, delivered the way the reference runner's gold agent delivers it."""

    name = "gold"

    def __init__(self, gold_path: Path):
        super().__init__()
        self.gold_path = Path(gold_path)

    def run(self, ctx, toolbox):
        toolbox.call("write_patch", {"patch": self.gold_path.read_text()})
        toolbox.call("submit", {})


def make_agent(config: dict, prompt_id: str, gold_path: Path | None = None):
    """Build the agent a model configuration names.

    ``config["provider"]`` is a provider family (openai | anthropic | gemini |
    maas) or a scripted baseline (gold | noop); ``endpoint`` selects the route
    exactly as the reference runner's ``--endpoint`` does (``azure``,
    ``vertex:global``, an OpenAI-compatible base URL, ...).
    """
    kind = config["provider"]
    if kind == "gold":
        if gold_path is None or not Path(gold_path).is_file():
            raise FileNotFoundError("the gold baseline needs the task's gold.diff")
        return GoldAgent(gold_path)
    if kind == "noop":
        return NoopAgent()
    if kind not in MODEL_FAMILIES:
        raise ValueError(f"unknown provider family {kind!r} (expected one of {MODEL_FAMILIES + SCRIPTED})")
    from .._vendor.pd_rl.providers import make_model_agent  # lazy: needs the provider SDK

    return make_model_agent(kind, model=config["model"], prompt=prompt_id, endpoint=config.get("endpoint"))


def required_env(config: dict) -> set:
    """Environment variables a configuration's adapter reads (names as the adapters spell them)."""
    kind, endpoint = config["provider"], config.get("endpoint") or ""
    if kind in SCRIPTED:
        return set()
    if kind == "openai":
        if endpoint == "azure":
            return {"AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"}
        return {"OPENAI_API_KEY"}
    if kind == "anthropic":
        if endpoint == "vertex":
            return {"ANTHROPIC_VERTEX_REGION", "GOOGLE_CLOUD_PROJECT"}
        return {"ANTHROPIC_API_KEY"}
    if kind == "gemini":
        return {"GOOGLE_CLOUD_PROJECT"} if endpoint.startswith("vertex") else {"GEMINI_API_KEY"}
    if kind == "maas":
        return {"GOOGLE_CLOUD_PROJECT"}
    return set()
