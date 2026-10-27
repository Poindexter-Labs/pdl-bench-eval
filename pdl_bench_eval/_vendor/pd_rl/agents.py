# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
from __future__ import annotations

class _Submit(Exception):
    """Raised by the submit tool (or a budget guard) to end the agent loop.

    str(exc) carries the termination reason when it is one of the contract's
    episode.ends_when values."""

class _AgentFault(Exception):
    """Unrecoverable agent-side error (maps to invalid_agent_fault)."""

class ScriptedAgent:
    provider = "scripted"
    model_version = "pd-rl-runner-0.1"
    model_parameters: dict = {}
    prompt_version = "scripted_v0"  # no prompt is rendered

    def __init__(self):
        self.usage = {"tokens_in": 0, "tokens_out": 0}

    @property
    def model_name(self):
        return self.name

class NoopAgent(ScriptedAgent):
    name = "noop"

    def run(self, ctx, toolbox):
        toolbox.call("submit", {})
