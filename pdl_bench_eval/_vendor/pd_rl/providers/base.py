# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Provider-independent model agent (DEC-040).

The shared agent loop lives here once: prompt rendering, tool-schema handoff,
budget and token accounting, trajectory emission, and the nudge/termination
semantics (including the DEC-033 dropped-turn handling). Each provider
contributes a thin adapter that owns only three things — converting the
tools_v0.1 schema to its function-call format, making one inference call,
and formatting message history. New providers are new adapters, not new
loops (DEC-039: evaluate as many models/providers as possible).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import yaml

from pdl_bench_eval._vendor.pd_rl.agents import _AgentFault, _Submit
from pdl_bench_eval._vendor.pd_rl.sandbox import ORIGIN_PATH

DEFAULT_PROMPT = "agent_prompt_v0.2"  # stress prompt; pilot pins v0.1 via --prompt
OPENING_USER = "Begin. Investigate the issue, fix it, then call submit."

# JSON-Schema subset that provider function-declaration schemas accept
_SCHEMA_KEYS = {"type", "description", "enum", "properties", "required", "items", "minimum"}


def sanitize_schema(schema: dict) -> dict:
    out = {}
    for key, value in schema.items():
        if key not in _SCHEMA_KEYS:
            continue
        if key == "properties":
            out[key] = {n: sanitize_schema(s) for n, s in value.items()}
        elif key == "items":
            out[key] = sanitize_schema(value)
        else:
            out[key] = value
    return out


def tool_list(tools_spec: dict):
    """Normalized [(name, description, parameters_or_None)] from tools_v0.1.

    parameters is None for zero-parameter tools (e.g. submit) — several APIs
    reject an empty-object schema; each adapter substitutes its own accepted
    empty form.
    """
    out = []
    for name, tool in tools_spec["tools"].items():
        params = sanitize_schema(tool["parameters"])
        out.append((name, tool["purpose"], params if params.get("properties") else None))
    return out


# Fraction of a budget at which the agent is warned it is running out.
#
# The agent is told its limits once, in the system prompt, and has no way to
# observe consumption: there is no usage tool, no per-turn counter, and nothing
# in the tool-result envelope carries a running total. Without a warning it is
# cut off mid-work by the hard check below with nothing written -- every
# zero-scoring episode in rlj-20260827-001515-b51ca2 (14 of them) was
# `token_budget_reached` with an EMPTY patch, on tasks the same models scored
# 0.8+ on when they finished.
#
# One warning per budget, not one per turn: a repeated nudge would spend the
# very budget it is warning about, and the model has already been told once.
# The threshold is a module constant rather than a profile field because
# agent_budget is closed (additionalProperties: false) and this is a property
# of the harness, not of a task's contract.
BUDGET_WARN_FRACTION = 0.8


def render_prompt(spec: dict, ctx: dict) -> str:
    manifest, limits = ctx["manifest"], ctx["limits"]
    policy = manifest.get("patch_policy", {})
    variables = {
        "working_directory": ctx["contract"]["terminal"]["default_working_directory"],
        "issue_text": ctx["issue_text"].strip(),
        "allowed_paths": ", ".join(policy.get("allowed_paths", ["(any)"])),
        "forbidden_paths": ", ".join(policy.get("forbidden_paths", ["(none)"])),
        "public_command": manifest["evaluation"]["public_command"],
        "max_turns": limits["agent_budget"]["max_turns"],
        "max_tokens_total": limits["agent_budget"]["max_tokens_total"],
    }
    # FrontierCode variables (D-6), supplied only when the task declares a
    # pull-request contract. A PR prompt rendered against a non-PR manifest
    # therefore raises KeyError rather than quietly producing a task
    # description that omits the branching requirement — which is the one
    # mis-order that does not error but does silently floor every episode.
    # validate.py refuses the combination up front (D-8a); this is the
    # backstop for anything that routes around the validator.
    pull_request = manifest["evaluation"].get("pull_request")
    if pull_request:
        variables.update({
            "base_branch": pull_request["base_branch"],
            "branch_name": pull_request["branch_name"],
            "origin_path": ctx.get("origin_path", ORIGIN_PATH),
        })
    return spec["system_template"].format(**variables)


@dataclass
class ToolCall:
    id: str
    name: str
    args: dict


@dataclass
class ToolResult:
    id: str
    name: str
    output: str = None
    error: str = None
    # pd_rl.images.Image objects the tool returned (read_file on an image,
    # P1-044). `output` still carries the text the model reads alongside them.
    images: list = field(default_factory=list)


def gcp_secret(name: str) -> str:
    """Replication build: credentials come from environment variables only.

    The reference runner falls back to Google Secret Manager; a researcher's
    copy never does, so a missing key is reported by the caller as missing.
    """
    return ""


def prompt_cache_enabled() -> bool:
    """Whether adapters should place prompt-cache breakpoints.

    On by default. The kill switch exists because caching is the one thing an
    adapter does that changes what the PROVIDER charges without changing what
    the harness measures — so if a provider ever mis-reports cache usage, the
    fix must not require a code change mid-batch. PDL_PROMPT_CACHE=0 restores
    the uncached behaviour every row before 2026-08-29 was measured under.

    Only adapters that place breakpoints read this (Anthropic today). OpenAI
    and Gemini cache automatically with no request-side opt-in, so there is
    nothing for the switch to turn off there.
    """
    return os.environ.get("PDL_PROMPT_CACHE", "1").strip().lower() not in (
        "0", "false", "no", "off",
    )


@dataclass
class Turn:
    """One normalized model response."""

    calls: list = field(default_factory=list)
    # tokens_in is ALWAYS the full billed input for the turn, cached or not.
    # Every adapter is responsible for making that true on its own provider —
    # see the note in anthropic_provider.generate, where it is not free.
    tokens_in: int = 0
    tokens_out: int = 0
    # Diagnostics, not budget. cached_tokens_in is the SUBSET of tokens_in the
    # provider served from a prompt cache; cache_write_tokens the subset it
    # wrote into one. Neither is added to anything — they exist so the cost of
    # a run can be read off its trajectory.
    cached_tokens_in: int = 0
    cache_write_tokens: int = 0
    finish_reason: str = ""
    model_version: str = None
    dropped: bool = False  # provider returned no usable content (DEC-033 analog)


class ProviderAdapter:
    """Interface. Concrete adapters set provider_label/name/model_id and
    implement the four conversation methods below.

    Images (P1-044): an adapter that can show a model images sets
    `accepts_images` and takes `images` in begin() (the task's own, attached to
    the opening message) and on ToolResult.images in add_tool_results(). The
    default is False, so an adapter that has not been taught images is refused
    an image task up front rather than silently dropping what it cannot send.
    """

    provider_label = "?"
    name = "?"
    model_id = None
    accepts_images = False

    def parameters(self) -> dict:
        return {}

    def begin(self, system: str, tools_spec: dict, opening_user: str, images=None) -> None:
        raise NotImplementedError

    def generate(self) -> Turn:
        raise NotImplementedError

    def add_user_text(self, text: str) -> None:
        raise NotImplementedError

    def add_tool_results(self, results) -> None:
        raise NotImplementedError


class ModelAgent:
    """Drives an episode through the toolbox using any ProviderAdapter."""

    def __init__(self, adapter: ProviderAdapter, prompt_id: str = None):
        self.adapter = adapter
        self.provider = adapter.provider_label
        self.name = adapter.name
        self.model_name = adapter.model_id
        self.model_version = adapter.model_id  # refined from the first response
        self.model_parameters = adapter.parameters()
        self.prompt_id = prompt_id or DEFAULT_PROMPT
        self.prompt_version = self.prompt_id
        self.accepts_images = bool(getattr(adapter, "accepts_images", False))
        self.usage = {"tokens_in": 0, "tokens_out": 0,
                      "cached_tokens_in": 0, "cache_write_tokens": 0}

    def _budget_warning(self, turn: int, budget: dict,
                        pull_request: dict = None) -> str | None:
        """The one-shot nudge to stop exploring and deliver.

        Returns None unless a budget has just crossed the threshold for the
        first time. Phrased as an instruction rather than a statistic: the
        failure being fixed is an agent that had the answer and never wrote it
        down, so the actionable half must come first.

        THE DELIVERABLE IS TASK-SHAPED, and getting that wrong wastes the
        warning. Measured on gemini-3.6-flash / pdl_fc_campaign_stack_009
        (ep-20260828-055751-5c24): the generic wording below fired at turn 64
        on 81.6% of budget, and the agent — which had already created its
        branch on turn 4 and written a fix that passes 13 of 16 hidden tests —
        spent its remaining ~1M tokens on `git status` and `git diff` and never
        committed. It scored the 0.20 floor for a 0.85 episode. The warning was
        obeyed in the letter: its deliverable *was* written. For a PR task the
        act that remains is `git commit` and `git push`, and the warning has to
        say so.
        """
        remedy = (
            "Commit your work and push your branch NOW: "
            f"`git add -A && git commit -m ... && git push origin {pull_request['branch_name']}`. "
            "Nothing uncommitted and nothing unpushed is evaluated — if you run "
            "out of budget before pushing, work that would have scored well "
            "scores as though you had done nothing. Then call submit."
        ) if pull_request else (
            "Stop investigating now and write your deliverable, then call "
            "submit. If you run out of budget before writing it, your work is "
            "lost and scores zero."
        )

        used = self.usage["tokens_in"] + self.usage["tokens_out"]
        cap = budget["max_tokens_total"]
        if not self._warned_tokens and used >= cap * BUDGET_WARN_FRACTION:
            self._warned_tokens = True
            return (
                f"BUDGET WARNING: you have used {used:,} of your {cap:,} token "
                f"budget ({used / cap:.0%}). {remedy}"
            )
        max_turns = budget["max_turns"]
        if not self._warned_turns and turn >= max_turns * BUDGET_WARN_FRACTION:
            self._warned_turns = True
            return (
                f"BUDGET WARNING: this was turn {turn} of {max_turns}. {remedy}"
            )
        return None

    def run(self, ctx, toolbox):
        prompt_spec = yaml.safe_load(
            (ctx["root"] / "prompts" / f"{self.prompt_id}.yaml").open()
        )
        self.prompt_version = prompt_spec["prompt_id"]
        system = render_prompt(prompt_spec, ctx)
        images = ctx.get("images") or []
        if images:
            # The task's own images (issue.images) ride on the opening message,
            # named so the model can tie each to the file in its workspace.
            names = ", ".join(f"`{img.name}`" for img in images)
            opening = f"{OPENING_USER}\n\nAttached: the task's image{'s' if len(images) > 1 else ''} ({names})."
            self.adapter.begin(system, ctx["specs"]["tools"], opening, images=images)
        else:
            self.adapter.begin(system, ctx["specs"]["tools"], OPENING_USER)

        budget = ctx["limits"]["agent_budget"]
        self._warned_tokens = False
        self._warned_turns = False
        # DynamicSWE (P1-038): turn-kind triggers read toolbox.turn only when a
        # model agent is advancing it; scripted agents leave it at 1 and the
        # controller falls back to the tool-call count.
        toolbox.turn_driven = True
        for turn in range(1, budget["max_turns"] + 1):
            toolbox.turn = turn
            started = time.monotonic()
            # API failures (unavailable model, quota, network) are infra, not
            # agent behavior: they propagate to the runner's DEC-033 net and
            # the episode records void_infra_failure. Only HTTP 429 is waited
            # out first, inside the OpenAI and Google adapters (DEC-064, P1-047).
            result = self.adapter.generate()
            latency_ms = int((time.monotonic() - started) * 1000)

            self.usage["tokens_in"] += result.tokens_in
            self.usage["tokens_out"] += result.tokens_out
            self.usage["cached_tokens_in"] += result.cached_tokens_in
            self.usage["cache_write_tokens"] += result.cache_write_tokens
            if result.model_version:
                self.model_version = result.model_version
            toolbox.emit_model_call(
                {
                    "turn": turn,
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "cached_tokens_in": result.cached_tokens_in,
                    "cache_write_tokens": result.cache_write_tokens,
                    "latency_ms": latency_ms,
                    "stop_reason": result.finish_reason,
                }
            )
            if self.usage["tokens_in"] + self.usage["tokens_out"] >= budget["max_tokens_total"]:
                raise _Submit("token_budget_reached")

            if result.calls:
                results = []
                for call in result.calls:
                    try:
                        output = toolbox.call(call.name, dict(call.args or {}))
                        shown = toolbox.take_images() if hasattr(toolbox, "take_images") else []
                        results.append(ToolResult(call.id, call.name, output=output, images=shown))
                    except _AgentFault as exc:
                        # malformed call: report back to the model instead of dying
                        results.append(ToolResult(call.id, call.name, error=str(exc)))
                self.adapter.add_tool_results(results)
            elif result.dropped:
                self.adapter.add_user_text(
                    f"Your last turn produced no usable output "
                    f"(finish_reason={result.finish_reason}). Re-issue it as a single "
                    "valid tool call; call submit when finished."
                )
            else:
                self.adapter.add_user_text(
                    "Use the tools to proceed; call submit when finished."
                )

            # Last thing appended, so it is the most recent instruction the
            # model reads on its next turn rather than being buried above a
            # wall of tool output.
            warning = self._budget_warning(
                turn, budget,
                pull_request=(ctx["manifest"].get("evaluation") or {}).get("pull_request"),
            )
            if warning:
                self.adapter.add_user_text(warning)
                toolbox.emit_budget_warning(
                    {
                        "turn": turn,
                        "tokens_used": self.usage["tokens_in"] + self.usage["tokens_out"],
                        "max_tokens_total": budget["max_tokens_total"],
                        "max_turns": budget["max_turns"],
                    }
                )

        raise _Submit("max_turns_reached")
