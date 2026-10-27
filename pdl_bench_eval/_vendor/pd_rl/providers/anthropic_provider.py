# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Anthropic adapter: Claude via the Messages API.

Two endpoints (DEC-041/DEC-044):
  * native (default) — api.anthropic.com with ANTHROPIC_API_KEY, falling
    back to the `anthropic-api-key` Secret Manager entry when the env var
    is unset, so the operator VM needs no key in its unit file; label
    anthropic:api.
  * vertex — Claude as a Vertex AI partner model, billed to the GCP
    project and authenticated with ADC (no Anthropic key). Requires
    ANTHROPIC_VERTEX_REGION (never assumed — the region is a recorded
    provenance/sovereignty fact); project from ANTHROPIC_VERTEX_PROJECT
    or GOOGLE_CLOUD_PROJECT, defaulting to the org project. Label
    anthropic-vertex:<region>. Vertex model ids carry an @version
    suffix (see the Model Garden card).

Claude tool-use format: system is a top-level argument, tools carry an
input_schema, and tool results are user-message content blocks keyed by
tool_use_id. max_tokens is required by the API, so it is set and recorded;
temperature is left at the vendor default.
"""

from __future__ import annotations

import os
import time

from pdl_bench_eval._vendor.pd_rl.providers.base import (
    ProviderAdapter, ToolCall, Turn, gcp_secret, prompt_cache_enabled, tool_list,
)

DEFAULT_MAX_TOKENS = 8192

#: Patient rate-limit backoff, the DEC-064 ladder the OpenAI and Google
#: adapters already use (node-scaling scope): the SDK's two quick retries
#: cannot outwait a per-minute window, and at 64+ episodes in flight Fable
#: runs close to its OTPM limit. ~5.5 minutes of waits, then the 429
#: propagates and the episode voids honestly. A spend-cap 429
#: (enforced_spend_limit_reached) is not a window and propagates at once.
_RATE_LIMIT_WAITS_S = (20, 40, 60, 90, 120)
DEFAULT_VERTEX_PROJECT = "poindexter-labs-rl-environment"

# The native key resolves from the ENVIRONMENT first, then Secret Manager —
# the same custody the Azure coordinates got in P1-017's amendment, for the
# same reason: the operator VM's attached service account reads the secret, so
# no provider key sits in the systemd unit and the VM stays keyless.
#
# This is what makes a native-only model orderable from the console at all.
# P1-017 scoped the console lineup to Vertex-served models because the VM held
# no third-party keys; that is now a solved problem rather than a constraint,
# which matters specifically for the Fable family — DEC-045 declined Anthropic's
# Vertex data-sharing requirement on contamination-cleanliness grounds
# (DEC-010), so native is the ONLY route Fable will ever have here.
ANTHROPIC_KEY_SECRET = "anthropic-api-key"


def _anthropic_key() -> str:
    return (os.environ.get("ANTHROPIC_API_KEY", "").strip()
            or gcp_secret(ANTHROPIC_KEY_SECRET))


# PROMPT CACHING. Claude is the only provider in this harness that requires a
# request-side opt-in: OpenAI and Gemini cache long prefixes automatically.
#
# Why it is worth the code. An agentic turn re-sends the whole conversation,
# so input is re-billed every turn and grows quadratically in turn count.
# Measured over the 106 scored episodes of python_pr_5m_v0.1, output was 2.0%
# of all tokens spent — the bill is almost entirely the model re-reading its
# own transcript. Cache reads are charged at a tenth of the input rate, so
# caching the prefix is the single largest lever available.
#
# The default 5-minute TTL is refreshed on every hit, so a turn every few
# seconds keeps the entry alive for the whole episode (measured: 6-11 minutes
# wall clock). A tool call that outruns the TTL — the profiles allow a single
# 600-second command — expires the entry and the next turn silently pays full
# price for one turn before re-warming. That is a cost regression, never a
# correctness one.
_CACHE_CONTROL = {"type": "ephemeral"}
# Anthropic allows at most four breakpoints per request. Two are spent on the
# rolling pair below and one on the system block, leaving one spare.
_ROLLING_BREAKPOINTS = 2


def _image_block(img):
    return {"type": "image",
            "source": {"type": "base64", "media_type": img.mime, "data": img.b64()}}


class AnthropicAdapter(ProviderAdapter):
    name = "anthropic"
    # Every Claude model this adapter routes reads images, in a user message
    # and inside a tool_result (P1-044).
    accepts_images = True

    def __init__(self, model_id=None, max_tokens=DEFAULT_MAX_TOKENS, vertex=False):
        if not model_id:
            raise ValueError("anthropic agent requires --model")
        self._vertex = bool(vertex)
        if self._vertex:
            region = os.environ.get("ANTHROPIC_VERTEX_REGION", "").strip().lower()
            if not region:
                raise ValueError(
                    "ANTHROPIC_VERTEX_REGION is not set (required for "
                    "--endpoint vertex; regions are recorded, never assumed)"
                )
            # Lowercase matters: the SDK special-cases exactly "global" to
            # pick the non-regional aiplatform host.
            self._region = region
            self._project = (
                os.environ.get("ANTHROPIC_VERTEX_PROJECT")
                or os.environ.get("GOOGLE_CLOUD_PROJECT")
                or DEFAULT_VERTEX_PROJECT
            )
            self.provider_label = f"anthropic-vertex:{region}"
            self._api_key = None
        else:
            self._region = None
            self._project = None
            # Resolved once at construction, not per call: the Secret Manager
            # read is a gcloud subprocess, and an unreadable key must fail the
            # episode before any budget is spent rather than mid-loop.
            self._api_key = _anthropic_key()
            if not self._api_key:
                raise ValueError(
                    "no Anthropic key: ANTHROPIC_API_KEY is unset and the Secret "
                    f"Manager entry {ANTHROPIC_KEY_SECRET!r} is unreadable."
                )
            self.provider_label = "anthropic:api"
        self.model_id = model_id
        self._max_tokens = max_tokens
        self._cache = prompt_cache_enabled()
        # Recorded so a row says whether it was measured with caching on.
        # model_parameters is a free-form JSON column, so this costs no schema
        # change — and without it, a future cost analysis cannot tell a cached
        # row from an uncached one.
        self._params = {"max_tokens": max_tokens,
                        "prompt_cache": "ephemeral" if self._cache else "off"}
        if self._vertex:
            self._params["vertex_project"] = self._project
        self._client = None
        self._system = None
        self._tools = None
        self._messages = None

    def parameters(self):
        return dict(self._params)

    def begin(self, system, tools_spec, opening_user, images=None):
        import anthropic

        if self._vertex:
            self._client = anthropic.AnthropicVertex(
                project_id=self._project, region=self._region
            )
        else:
            self._client = anthropic.Anthropic(api_key=self._api_key)
        self._system = system
        self._tools = [
            {
                "name": name,
                "description": desc,
                "input_schema": params or {"type": "object", "properties": {}},
            }
            for name, desc, params in tool_list(tools_spec)
        ]
        if images:
            self._messages = [{"role": "user", "content": [
                {"type": "text", "text": opening_user}, *[_image_block(i) for i in images]]}]
        else:
            self._messages = [{"role": "user", "content": opening_user}]

    def _system_param(self):
        """System as a cacheable block when caching is on, else a plain str.

        ONE breakpoint here covers the tool list as well. The cached prefix is
        ordered tools -> system -> messages, and a breakpoint caches everything
        up to and including itself, so marking the last tool separately would
        buy nothing and spend one of the four the API allows.
        """
        if not self._cache:
            return self._system
        return [{"type": "text", "text": self._system,
                 "cache_control": dict(_CACHE_CONTROL)}]

    def _mark_cache_breakpoints(self):
        """Move the rolling breakpoints onto the newest user turns.

        The conversation grows every turn, so a fixed breakpoint would stop
        covering it almost immediately. Marking the last two user messages
        keeps one entry warm while the newer one is written: the request reads
        the longest cached prefix it can find and writes only the delta beyond
        it, so each turn pays full price for its own new content and a tenth
        for everything before it.

        TWO rather than one because a single rolling mark has no fallback: if
        the newest write did not land, the request falls back past every
        message to the system block and re-reads the entire transcript.
        """
        for message in self._messages:
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        block.pop("cache_control", None)
        users = [i for i, m in enumerate(self._messages) if m.get("role") == "user"]
        for index in users[-_ROLLING_BREAKPOINTS:]:
            message = self._messages[index]
            if isinstance(message.get("content"), str):
                # add_user_text stores a bare string; only blocks take a
                # cache_control key.
                message["content"] = [{"type": "text", "text": message["content"]}]
            blocks = message["content"]
            if blocks and isinstance(blocks[-1], dict):
                blocks[-1]["cache_control"] = dict(_CACHE_CONTROL)

    def _with_rate_limit_patience(self, invoke):
        """Run one API call, waiting out per-minute 429 windows (DEC-064)."""
        import anthropic

        for wait in _RATE_LIMIT_WAITS_S:
            try:
                return invoke()
            except anthropic.RateLimitError as exc:
                if "enforced_spend_limit_reached" in str(exc):
                    raise
                print(f"[agent] provider rate-limited (429); waiting {wait}s", flush=True)
                time.sleep(wait)
        return invoke()

    def generate(self):
        if self._cache:
            self._mark_cache_breakpoints()
        response = self._with_rate_limit_patience(
            lambda: self._client.messages.create(
                model=self.model_id,
                max_tokens=self._max_tokens,
                system=self._system_param(),
                tools=self._tools,
                messages=self._messages,
            )
        )
        # store the assistant turn verbatim for the next request
        self._messages.append(
            {"role": "assistant", "content": [block.model_dump() for block in response.content]}
        )
        calls = []
        for block in response.content:
            if block.type == "tool_use":
                calls.append(ToolCall(id=block.id, name=block.name, args=dict(block.input or {})))
        usage = response.usage
        # ACCOUNTING, and the reason enabling caching is not a one-line change.
        #
        # Anthropic's `input_tokens` EXCLUDES both cache reads and cache
        # writes; they are reported separately. Every other provider here
        # reports a total with the cached part as a subset. So taking
        # input_tokens alone would have made a cached Claude episode look like
        # it spent a tenth of its real input, silently multiplying the
        # effective agent_budget by ~10 and making every row after this change
        # incomparable with every row before it — a measurement change wearing
        # the costume of a cost optimisation.
        #
        # Summing the three keeps tokens_in meaning exactly what it meant
        # before: total input the model read this turn. The budget, the 80%
        # warning and token_budget_reached are all unaffected by caching, which
        # is the point.
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        return Turn(
            calls=calls,
            tokens_in=(getattr(usage, "input_tokens", 0) or 0) + cache_read + cache_write,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            cached_tokens_in=cache_read,
            cache_write_tokens=cache_write,
            finish_reason=response.stop_reason or "",
            model_version=getattr(response, "model", None),
            dropped=(not calls and not any(b.type == "text" and b.text.strip()
                                           for b in response.content)),
        )

    def add_tool_results(self, results):
        content = []
        for r in results:
            block = {"type": "tool_result", "tool_use_id": r.id}
            if r.error is None and r.images:
                # A tool_result carries images natively: the model reads them
                # as the output of the call that returned them.
                block["content"] = [{"type": "text", "text": r.output},
                                    *[_image_block(i) for i in r.images]]
            elif r.error is None:
                block["content"] = r.output
            else:
                block["content"] = r.error
                block["is_error"] = True
            content.append(block)
        self._messages.append({"role": "user", "content": content})

    def add_user_text(self, text):
        self._messages.append({"role": "user", "content": text})
