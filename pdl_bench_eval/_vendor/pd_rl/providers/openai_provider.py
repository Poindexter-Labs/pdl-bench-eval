# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""OpenAI adapter: native OpenAI (Responses API), any OpenAI-compatible host
(base_url, Chat Completions), or Azure OpenAI (Chat Completions).

Native OpenAI routes through the Responses API (/v1/responses): OpenAI's
current flagship line is reasoning models, and on Chat Completions they
cannot use function tools with reasoning on (DEC-040 follow-up). Responses
handles reasoning + non-reasoning uniformly and threads reasoning state
server-side via previous_response_id. Compat hosts (Together/Fireworks/vLLM)
and Azure implement Chat Completions, not Responses, so those stay on the
chat path. One adapter, two protocols, selected at construction.

Azure UK South is a London-region sovereign path (openai-azure:<region>,
region read from AZURE_OPENAI_REGION — never assumed). Sampling follows the
vendor default (no temperature/effort forced) and model_parameters records
the API mode used.
"""

from __future__ import annotations

import json
import os
import time
from urllib.parse import urlparse

# Azure coordinates resolve from the ENVIRONMENT first, then Secret Manager.
#
# The judge has read its key this way since P1-018 so the operator VM can stay
# keyless — its attached service account reads the secret and no provider key
# sits in the systemd unit. The agent adapter did not, so running gpt-5.6-sol
# as an EVALUATED MODEL (rather than as a judge) still required both values in
# the process environment, which is why the console lineup stayed Vertex-only
# (P1-017: "no third-party keys on the VM"). Reading both from Secret Manager
# removes that constraint without putting a key anywhere.
#
# Environment still wins, so a resource can be redirected for one run without
# touching Secret Manager.
AZURE_ENDPOINT_SECRET = "azure-openai-endpoint"
AZURE_KEY_SECRET = "azure-openai-api-key"


def _azure_endpoint() -> str:
    return (os.environ.get("AZURE_OPENAI_ENDPOINT", "").strip()
            or gcp_secret(AZURE_ENDPOINT_SECRET))


def _azure_key() -> str:
    return (os.environ.get("AZURE_OPENAI_API_KEY", "").strip()
            or gcp_secret(AZURE_KEY_SECRET))


_AZURE_REGION_CACHE = {}


def _azure_region(endpoint: str, key: str) -> str:
    """The region Azure says it served from, for the recorded provider label.

    ASKED, NOT ASSUMED. DEC-044 requires a region to be recorded rather than
    inferred, and the resource name is not a region — "tidor-sweden-resource"
    is a string someone chose. Azure returns the real one in the `x-ms-region`
    response header, so the label reports what actually served the request.
    Without this the recorded label was `openai-azure:unknown-region`: honest,
    but useless for the sovereignty filtering DEC-053 needs.

    AZURE_OPENAI_REGION still wins when set. Cached per endpoint — one extra
    GET per process, not per call — and a failure degrades to the old
    placeholder rather than breaking the episode.
    """
    override = os.environ.get("AZURE_OPENAI_REGION", "").strip()
    if override:
        return override
    if endpoint in _AZURE_REGION_CACHE:
        return _AZURE_REGION_CACHE[endpoint]
    region = "unknown-region"
    try:
        import urllib.request

        req = urllib.request.Request(
            f"{endpoint.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {key}"},
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            # "Sweden Central" -> "swedencentral", Azure's own region id form,
            # so the label is greppable and matches their documentation.
            raw = response.headers.get("x-ms-region", "")
            region = raw.replace(" ", "").lower() or "unknown-region"
    except Exception:  # noqa: BLE001 - a label must never fail an episode
        pass
    _AZURE_REGION_CACHE[endpoint] = region
    return region

from pdl_bench_eval._vendor.pd_rl.providers.base import (
    ProviderAdapter, ToolCall, Turn, gcp_secret, tool_list,
)

#: Patient rate-limit backoff (DEC-064): per-minute request caps on
#: compat providers (first seen: Cohere) starve agentic call patterns —
#: the SDK's two quick retries cannot outwait a minute-granularity
#: window, so episodes void on weather the adapter could simply wait
#: out. Waits total ~5.5 minutes before the 429 propagates and the
#: episode voids honestly. ONLY HTTP 429 is retried here; every other
#: error keeps its DEC-033 semantics.
_RATE_LIMIT_WAITS_S = (20, 40, 60, 90, 120)


def _gcloud_access_token():
    import subprocess

    try:
        proc = subprocess.run(
            ["gcloud", "auth", "print-access-token"],
            capture_output=True, text=True, timeout=30, check=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"could not obtain gcloud access token: {exc}")
    return proc.stdout.strip()


def _images_note(results):
    """(caption, images) for the images a batch of tool results returned."""
    shown = [(r, i) for r in results if r.error is None for i in r.images]
    if not shown:
        return None, []
    caption = "Images returned by the tool calls above: " + ", ".join(
        f"{r.name} -> {i.name}" for r, i in shown)
    return caption, [i for _, i in shown]


class OpenAIAdapter(ProviderAdapter):
    name = "openai"

    @property
    def accepts_images(self):
        # P1-044, probed live 2026-09-29 (an image in the opening message and
        # one returned by read_file): GPT-5.6 on Azure (Responses) and Grok 4.3
        # on Vertex openapi (Chat) read both. Llama 4 Maverick on Vertex refuses
        # any request that combines vision input with function calling (HTTP
        # 400), and an agent episode always carries tools — so it is refused
        # an image task up front instead of voiding on turn one.
        return not self.model_id.startswith("meta/llama")

    def __init__(self, model_id=None, base_url=None, azure=False):
        if not model_id:
            raise ValueError(
                "openai agent requires --model (for classic Azure this is the "
                "deployment name; AI Foundry v1 endpoints route by model name)"
            )
        self.model_id = model_id
        self.base_url = base_url
        self.azure = azure
        # Protocol per route: native OpenAI and Azure AI Foundry v1
        # (/openai/v1) speak the Responses API (reasoning-model capable —
        # the v1 surface routes by model name, plain OpenAI client, no
        # api-version; verified against the production Poindexter flow).
        # Compat hosts and classic Azure stay on Chat Completions.
        self.azure_v1 = azure and "/openai/v1" in _azure_endpoint()
        if self.azure_v1 or not (base_url or azure):
            self.mode = "responses"
        else:
            self.mode = "chat"
        if azure:
            self.provider_label = f"openai-azure:{_azure_region(_azure_endpoint(), _azure_key())}"
        elif base_url:
            self.provider_label = f"openai-compat:{urlparse(base_url).hostname or base_url}"
        else:
            self.provider_label = "openai:api"
        self._params = {"api_mode": "responses" if self.mode == "responses" else "chat_completions",
                        "sampling": "provider-default"}
        if base_url and (urlparse(base_url).hostname or "").endswith(
            "aiplatform.googleapis.com"
        ):
            self._params["auth"] = "gcloud-access-token"
        self._client = None
        # chat state
        self._messages = None
        self._chat_tools = None
        # responses state
        self._system = None
        self._resp_tools = None
        self._pending = None
        self._prev_id = None

    def parameters(self):
        return dict(self._params)

    def _make_client(self):
        from openai import OpenAI

        if self.azure:
            endpoint = _azure_endpoint()
            key = _azure_key()
            if not endpoint:
                raise RuntimeError(
                    "no Azure endpoint: AZURE_OPENAI_ENDPOINT is unset and the "
                    f"Secret Manager entry {AZURE_ENDPOINT_SECRET!r} is unreadable. "
                    "An endpoint is a deployment coordinate and is never assumed "
                    "(DEC-044)."
                )
            if not key:
                raise RuntimeError(
                    "no Azure key: AZURE_OPENAI_API_KEY is unset and the Secret "
                    f"Manager entry {AZURE_KEY_SECRET!r} is unreadable."
                )
            if self.azure_v1:
                # AI Foundry v1: a plain OpenAI-shaped surface. The
                # AzureOpenAI class must NOT be used here — with an
                # api-version it rewrites requests onto the legacy
                # /openai/deployments/{name}/ paths, which 404 against
                # the v1 base for every name (the original M4 blocker).
                return OpenAI(base_url=endpoint.rstrip("/"), api_key=key)
            from openai import AzureOpenAI

            return AzureOpenAI(
                api_key=key,
                azure_endpoint=endpoint,
                api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"),
            )
        kwargs = {}
        if self.base_url:
            kwargs["base_url"] = self.base_url
            host = urlparse(self.base_url).hostname or ""
            if host.endswith("aiplatform.googleapis.com"):
                # Vertex MaaS openapi surface (DEC-044/051): googleapis
                # never accepts an OpenAI key — auth is a gcloud access
                # token, fetched fresh per episode so a shell-exported
                # token cannot expire mid-batch and OPENAI_API_KEY stays
                # free for native OpenAI entries in the same batch.
                kwargs["api_key"] = _gcloud_access_token()
        return OpenAI(**kwargs)  # otherwise OPENAI_API_KEY from env

    def begin(self, system, tools_spec, opening_user, images=None):
        self._client = self._make_client()
        tools = tool_list(tools_spec)
        if self.mode == "responses":
            self._system = system
            self._resp_tools = [
                {
                    "type": "function",
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                }
                for name, desc, params in tools
            ]
            self._pending = [{"role": "user", "content": opening_user if not images else [
                {"type": "input_text", "text": opening_user},
                *[{"type": "input_image", "image_url": i.data_url()} for i in images]]}]
            self._prev_id = None
        else:
            self._chat_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": desc,
                        "parameters": params or {"type": "object", "properties": {}},
                    },
                }
                for name, desc, params in tools
            ]
            self._messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": opening_user if not images else [
                    {"type": "text", "text": opening_user},
                    *[{"type": "image_url", "image_url": {"url": i.data_url()}} for i in images]]},
            ]

    # -- patient rate-limit handling ------------------------------------------
    @staticmethod
    def _with_rate_limit_patience(invoke):
        """Run one API call, waiting out per-minute 429 windows.

        Retries ONLY RateLimitError; on the final attempt the error
        propagates so the episode voids honestly (DEC-033 net).
        """
        from openai import RateLimitError

        for wait in _RATE_LIMIT_WAITS_S:
            try:
                return invoke()
            except RateLimitError:
                print(
                    f"[agent] provider rate-limited (429); waiting {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        return invoke()

    # -- Responses API path --------------------------------------------------
    def _generate_responses(self):
        response = self._with_rate_limit_patience(
            lambda: self._client.responses.create(
                model=self.model_id,
                instructions=self._system,
                input=self._pending,
                tools=self._resp_tools,
                tool_choice="auto",
                previous_response_id=self._prev_id,
            )
        )
        self._prev_id = response.id
        self._pending = []
        calls = []
        for item in response.output:
            if getattr(item, "type", None) == "function_call":
                try:
                    args = json.loads(item.arguments or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append(ToolCall(id=item.call_id, name=item.name, args=args))
        usage = response.usage
        status = response.status or ""
        if status == "incomplete" and getattr(response, "incomplete_details", None):
            status = getattr(response.incomplete_details, "reason", status) or status
        has_text = bool((getattr(response, "output_text", "") or "").strip())
        # OpenAI caches prefixes over ~1024 tokens AUTOMATICALLY; there is no
        # opt-in to add. input_tokens is the full input with cached_tokens as a
        # subset, so tokens_in is already correct — unlike Anthropic, where the
        # cached part is reported outside input_tokens.
        in_details = getattr(usage, "input_tokens_details", None)
        return Turn(
            calls=calls,
            tokens_in=getattr(usage, "input_tokens", 0) or 0,
            tokens_out=getattr(usage, "output_tokens", 0) or 0,
            cached_tokens_in=getattr(in_details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(in_details, "cache_write_tokens", 0) or 0,
            finish_reason=status,
            model_version=getattr(response, "model", None),
            dropped=(not calls and not has_text),
        )

    def _add_tool_results_responses(self, results):
        for r in results:
            self._pending.append(
                {
                    "type": "function_call_output",
                    "call_id": r.id,
                    "output": r.output if r.error is None else f"ERROR: {r.error}",
                }
            )
        # A function_call_output is text; images follow as one user message.
        caption, shown = _images_note(results)
        if shown:
            self._pending.append({"role": "user", "content": [
                {"type": "input_text", "text": caption},
                *[{"type": "input_image", "image_url": i.data_url()} for i in shown]]})

    # -- Chat Completions path -----------------------------------------------
    def _generate_chat(self):
        response = self._with_rate_limit_patience(
            lambda: self._client.chat.completions.create(
                model=self.model_id,
                messages=self._messages,
                tools=self._chat_tools,
                tool_choice="auto",
            )
        )
        choice = response.choices[0]
        message = choice.message
        self._messages.append(message.model_dump(exclude_none=True))
        calls = []
        for tc in message.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=tc.id, name=tc.function.name, args=args))
        usage = response.usage
        # Chat Completions reports the same automatic caching under a different
        # name; prompt_tokens likewise already includes the cached subset.
        prompt_details = getattr(usage, "prompt_tokens_details", None)
        return Turn(
            calls=calls,
            tokens_in=getattr(usage, "prompt_tokens", 0) or 0,
            tokens_out=getattr(usage, "completion_tokens", 0) or 0,
            cached_tokens_in=getattr(prompt_details, "cached_tokens", 0) or 0,
            cache_write_tokens=getattr(prompt_details, "cache_write_tokens", 0) or 0,
            finish_reason=choice.finish_reason or "",
            model_version=getattr(response, "model", None),
            dropped=(not calls and not (message.content or "").strip()),
        )

    def _add_tool_results_chat(self, results):
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.id,
                    "content": r.output if r.error is None else f"ERROR: {r.error}",
                }
            )
        # A tool message is text on Chat Completions; images follow as a user
        # message after every tool message of the batch.
        caption, shown = _images_note(results)
        if shown:
            self._messages.append({"role": "user", "content": [
                {"type": "text", "text": caption},
                *[{"type": "image_url", "image_url": {"url": i.data_url()}} for i in shown]]})

    # -- dispatch ------------------------------------------------------------
    def generate(self):
        return self._generate_responses() if self.mode == "responses" else self._generate_chat()

    def add_tool_results(self, results):
        if self.mode == "responses":
            self._add_tool_results_responses(results)
        else:
            self._add_tool_results_chat(results)

    def add_user_text(self, text):
        if self.mode == "responses":
            self._pending.append({"role": "user", "content": text})
        else:
            self._messages.append({"role": "user", "content": text})
