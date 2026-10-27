# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Google adapter: Gemini via AI Studio (GEMINI_API_KEY) or Vertex AI.

Vertex mode pins inference to a regional endpoint (europe-west2) under ADC —
the UK-resident, budget-attributed path (DEC-041). AI Studio remains the
key-based non-sovereign path used for early smoke work.
"""

from __future__ import annotations

import os
import re
import time

from pdl_bench_eval._vendor.pd_rl.providers.base import ProviderAdapter, ToolCall, Turn, tool_list

DEFAULT_MODEL = "gemini-2.5-flash"  # Gemini keeps a default; other providers require --model
PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT", "")

#: Patient rate-limit backoff (P1-047, the DEC-064 ladder ported from the
#: OpenAI adapter): Vertex answers a shared-capacity squeeze with 429
#: RESOURCE_EXHAUSTED ("TPM quota exhausted"), and this adapter used to let
#: the first one void the episode — 28 of 38 voids in the first Gemini 4
#: Argon job (rlj-20261013-232626-6fc6bd), all inside its busiest two hours.
#: Same waits as DEC-064 (~5.5 minutes) before the 429 propagates and the
#: episode voids honestly. ONLY HTTP 429 is retried; every other error keeps
#: its DEC-033 semantics.
_RATE_LIMIT_WAITS_S = (20, 40, 60, 90, 120)

#: Per-request HTTP timeout (P1-047). The SDK default is no timeout, so a
#: stalled connection blocked generate() forever — the agent and absolute
#: episode clocks are only checked between turns, so nothing inside the
#: episode could end it. The prime suspect for the 13-hour wedge in the
#: same job. Twenty minutes: Google's Argon guide asks for at least 15 for
#: hard prompts, and that job's own calls reached 631s (p99 495s across
#: 2,924) — the first value here, 600s, would have voided a live turn. A
#: timeout raises, and the runner's DEC-033 net records void_infra_failure.
REQUEST_TIMEOUT_S = 1200


def _generation(model_id):
    """Major Gemini generation from the id: `gemini-2.5-flash` -> 2,
    `gemini-3.8-flash` -> 3, `gemini-4-argon` -> 4. The adapter branches on
    `>= 3` in two places (temperature, image-in-tool-result shape) because the
    thinking generations share both behaviours and a prefix test on
    `gemini-3` silently dropped Gemini 4 into the 2.5 branch (found adding
    `gemini-4-argon`, 2026-10-13). An id that does not parse counts as 0 —
    the 2.5-era branch, which is the conservative default for a shape that
    has never been probed."""
    m = re.match(r"gemini-(\d+)", model_id or "")
    return int(m.group(1)) if m else 0


class GoogleAdapter(ProviderAdapter):
    name = "gemini"
    # Gemini 2.5 and 3.x read inline images in a user turn (P1-044).
    accepts_images = True

    def __init__(self, model_id=None, vertex=False, location="europe-west2"):
        self.model_id = model_id or DEFAULT_MODEL
        self.vertex = vertex
        self.location = location
        if vertex:
            self.provider_label = f"google-vertex:{location}"
        else:
            if not os.environ.get("GEMINI_API_KEY"):
                raise ValueError(
                    "GEMINI_API_KEY is not set — create a key at aistudio.google.com "
                    "and export it, or use the Vertex endpoint for a sovereign run"
                )
            self.provider_label = "google-aistudio"
        # Gemini 3 thinking models degrade at temperature 0 (Google recommends
        # 1.0). Gemini 4 goes further: the Argon developer guide (read
        # 2026-10-13) says custom sampling parameters are IGNORED and the model
        # runs at its default of 1.0, so 1.0 is also the only truthful record
        # there. Recorded model_parameters reflect whichever is used.
        temperature = 1.0 if _generation(self.model_id) >= 3 else 0
        self._params = {"temperature": temperature}
        self._client = None
        self._config = None
        self._contents = None
        self._types = None

    def parameters(self):
        return dict(self._params)

    def begin(self, system, tools_spec, opening_user, images=None):
        from google import genai
        from google.genai import types

        self._types = types
        declarations = [
            {"name": name, "description": desc, "parameters": params}
            for name, desc, params in tool_list(tools_spec)
        ]
        http_options = types.HttpOptions(timeout=REQUEST_TIMEOUT_S * 1000)  # milliseconds
        if self.vertex:
            self._client = genai.Client(
                vertexai=True, project=PROJECT, location=self.location,
                http_options=http_options,
            )
        else:
            self._client = genai.Client(http_options=http_options)  # GEMINI_API_KEY
        self._config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self._params["temperature"],
            tools=[types.Tool(function_declarations=declarations)],
        )
        self._contents = [
            types.Content(role="user", parts=[
                types.Part(text=opening_user),
                *[types.Part.from_bytes(data=i.data, mime_type=i.mime) for i in images or []],
            ])
        ]

    def _with_rate_limit_patience(self, invoke):
        """Run one API call, waiting out 429 RESOURCE_EXHAUSTED windows.

        Retries ONLY HTTP 429; on the final attempt the error propagates so
        the episode voids honestly (DEC-033 net).
        """
        from google.genai import errors

        for wait in _RATE_LIMIT_WAITS_S:
            try:
                return invoke()
            except errors.ClientError as exc:
                if exc.code != 429:
                    raise
                print(
                    f"[agent] provider rate-limited (429); waiting {wait}s",
                    flush=True,
                )
                time.sleep(wait)
        return invoke()

    def generate(self):
        response = self._with_rate_limit_patience(
            lambda: self._client.models.generate_content(
                model=self.model_id, contents=self._contents, config=self._config
            )
        )
        usage = response.usage_metadata
        tokens_in = usage.prompt_token_count or 0
        tokens_out = (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        # Gemini caches long prefixes IMPLICITLY on 2.5+ and 3.x — there is no
        # request-side opt-in to add, and nothing for PDL_PROMPT_CACHE to turn
        # off. prompt_token_count is already the full input with the cached
        # part as a subset, so tokens_in needs no adjustment; the subset is
        # reported here for cost visibility only.
        cached = getattr(usage, "cached_content_token_count", 0) or 0
        candidate = response.candidates[0]
        content = candidate.content
        parts = (content.parts if content is not None else None) or []
        calls = []
        for part in parts:
            fc = getattr(part, "function_call", None)
            if fc:
                calls.append(ToolCall(id=fc.name, name=fc.name, args=dict(fc.args or {})))
        if content is not None:
            self._contents.append(content)
        reason = getattr(candidate.finish_reason, "name", str(candidate.finish_reason))
        return Turn(
            calls=calls,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cached_tokens_in=cached,
            finish_reason=reason,
            model_version=getattr(response, "model_version", None),
            dropped=(not calls and not parts),
        )

    def add_tool_results(self, results):
        types = self._types
        parts = []
        # Images a tool returned (P1-044) travel differently by generation,
        # measured 2026-09-29: Gemini 3.x takes them INSIDE the function
        # response (multimodal function responses) and rejects inline parts
        # beside it ("Requests ending with a model turn are not supported");
        # 2.5 has no multimodal function response and takes them as inline
        # parts after the responses in the same user turn. Gemini 4 Argon
        # probed 2026-10-13 on Vertex global: same as 3.x — the multimodal
        # function response is accepted (and the number in the image read
        # back correctly); the inline form is rejected with the same 400.
        multimodal_fr = _generation(self.model_id) >= 3
        trailing = []
        for r in results:
            payload = {"output": r.output} if r.error is None else {"error": r.error}
            shown = r.images if r.error is None else []
            if shown and multimodal_fr:
                parts.append(types.Part.from_function_response(
                    name=r.name, response=payload,
                    parts=[types.FunctionResponsePart(inline_data=types.FunctionResponseBlob(
                        mime_type=img.mime, data=img.data, display_name=img.name)) for img in shown]))
                continue
            parts.append(types.Part.from_function_response(name=r.name, response=payload))
            for img in shown:
                trailing.append(types.Part(text=f"Image returned by {r.name}: {img.name}"))
                trailing.append(types.Part.from_bytes(data=img.data, mime_type=img.mime))
        self._contents.append(types.Content(role="user", parts=parts + trailing))

    def add_user_text(self, text):
        types = self._types
        self._contents.append(
            types.Content(role="user", parts=[types.Part(text=text)])
        )
