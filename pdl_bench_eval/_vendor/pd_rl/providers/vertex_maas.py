# Vendored from the PD RL Environment repository by tools/vendor_rl.py; see SOURCE_MANIFEST.json.
"""Vertex AI MaaS adapter: partner models over rawPredict (DEC-045).

Some Model Garden partner publishers (Mistral, at minimum) do not serve
the unified ``endpoints/openapi`` Chat Completions surface — their
models answer only on the per-model ``:rawPredict`` verb, with a
request/response body that is itself OpenAI Chat Completions-shaped.
This adapter builds the same message/tool payloads as the OpenAI chat
path and posts them directly, authenticating with an access token from
``gcloud auth print-access-token`` fetched at episode start (and
refetched once on a 401, so hour-long token validity cannot bite).

Model ids are ``<publisher>/<model>`` exactly as the Model Garden card's
resource name shows (``publishers/mistralai/models/mistral-medium-3`` ->
``mistralai/mistral-medium-3``). The region comes from ``--endpoint``
and is recorded in the provider label ``vertex-maas:<region>`` — never
assumed (DEC-041).
"""

from __future__ import annotations

import json
import os
import re
import subprocess

from pdl_bench_eval._vendor.pd_rl.providers.base import ProviderAdapter, ToolCall, Turn, tool_list

DEFAULT_VERTEX_PROJECT = "poindexter-labs-rl-environment"
_TIMEOUT_SECONDS = 600

# The rawPredict surface rejects any request whose history contains a
# function name outside this alphabet ("Function name was ... but must
# be a-z, A-Z, 0-9, or contain underscores and dashes").
_VALID_TOOL_NAME = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
_MALFORMED_TOOL_NAME = "malformed_tool_name"


class VertexMaaSAdapter(ProviderAdapter):
    name = "maas"

    def __init__(self, model_id=None, region=None):
        if not model_id or "/" not in model_id:
            raise ValueError(
                "maas agent requires --model <publisher>/<model> "
                "(from the Model Garden card's resource name)"
            )
        if not region:
            raise ValueError(
                "maas agent requires --endpoint <region> (e.g. us-central1 "
                "or global; regions are recorded, never assumed)"
            )
        self.model_id = model_id
        self._publisher, self._model = model_id.split("/", 1)
        self._region = region.strip().lower()
        self._project = (
            os.environ.get("VERTEX_MAAS_PROJECT")
            or os.environ.get("GOOGLE_CLOUD_PROJECT")
            or DEFAULT_VERTEX_PROJECT
        )
        self.provider_label = f"vertex-maas:{self._region}"
        self._params = {
            "api_mode": "vertex_rawpredict",
            "sampling": "provider-default",
            "vertex_project": self._project,
        }
        self._token = None
        self._messages = None
        self._tools = None

    def parameters(self):
        return dict(self._params)

    def _url(self):
        if self._region == "global":
            host = "aiplatform.googleapis.com"
        else:
            host = f"{self._region}-aiplatform.googleapis.com"
        return (
            f"https://{host}/v1/projects/{self._project}/locations/"
            f"{self._region}/publishers/{self._publisher}/models/"
            f"{self._model}:rawPredict"
        )

    def _fetch_token(self):
        try:
            out = subprocess.run(
                ["gcloud", "auth", "print-access-token"],
                capture_output=True, text=True, timeout=30, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"could not obtain gcloud access token: {exc}")
        return out.stdout.strip()

    # Not yet (P1-044, probed 2026-09-29). Mistral Medium 3 reads an image
    # correctly, but with one in the conversation and tool_choice "auto" it
    # answers in TEXT ("read_file") instead of calling the tool; "any" makes it
    # call correctly. Switching this route to "any" would change how it behaves
    # on every task, image or not, so the MaaS routes are refused image tasks
    # until that is decided. The image blocks below are built and ready.
    accepts_images = False

    def begin(self, system, tools_spec, opening_user, images=None):
        self._token = self._fetch_token()
        self._tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": desc,
                    "parameters": params or {"type": "object", "properties": {}},
                },
            }
            for name, desc, params in tool_list(tools_spec)
        ]
        self._messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": opening_user if not images else [
                {"type": "text", "text": opening_user},
                *[{"type": "image_url", "image_url": {"url": i.data_url()}} for i in images]]},
        ]

    def _post(self, payload):
        import httpx

        for attempt in (1, 2):
            response = httpx.post(
                self._url(),
                headers={
                    "Authorization": f"Bearer {self._token}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=_TIMEOUT_SECONDS,
            )
            if response.status_code == 401 and attempt == 1:
                self._token = self._fetch_token()
                continue
            if response.status_code >= 400:
                raise RuntimeError(
                    f"rawPredict HTTP {response.status_code}: {response.text[:500]}"
                )
            return response.json()

    def generate(self):
        body = self._post(
            {
                "model": self._model,
                "messages": self._messages,
                "tools": self._tools,
                "tool_choice": "auto",
            }
        )
        choice = (body.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        for tc in message.get("tool_calls") or []:
            # Mistral-family models sometimes emit the arguments JSON in
            # the function NAME field. Stored verbatim, that name is
            # replayed in every later request and the API rejects them
            # all (HTTP 400) — the episode dies without the model ever
            # seeing an error. Store a valid placeholder instead; the
            # runner rejects it as an unknown tool on the normal
            # charged-turn error path (DEC-034 lineage).
            fn = tc.setdefault("function", {})
            if not _VALID_TOOL_NAME.match(fn.get("name") or ""):
                fn["name"] = _MALFORMED_TOOL_NAME
        stored = {k: v for k, v in message.items() if v is not None}
        self._messages.append(stored)
        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            calls.append(
                ToolCall(id=tc.get("id") or f"call_{len(calls)}", name=fn.get("name", ""), args=args)
            )
        usage = body.get("usage") or {}
        # OpenAI-shaped usage: prompt_tokens is the full input, and the
        # publisher reports any cached subset under prompt_tokens_details.
        # Absent for publishers that do not cache, which .get handles.
        prompt_details = usage.get("prompt_tokens_details") or {}
        return Turn(
            calls=calls,
            tokens_in=usage.get("prompt_tokens", 0) or 0,
            tokens_out=usage.get("completion_tokens", 0) or 0,
            cached_tokens_in=prompt_details.get("cached_tokens", 0) or 0,
            finish_reason=choice.get("finish_reason") or "",
            model_version=body.get("model"),
            dropped=(not calls and not (message.get("content") or "").strip()),
        )

    def add_tool_results(self, results):
        for r in results:
            self._messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.id,
                    "name": r.name,
                    "content": r.output if r.error is None else f"ERROR: {r.error}",
                }
            )
        from pdl_bench_eval._vendor.pd_rl.providers.openai_provider import _images_note

        caption, shown = _images_note(results)
        if shown:
            self._messages.append({"role": "user", "content": [
                {"type": "text", "text": caption},
                *[{"type": "image_url", "image_url": {"url": i.data_url()}} for i in shown]]})

    def add_user_text(self, text):
        self._messages.append({"role": "user", "content": text})
