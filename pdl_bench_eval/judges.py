"""LLM judge backends and the checkpoint panel.

A judge member is ``provider:model`` with an optional ``@base_url``:

* ``openai:gpt-5.6``                      OpenAI API (OPENAI_API_KEY, OPENAI_BASE_URL)
* ``openai:gpt-5.6-sol@https://h/v1``     any OpenAI-compatible endpoint
* ``azure:gpt-5.6-sol``                   Azure OpenAI (AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY)
* ``anthropic:claude-opus-5``             Anthropic API (ANTHROPIC_API_KEY)
* ``google:gemini-3.1-pro-preview``       Gemini (GEMINI_API_KEY) or Vertex AI when
                                          GOOGLE_CLOUD_PROJECT is set (Application Default Credentials)

The reference panel for every published SWEQnA number was
``google:gemini-3.1-pro-preview, anthropic:claude-opus-5, openai:gpt-5.6-sol``,
each called once; an item's score is the median of the members' binary votes.
"""
from __future__ import annotations

import json
import os
import statistics
import time
from dataclasses import dataclass

from .checkpoints import render_for_judge, weighted_score

MIN_PANEL_MEMBERS = 2
# Output budget for a Claude member (reference: 8192 -> 32768 on 2026-08-29; SWEQnA rubrics
# are 24-40 binary criteria and claude-opus-5 spent 8192 reasoning without a verdict).
ANTHROPIC_JUDGE_MAX_TOKENS = int(os.environ.get("ANTHROPIC_JUDGE_MAX_TOKENS", "32768"))
# Pause between a failed judgment and its retry (reference: back-to-back retries let one
# provider hiccup void an episode).
JUDGE_RETRY_BACKOFF_S = 3.0
CHECKPOINT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "pass": {"type": "boolean"}, "evidence": {"type": "string"}},
                "required": ["id", "pass"],
            },
        }
    },
    "required": ["results"],
}


class JudgeError(RuntimeError):
    """A member could not deliver a verdict; the grade is void, never 0."""


class JudgeRefusal(JudgeError):
    """A member declined to judge the content: categorical, so the panel drops it (reference rule)."""


@dataclass
class JudgeMember:
    provider: str
    model: str
    base_url: str | None = None
    k: int = 1

    @property
    def label(self) -> str:
        return f"{self.provider}:{self.model}"


def parse_member(spec: str) -> JudgeMember:
    spec = spec.strip()
    if ":" not in spec:
        raise ValueError(f"judge spec {spec!r} must be provider:model[@base_url]")
    provider, rest = spec.split(":", 1)
    base_url = None
    if "@" in rest:
        rest, base_url = rest.split("@", 1)
    provider = provider.lower()
    if provider not in ("openai", "azure", "anthropic", "google"):
        raise ValueError(f"unknown judge provider {provider!r}")
    return JudgeMember(provider=provider, model=rest, base_url=base_url)


def parse_members(specs) -> list:
    return [parse_member(s) for s in specs]


def panel_from_policy(policy: dict) -> list:
    return [JudgeMember(provider=m["provider"], model=m["model"], k=int(policy["judge"].get("k", 1)))
            for m in policy["judge"]["panel"]]


# --- completions ------------------------------------------------------------------

def _complete_openai(member: JudgeMember, system: str, user: str, schema: dict | None,
                     reasoning: str | None = None) -> tuple:
    from openai import OpenAI

    client = OpenAI(base_url=member.base_url or os.environ.get("OPENAI_BASE_URL") or None,
                    api_key=os.environ.get("OPENAI_API_KEY"))
    return _chat_json(client, member, system, user, schema, reasoning)


def _complete_azure(member: JudgeMember, system: str, user: str, schema: dict | None,
                    reasoning: str | None = None) -> tuple:
    from openai import AzureOpenAI

    client = AzureOpenAI(azure_endpoint=member.base_url or os.environ["AZURE_OPENAI_ENDPOINT"],
                         api_key=os.environ["AZURE_OPENAI_API_KEY"],
                         api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21"))
    return _chat_json(client, member, system, user, schema, reasoning)


def _chat_json(client, member, system, user, schema, reasoning):
    kwargs = {"model": member.model,
              "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if schema is CHECKPOINT_RESPONSE_SCHEMA:
        # The reference checkpoint panel asks for a JSON object at temperature 1.0.
        kwargs["response_format"] = {"type": "json_object"}
        kwargs["temperature"] = 1.0
    elif schema is not None:
        # The reference Q&A graders use a strict JSON schema.
        kwargs["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "verdict", "schema": schema, "strict": True}}
    if reasoning:
        kwargs["reasoning_effort"] = reasoning
    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as exc:  # models that reject response_format or reasoning_effort
        if schema is None and not reasoning:
            raise
        kwargs.pop("response_format", None)
        kwargs.pop("reasoning_effort", None)
        kwargs.pop("temperature", None)
        try:
            resp = client.chat.completions.create(**kwargs)
        except Exception:
            raise JudgeError(f"{member.label}: {type(exc).__name__}: {exc}") from None
    text = resp.choices[0].message.content or ""
    return text, getattr(resp, "model", None) or member.model


def _complete_anthropic(member: JudgeMember, system: str, user: str, schema: dict | None,
                        reasoning: str | None = None) -> tuple:
    import anthropic

    client = anthropic.Anthropic(base_url=member.base_url or None)
    try:
        # Streamed, as the reference does: the SDK refuses a non-streaming request whose
        # max_tokens implies a possible >10-minute generation. No sampling parameter: the
        # pinned SDK (1.x) exposes none on stream(), and the provider default is the
        # temperature 1.0 the reference set explicitly.
        with client.messages.stream(model=member.model, max_tokens=ANTHROPIC_JUDGE_MAX_TOKENS,
                                    system=system, messages=[{"role": "user", "content": user}]) as stream:
            resp = stream.get_final_message()
    except Exception as exc:
        raise JudgeError(f"{member.label}: {type(exc).__name__}: {exc}") from None
    blocks = list(resp.content or [])
    text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", None) == "text")
    if not text.strip():
        stop_reason = getattr(resp, "stop_reason", None)
        kinds = [getattr(b, "type", "?") for b in blocks] or ["(no content blocks)"]
        advice = {
            "max_tokens": "the budget was spent before the verdict; raise ANTHROPIC_JUDGE_MAX_TOKENS.",
            "refusal": ("the model declined to answer; this is a property of the content, so the panel "
                        "drops this member and scores on the rest."),
        }.get(stop_reason, "no text block was returned and the stop reason is not one this harness recognises.")
        problem = (f"{member.label} returned no text to parse (stop_reason={stop_reason!r}, blocks={kinds}, "
                   f"max_tokens={ANTHROPIC_JUDGE_MAX_TOKENS}). {advice}")
        raise (JudgeRefusal if stop_reason == "refusal" else JudgeError)(problem)
    return text, getattr(resp, "model", None) or member.model


def _complete_google(member: JudgeMember, system: str, user: str, schema: dict | None,
                     reasoning: str | None = None) -> tuple:
    from google import genai
    from google.genai import types

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        client = genai.Client(vertexai=True, project=project,
                              location=os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
    else:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    config = types.GenerateContentConfig(system_instruction=system, response_mime_type="application/json",
                                         response_schema=schema) if schema else \
        types.GenerateContentConfig(system_instruction=system)
    try:
        resp = client.models.generate_content(model=member.model, contents=user, config=config)
    except Exception as exc:
        raise JudgeError(f"{member.label}: {type(exc).__name__}: {exc}") from None
    return resp.text or "", getattr(resp, "model_version", None) or member.model


BACKENDS = {"openai": _complete_openai, "azure": _complete_azure,
            "anthropic": _complete_anthropic, "google": _complete_google}


def complete(member: JudgeMember, system: str, user: str, schema: dict | None = None,
             reasoning: str | None = None) -> tuple:
    """(text, model_version) from one judge call."""
    return BACKENDS[member.provider](member, system, user, schema, reasoning)


def parse_json_reply(text: str, context: str = "") -> object:
    body = (text or "").strip()
    if body.startswith("```"):
        body = body.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        return json.loads(body)
    except (TypeError, ValueError) as exc:
        shown = body[:300] if body else "(empty reply)"
        raise JudgeError(f"could not parse a JSON verdict{(' [' + context + ']') if context else ''}: "
                         f"{type(exc).__name__}: {exc}; reply was {shown!r}") from None


def parse_checkpoint_verdicts(text: str, expected_ids, context: str = "") -> tuple:
    """(verdicts {id: bool}, evidence {id: str}) from one member's reply; every id must be decided."""
    payload = parse_json_reply(text, context)
    results = payload.get("results") if isinstance(payload, dict) else payload
    if isinstance(results, dict):
        results = [{"id": k, **(v if isinstance(v, dict) else {"pass": v})} for k, v in results.items()]
    if not isinstance(results, list):
        raise JudgeError(f"checkpoint reply is not a results list [{context}]")
    verdicts, evidence = {}, {}
    expected = set(expected_ids)
    for entry in results:
        if not isinstance(entry, dict):
            continue
        rid = str(entry.get("id", "")).strip()
        if rid not in expected:
            bare = rid.strip("[]()<>`'\" ").strip()
            if bare in expected:
                rid = bare
        verdict = entry.get("pass")
        if isinstance(verdict, str):
            verdict = verdict.strip().lower() in ("true", "pass", "passed", "yes", "met", "1")
        verdicts[rid] = bool(verdict)
        evidence[rid] = str(entry.get("evidence", "") or "")[:400]
    missing = [i for i in expected_ids if i not in verdicts]
    if missing:
        raise JudgeError(f"judge omitted {len(missing)} of {len(expected_ids)} checkpoint(s) [{context}]: {missing[:6]}")
    return verdicts, evidence


def judge_checkpoints(policy: dict, members: list, brief: str, items: list, deliverables_text: str) -> dict:
    """Panel verdicts per checkpoint under ``policy`` (scoring_llm_v0.3 semantics).

    Reference panel discipline: every member is called; a member that cannot deliver a
    verdict set within its retry budget voids the grade; a member that REFUSES is dropped
    and recorded, down to the MIN_PANEL_MEMBERS floor.
    """
    judge_cfg = policy["judge"]
    llm_items = [it for it in items if it.method == "llm"]
    ids = [it.id for it in llm_items]
    system_prompt = judge_cfg["system_prompt"]
    user_text = judge_cfg["user_template"].format(brief=brief.strip(), checkpoints=render_for_judge(items),
                                                  deliverables=deliverables_text)
    votes = {rid: [] for rid in ids}
    judgments, per_member, refusals = [], [], []
    for member in members:
        k = member.k
        got, failures = [], []
        for attempt in range(k * 2):            # one retry budget per judgment
            if len(got) == k:
                break
            if attempt and failures:
                time.sleep(JUDGE_RETRY_BACKOFF_S)
            try:
                text, version = complete(member, system_prompt, user_text, CHECKPOINT_RESPONSE_SCHEMA)
                verdicts, evidence = parse_checkpoint_verdicts(text, ids, member.label)
                got.append({"judge_model": member.model, "provider": member.provider,
                            "model_version": version, "verdicts": verdicts, "evidence": evidence})
            except JudgeRefusal as exc:
                failures.append(f"JudgeRefusal: {exc}")
                break
            except JudgeError as exc:
                failures.append(str(exc))
        if len(got) < k:
            if any(f.startswith("JudgeRefusal:") for f in failures):
                refusals.append({"model": member.model, "provider": member.provider, "reason": failures[-1][:400]})
                continue
            raise JudgeError(f"judge {member.label} produced {len(got)}/{k} verdict set(s); "
                             f"failures: {failures[-3:]}")
        member_item_scores = {}
        for rid in ids:
            member_votes = [1.0 if j["verdicts"][rid] else 0.0 for j in got]
            votes[rid].extend(member_votes)
            member_item_scores[rid] = sum(member_votes) / len(member_votes)
        member_total, _ = weighted_score(items, member_item_scores)
        judgments.extend(got)
        per_member.append({"model": member.model, "provider": member.provider,
                           "model_version": got[0]["model_version"], "score": round(member_total, 6)})
    if len(per_member) < MIN_PANEL_MEMBERS:
        raise JudgeError(f"panel has {len(per_member)} member(s)"
                         + (f" after {len(refusals)} refusal(s)" if refusals else "")
                         + f"; the reference policy needs at least {MIN_PANEL_MEMBERS}")
    item_scores = {rid: round(statistics.median(votes[rid]), 6) for rid in ids}
    aggregated, gate_failed = weighted_score(items, item_scores)
    detail = [{"id": it.id, "weight": it.weight, "question": it.question, "score": item_scores[it.id],
               "votes": votes[it.id], "evidence": {j["judge_model"]: j["evidence"].get(it.id, "") for j in judgments}}
              for it in llm_items]
    return {
        "judge_model": "panel:" + ",".join(m.label for m in members),
        "judge_prompt_version": judge_cfg["judge_prompt_version"],
        "judge_panel": per_member,
        "judge_refusals": refusals or None,
        "warnings": [f"judge {r['model']} refused to score this deliverable and was dropped; scored on "
                     f"{len(per_member)} of {len(members)} members" for r in refusals],
        "aggregated": aggregated,
        "gate_failed": gate_failed,
        "all_passed": all(item_scores[it.id] >= 0.999 for it in llm_items),
        "disputed": [it.id for it in llm_items if 0.0 < item_scores[it.id] < 1.0],
        "items": detail,
        "user_text_chars": len(user_text),
    }


def complete_json(member: JudgeMember, system: str, user: str, schema: dict, reasoning: str | None = None,
                  retries: int = 2) -> tuple:
    """One structured verdict: (payload, model_version). Retries malformed replies."""
    failures = []
    for _ in range(retries + 1):
        try:
            text, version = complete(member, system, user, schema, reasoning)
            payload = parse_json_reply(text, member.label)
            if not isinstance(payload, dict):
                raise JudgeError(f"{member.label}: reply is not a JSON object")
            return payload, version
        except JudgeError as exc:
            failures.append(str(exc))
    raise JudgeError(f"{member.label}: no usable verdict after {retries + 1} attempts: {failures[-1]}")
