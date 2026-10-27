"""Grade one response to a single-turn Q&A task (STEM, Business Operations, Life Administration).

``objective`` tasks: the response's final answer must match ``FINAL ANSWER``
(deterministic normalised match first, then the comparator prompt).
``rubric`` tasks: the response is graded HIT/MISS per criterion of ``RUBRICS``;
score = 10 * earned / total weight; pass is a score strictly above 8.5.
``hybrid`` tasks: both are recorded; the published pass rule is the rubric one.
An empty response scores 0 without any judge call.
"""
from __future__ import annotations

import json
import math
import re
from importlib import resources

import yaml

from . import EVALUATOR_ID
from .judges import JudgeError, JudgeMember, complete_json, parse_member

RUBRIC_PASS_THRESHOLD = 8.5


def _policy(name: str) -> dict:
    return yaml.safe_load(resources.files("pdl_bench_eval").joinpath("policies", f"{name}.yaml").read_text())


def normalize_direct_answer(value) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        text = text[1:-1].strip()
    wrappers = (("$", "$"), ("\\(", "\\)"), ("\\[", "\\]"))
    changed = True
    while changed:
        changed = False
        for left, right in wrappers:
            if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
                text = text[len(left):-len(right)].strip()
                changed = True
    return text


def direct_answers_match(reference, candidate) -> bool:
    ref, cand = normalize_direct_answer(reference), normalize_direct_answer(candidate)
    return bool(ref) and ref == cand


def match_answer(problem: str, reference: str, response: str, member: JudgeMember | None = None,
                 reasoning: str | None = None) -> dict:
    policy = _policy("qa_answer_match_v1")
    if direct_answers_match(reference, response):
        return {"is_match": True, "method": "deterministic-exact-match", "judge": None}
    member = member or parse_member(policy["reference_judge"])
    user = policy["user_template"].format(problem=problem, solution=reference, model_solution=response)
    payload, version = complete_json(member, policy["system_prompt"].strip(), user, policy["schema"],
                                     reasoning or policy.get("reference_reasoning"))
    verdict = payload.get("is_match")
    if isinstance(verdict, str):
        verdict = verdict.strip().lower() == "true"
    return {"is_match": bool(verdict is True), "method": "llm", "judge": member.label,
            "judge_version": version, "prompt_version": policy["judge_prompt_version"]}


def score_criteria(payload: dict) -> dict:
    criteria = payload.get("criteria") if isinstance(payload, dict) else None
    if not isinstance(criteria, list) or not criteria:
        raise JudgeError("judge returned no criteria")
    total = earned = 0.0
    for item in criteria:
        if not isinstance(item, dict) or not str(item.get("criterion", "")).strip():
            raise JudgeError("judge returned an invalid criterion")
        weight = item.get("weight")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise JudgeError("judge returned an invalid weight")
        if item.get("verdict") not in {"HIT", "MISS"}:
            raise JudgeError("judge returned an invalid verdict")
        total += weight
        if item["verdict"] == "HIT":
            earned += weight
    if total <= 0 or not math.isfinite(total):
        raise JudgeError("rubric must have a positive total weight")
    return {"criteria": criteria, "earned_weight": earned, "total_weight": total,
            "score": round(10 * earned / total, 6), "max_score": 10}


def grade_rubric(problem: str, rubric: str, response: str, member: JudgeMember | None = None,
                 reasoning: str | None = None) -> dict:
    policy = _policy("qa_rubric_sheet_v1")
    if not response.strip():
        return {"score": 0, "max_score": 10, "criteria": [], "empty_response": True, "judge": None,
                "prompt_version": policy["judge_prompt_version"]}
    member = member or parse_member(policy["reference_judge"])
    user = json.dumps({"problem": problem, "rubric": rubric, "response": response}, ensure_ascii=False)
    payload, version = complete_json(member, policy["system_prompt"].strip(), user, policy["schema"],
                                     reasoning or policy.get("reference_reasoning"))
    graded = score_criteria(payload)
    graded.update(judge=member.label, judge_version=version, prompt_version=policy["judge_prompt_version"])
    return graded


def grade_qa(row: dict, response: str, *, match_member: JudgeMember | None = None,
             rubric_member: JudgeMember | None = None, reasoning: str | None = None) -> dict:
    """Grade ``response`` to the task described by ``row`` (a row of the overall config)."""
    grading_type = str(row.get("grading_type") or "").strip()
    record = {"evaluator": EVALUATOR_ID, "task_id": row.get("ID"), "bench": row.get("bench"),
              "grading_type": grading_type, "outcome": "valid_scored", "is_match": None,
              "rubric_score": None, "passed": False, "notes": []}
    if not response.strip():
        record.update(notes=["empty response; no judge call made"], rubric_score=0.0 if grading_type != "objective" else None,
                      is_match=False if grading_type != "rubric" else None)
        return record
    try:
        if grading_type in ("objective", "hybrid"):
            m = match_answer(row["PROMPT"], str(row.get("FINAL ANSWER") or ""), response, match_member, reasoning)
            record["is_match"], record["match"] = m["is_match"], m
        if grading_type in ("rubric", "hybrid"):
            r = grade_rubric(row["PROMPT"], str(row.get("RUBRICS") or ""), response, rubric_member, reasoning)
            record["rubric_score"], record["rubric"] = r["score"], r
    except JudgeError as exc:
        record.update(outcome="void_evaluator_failure", notes=[f"judge failure: {exc}"])
        return record
    if grading_type == "objective":
        record["passed"] = bool(record["is_match"])
    elif grading_type in ("rubric", "hybrid"):
        record["passed"] = record["rubric_score"] is not None and record["rubric_score"] > RUBRIC_PASS_THRESHOLD
    else:
        record.update(outcome="unscored", notes=[f"grading_type {grading_type!r} is not graded by this tool"])
    return record
