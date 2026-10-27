"""Grade a frozen episode with the validated graders and persist a schema-valid episode record.

Grader-side code: part of an attempt's evaluation identity (``records.grader_hash``), not of
the identity that produced the episode, so a fix here regrades saved episodes.
"""
from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator

from ..bundles import TaskBundle
from . import assets


def _panel_entry(member: dict) -> dict:
    """The reference record's per-member shape (model, provider, endpoint, model_version, scores, median)."""
    score = member.get("score")
    return {"model": member.get("model"), "provider": member.get("provider"), "endpoint": member.get("endpoint"),
            "model_version": member.get("model_version"),
            "scores": [score] if isinstance(score, (int, float)) else [], "median": score}


def grade_episode(record: dict, task: TaskBundle, family: dict, *, dataset_root: Path, work_root: Path,
                  members=None, calibration: dict | None = None, keep_work: bool = False) -> dict:
    """Grade a frozen episode with the validated graders; returns (updated record, grade record)."""
    from ..docker_env import DockerEnv
    from ..repo_grader import grade_repository_task
    from ..sweqna_grader import grade_sweqna_task

    env = DockerEnv.ensure(dataset_root, task.image_kind)
    patch_text = record.get("patch_text", "")
    if family["kind"] == "qna":
        grade = grade_sweqna_task(task, patch_text=patch_text, env=env, work_root=work_root, members=members,
                                  policy_id=family["scoring"], keep_work=keep_work)
    else:
        grade = grade_repository_task(task, patch_text, env=env, work_root=work_root, policy_id=family["scoring"],
                                      calibration=calibration, keep_work=keep_work)
    judge = grade.get("judge") or {}
    panel = judge.get("judge_panel") or []
    outcome = grade["outcome"]
    if outcome == "invalid_submission":        # cannot arise from a runner-extracted patch; recorded, not hidden
        outcome = "invalid_agent_fault"
    record.update({
        "outcome": outcome,
        "reward_total": grade.get("reward"),
        "evaluator_image_digest": grade.get("image_id") or record["evaluator_image_digest"],
        "hidden_tests_passed": grade.get("hidden_tests_passed"),
        "hidden_tests_total": grade.get("hidden_tests_total"),
        "property_gate_passed": grade.get("property_gate_passed"),
        "property_gate_total": grade.get("property_gate_total"),
        "regression_score": grade.get("regression_score"),
        "build_quality_score": grade.get("build_quality_score"),
        "judge_model": grade.get("judge_model"),
        "judge_prompt_version": judge.get("judge_prompt_version"),
        "judge_panel": [_panel_entry(m) for m in panel] if panel else None,
        "judge_refusals": judge.get("judge_refusals") or None,
        "judge_scores": [m["score"] for m in panel if isinstance(m.get("score"), (int, float))] or None,
    })
    return record, grade


def validate_record(record: dict) -> dict:
    """The persisted episode record: schema-validated against the reference schema."""
    clean = {k: v for k, v in record.items() if k != "patch_text"}
    errors = list(Draft202012Validator(assets.record_schema()).iter_errors(clean))
    if errors:
        raise RuntimeError(f"runner produced an invalid episode record: {errors[0].message}")
    return clean


def write_record(record: dict, out_dir: Path) -> dict:
    clean = validate_record(record)
    (Path(out_dir) / "episode_record.json").write_text(json.dumps(clean, indent=2) + "\n")
    return clean
