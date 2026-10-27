"""Grade a written answer for a SWEQnA task (question answering over a repository).

The reference runner: the agent's changed files are confined by the patch
policy (``answers/**``); the hidden suite is a PROPERTY GATE (file exists, size
bounds, required headings) that must collect but is never scored; the reward is
the judge panel's weighted checkpoint aggregate over ``hidden_tests/rubric.md``,
with the task's issue text as the brief and the changed answer files as the
deliverables.
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from . import EVALUATOR_ID
from .bundles import TaskBundle, unpack
from .checkpoints import CheckpointError, load_checkpoints
from .docker_env import DockerEnv, EvaluatorError, SetupFailed
from .judges import JudgeError, judge_checkpoints, panel_from_policy
from .patches import apply_patch, patch_paths, policy_violations
from .pristine import EVALUATOR_ENV_HARDENING, fixture_index, interpreter_hooks, purge_bytecode
from .pytest_counts import pytest_counts
from .scoring import load_policy, passed, score

DEFAULT_POLICY = "scoring_llm_v0.3"
ANSWER_PATH = "answers/answer.md"
EVALUATOR_TIMEOUT_SEC = 15 * 60


def render_deliverables(repo_dir: Path, rel_paths) -> str:
    files = sorted(p for p in rel_paths if (Path(repo_dir) / p).is_file())
    if not files:
        return "[no deliverable files were produced]"
    blocks = []
    for rel in files:
        path = Path(repo_dir) / rel
        blocks.append(f"--- file: {rel} ({path.stat().st_size} bytes) ---\n{path.read_text(errors='replace')}")
    return "\n\n".join(blocks)


def _record(task, policy_id, env, members, **fields):
    base = {"evaluator": EVALUATOR_ID, "task_id": task.task_id, "task_version": task.manifest.get("task_version"),
            "scoring_policy": policy_id, "image": env.image if env else None,
            "judge_model": "panel:" + ",".join(m.label for m in members) if members else None,
            "outcome": None, "reward": None, "passed": False, "property_gate_passed": None,
            "property_gate_total": None, "notes": []}
    base.update(fields)
    base["passed"] = passed(base["reward"])
    return base


def grade_sweqna_task(task: TaskBundle, *, answer_text: str | None = None, patch_text: str | None = None,
                      env: DockerEnv, work_root: Path, members: list | None = None,
                      policy_id: str = DEFAULT_POLICY, evaluator_timeout: int = EVALUATOR_TIMEOUT_SEC,
                      keep_work: bool = False) -> dict:
    if (answer_text is None) == (patch_text is None):
        raise ValueError("give exactly one of answer_text or patch_text")
    policy = load_policy(policy_id)
    members = members or panel_from_policy(policy)
    work = Path(work_root) / task.task_id
    shutil.rmtree(work, ignore_errors=True)
    repo, hidden = unpack(task, work)
    started = time.time()
    try:
        fixture_paths = fixture_index(repo)
        if patch_text is not None:
            try:
                changed = sorted(patch_paths(patch_text))
            except ValueError as exc:
                return _record(task, policy_id, env, members, outcome="invalid_submission", reward=0.0, notes=[str(exc)])
            violations = policy_violations(changed, task.patch_policy)
            if violations:
                return _record(task, policy_id, env, members, outcome="invalid_agent_fault", reward=0.0,
                               notes=[f"patch touches files outside patch_policy: {violations}"])
            patch_file = work / "submission.diff"
            patch_file.write_text(patch_text)
            ok, how = apply_patch(repo, patch_file) if patch_text.strip() else (True, "empty patch: nothing applied")
            if not ok:
                return _record(task, policy_id, env, members, outcome="invalid_submission", reward=0.0,
                               notes=["patch does not apply: " + how])
        else:
            (repo / ANSWER_PATH).parent.mkdir(parents=True, exist_ok=True)
            (repo / ANSWER_PATH).write_text(answer_text)
            changed = [ANSWER_PATH]
        purge_bytecode(repo)
        planted = interpreter_hooks(repo, fixture_paths)
        if planted:
            return _record(task, policy_id, env, members, outcome="invalid_agent_fault", reward=0.0,
                           notes=[f"interpreter hooks planted in the evaluated tree: {planted}"])
        # property gate (collected, recorded, never scored)
        env_vars = dict(EVALUATOR_ENV_HARDENING)
        try:
            _, out = env.evaluate_with_setup(task.setup_command, task.evaluation["hidden_command"], repo, hidden,
                                             env_vars, evaluator_timeout)
        except SetupFailed as exc:
            return _record(task, policy_id, env, members, outcome="void_setup_failure", notes=[str(exc)])
        except EvaluatorError as exc:
            return _record(task, policy_id, env, members, outcome="void_evaluator_failure", notes=[str(exc)])
        gate = pytest_counts(out)
        gate_total = gate["passed"] + gate["failed"]
        if gate_total == 0:
            return _record(task, policy_id, env, members, outcome="void_evaluator_failure",
                           notes=["zero property assertions collected: " + out.strip()[-800:]])
        # judge
        try:
            items, source = load_checkpoints(hidden)
            deliverables = render_deliverables(repo, changed)
            info = judge_checkpoints(policy, members, task.issue_text, items, deliverables)
        except (CheckpointError, JudgeError) as exc:
            return _record(task, policy_id, env, members, outcome="void_evaluator_failure",
                           property_gate_passed=gate["passed"], property_gate_total=gate_total,
                           notes=[f"judge failure: {exc}"])
        reward = score(policy, {"judge_checkpoints": info["aggregated"]})
        answer_sha = hashlib.sha256((answer_text if answer_text is not None else patch_text).encode("utf-8", "replace")).hexdigest()
        return _record(task, policy_id, env, members, outcome="valid_scored", reward=reward,
                       property_gate_passed=gate["passed"], property_gate_total=gate_total,
                       submission_sha256=answer_sha, checkpoint_source=source, judge=info,
                       duration_sec=round(time.time() - started, 1))
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
