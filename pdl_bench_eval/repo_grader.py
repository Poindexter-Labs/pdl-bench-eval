"""Grade a patch against a repository task exactly as the reference runner does.

Order of operations (each step is a port of the runner's evaluator stage):

1. patch policy — every path the diff touches must match ``allowed_paths`` and
   none ``forbidden_paths``; a violation is ``invalid_agent_fault`` with reward 0.
2. apply the patch to the pristine fixture.
3. purge bytecode; refuse hook-named files the fixture did not ship.
4. in fresh containers, each prefixed by the task's ``setup_command``:
   ``public_command`` (regression suite), ``build_command``, ``hidden_command``.
5. count tests; compare the collected hidden total with the calibrated total.
   Zero collected is an evaluator failure (void). A partial collection voids,
   unless the build step itself failed on the graded tree — then the graded
   code does not import, and every uncollected test counts as failed.
6. score under the policy (``scoring_v0.2`` for every published number).
"""
from __future__ import annotations

import hashlib
import shutil
import time
from pathlib import Path

from . import EVALUATOR_ID
from .bundles import TaskBundle, unpack
from .docker_env import DockerEnv, EvaluatorError, SetupFailed
from .patches import apply_patch, patch_paths, policy_violations
from .pristine import EVALUATOR_ENV_HARDENING, fixture_index, interpreter_hooks, purge_bytecode
from .pytest_counts import pytest_counts
from . import per_test_limit
from .scoring import load_policy, passed, score

DEFAULT_POLICY = "scoring_v0.2"
EVALUATOR_TIMEOUT_SEC = 15 * 60   # evaluator_timeout_minutes in the published profiles
PER_TEST_TIMEOUT_SEC = per_test_limit.DEFAULT_LIMIT_S   # per_test_timeout_seconds in the published profiles


def _record(task: TaskBundle, policy_id: str, env: DockerEnv, patch_text: str, **fields) -> dict:
    base = {
        "evaluator": EVALUATOR_ID,
        "task_id": task.task_id,
        "task_version": task.manifest.get("task_version"),
        "scoring_policy": policy_id,
        "image": env.image,
        "image_id": env.image_id(),
        "patch_sha256": hashlib.sha256(patch_text.encode("utf-8", "replace")).hexdigest(),
        "outcome": None,
        "reward": None,
        "passed": False,
        "hidden_tests_passed": None,
        "hidden_tests_total": None,
        "regression_score": None,
        "build_quality_score": None,
        "notes": [],
    }
    base.update(fields)
    base["passed"] = passed(base["reward"])
    return base


def grade_repository_task(task: TaskBundle, patch_text: str, *, env: DockerEnv, work_root: Path,
                          policy_id: str = DEFAULT_POLICY, calibration: dict | None = None,
                          evaluator_timeout: int = EVALUATOR_TIMEOUT_SEC, per_test_timeout: int | None = PER_TEST_TIMEOUT_SEC,
                          keep_work: bool = False) -> dict:
    policy = load_policy(policy_id)
    work = Path(work_root) / task.task_id
    shutil.rmtree(work, ignore_errors=True)
    repo, hidden = unpack(task, work)
    started = time.time()
    try:
        fixture_paths = fixture_index(repo)
        # 1. patch policy
        try:
            paths = patch_paths(patch_text)
        except ValueError as exc:
            return _record(task, policy_id, env, patch_text, outcome="invalid_submission", reward=0.0,
                           notes=[str(exc)])
        violations = policy_violations(paths, task.patch_policy)
        if violations:
            return _record(task, policy_id, env, patch_text, outcome="invalid_agent_fault", reward=0.0,
                           notes=[f"patch touches files outside patch_policy: {violations}"])
        # 2. apply (an empty submission is a legitimate no-op: it grades the pristine tree)
        patch_file = work / "submission.diff"
        patch_file.write_text(patch_text)
        ok, how = apply_patch(repo, patch_file) if patch_text.strip() else (True, "empty patch: nothing applied")
        if not ok:
            return _record(task, policy_id, env, patch_text, outcome="invalid_submission", reward=0.0,
                           notes=["patch does not apply to the pristine fixture: " + how])
        # 3. pristine tree
        purged = purge_bytecode(repo)
        planted = interpreter_hooks(repo, fixture_paths)
        if planted:
            return _record(task, policy_id, env, patch_text, outcome="invalid_agent_fault", reward=0.0,
                           notes=[f"interpreter hooks planted in the evaluated tree: {planted}"])
        # 4. evaluate (public and hidden suites under the per-test limit; the build without)
        ev = task.evaluation
        setup = task.setup_command
        env_vars = dict(EVALUATOR_ENV_HARDENING)
        eval_hidden, public_env, hidden_env = hidden, {}, dict(env_vars)
        if per_test_timeout:
            eval_hidden = per_test_limit.stage(hidden, base_dir=work)
            public_env = per_test_limit.env(per_test_timeout, {})
            hidden_env = {**env_vars, **per_test_limit.env(per_test_timeout, env_vars)}
        try:
            rc, out = env.evaluate_with_setup(setup, ev["public_command"], repo, eval_hidden, public_env, evaluator_timeout)
            public = pytest_counts(out)
            build_cmd = ev.get("build_command")
            build_ok = build_rc = build_tail = None
            if build_cmd:
                build_rc, bout = env.evaluate_with_setup(setup, build_cmd, repo, eval_hidden, {}, evaluator_timeout)
                build_ok = build_rc == 0
                if not build_ok:
                    build_tail = bout.strip()[-2000:]
            rc, hout = env.evaluate_with_setup(setup, ev["hidden_command"], repo, eval_hidden, hidden_env, evaluator_timeout)
            hidden_counts = pytest_counts(hout)
        except SetupFailed as exc:
            return _record(task, policy_id, env, patch_text, outcome="void_setup_failure", reward=None,
                           notes=[str(exc)])
        except EvaluatorError as exc:
            return _record(task, policy_id, env, patch_text, outcome="void_evaluator_failure", reward=None,
                           notes=[str(exc)])
        # 5. collection checks
        collected_passed = hidden_counts["passed"]
        collected_total = hidden_counts["passed"] + hidden_counts["failed"]
        expected_total = ((calibration or {}).get(task.task_id) or {}).get("hidden_total")
        agent_broke_build = build_cmd is not None and build_rc not in (None, 0, 124, 137)
        notes = []
        if purged["dirs"] or purged["files"]:
            notes.append(f"purged bytecode: {purged['dirs']} dir(s), {purged['files']} file(s)")
        if build_tail:
            notes.append(f"build_command failed (rc={build_rc}): {build_tail[-400:]}")
        if collected_total == 0 and not agent_broke_build:
            return _record(task, policy_id, env, patch_text, outcome="void_evaluator_failure", reward=None,
                           notes=notes + ["zero hidden tests collected: " + hout.strip()[-800:]])
        hidden_passed, hidden_total = collected_passed, collected_total
        if expected_total and collected_total != expected_total:
            if agent_broke_build:
                hidden_passed, hidden_total = min(collected_passed, expected_total), expected_total
                notes.append(f"partial collection after a failed build: {collected_total} counted vs "
                             f"{expected_total} calibrated; uncollected tests count as failed")
            else:
                return _record(task, policy_id, env, patch_text, outcome="void_evaluator_failure", reward=None,
                               notes=notes + [f"partial collection: {collected_total} hidden tests collected vs "
                                              f"{expected_total} calibrated: " + hout.strip()[-800:]])
        if not expected_total:
            notes.append("no calibrated hidden total for this task; partial-collection check skipped")
        # 6. score
        public_total = public["passed"] + public["failed"]
        regression = public["passed"] / public_total if public_total else None
        build = None if build_ok is None else (1.0 if build_ok else 0.0)
        hidden_fraction = hidden_passed / hidden_total if hidden_total else None
        reward = score(policy, {"hidden_test_fraction": hidden_fraction,
                                "regression_test_fraction": regression,
                                "build_quality": build})
        return _record(task, policy_id, env, patch_text, outcome="valid_scored", reward=reward,
                       hidden_tests_passed=hidden_passed, hidden_tests_total=hidden_total,
                       regression_score=None if regression is None else round(regression, 6),
                       build_quality_score=build,
                       public_tests={"passed": public["passed"], "total": public_total},
                       hidden_collected={"passed": collected_passed, "total": collected_total},
                       build_rc=build_rc, patch_paths=sorted(paths), notes=notes,
                       duration_sec=round(time.time() - started, 1))
    finally:
        if per_test_timeout and 'eval_hidden' in locals() and eval_hidden != hidden:
            shutil.rmtree(eval_hidden.parent, ignore_errors=True)
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
