"""Attempt records and the report, in the format of the non-SWE replication packages.

One JSON record per attempt under ``results/attempts/<key>.json``, where the key
is a hash of the task, the execution configuration, the attempt number and
the implementation. Statuses: ``NEW`` → ``SOLVING`` → ``EPISODE_SAVED`` (the
agent finished and its patch is recorded) → ``GRADED``; ``SOLVER_FAILED`` and
``GRADING_FAILED`` are technical failures that stay ungraded until an explicit
retry. Nothing is ever pooled across configurations, implementations or
evaluation settings.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import defaultdict
from pathlib import Path

from .. import EVALUATOR_ID, __version__

PACKAGE = Path(__file__).resolve().parent.parent
STATUSES = ("NEW", "SOLVING", "SOLVER_FAILED", "EPISODE_SAVED", "GRADING_FAILED", "GRADED")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value) -> str:
    return digest(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode())


def atomic(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = value if isinstance(value, bytes) else (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("xb") as out:
        out.write(data)
        out.flush()
        os.fsync(out.fileno())
    tmp.replace(path)


SOLVER_PARTS = ("_vendor", "replication/episode.py", "replication/agents.py", "replication/assets.py",
                "replication/sandbox", "bundles.py")
GRADER_PARTS = ("replication/grading.py", "repo_grader.py", "sweqna_grader.py", "judges.py", "scoring.py", "checkpoints.py", "rubric.py",
                "patches.py", "pristine.py", "pytest_counts.py", "per_test_limit.py", "docker_env.py",
                "policies", "calibration.json", "eval_plugins")


def _files_under(parts) -> dict:
    out = {}
    for part in parts:
        root = PACKAGE / part
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        for p in paths:
            if "__pycache__" in p.parts or p.suffix not in (".py", ".yaml", ".json", ""):
                continue
            out[str(p.relative_to(PACKAGE))] = digest(p.read_bytes())
    return out


def implementation_hash() -> str:
    """Identity of the code that PRODUCED an attempt: the vendored agent loop, the episode runner,
    the sandbox images and the bundle reader. Grader code is part of the evaluation identity
    instead (``grader_hash``), so a grading fix regrades saved episodes rather than orphaning them."""
    return fingerprint({"evaluator": EVALUATOR_ID, "version": __version__, "files": _files_under(SOLVER_PARTS)})


def grader_hash() -> str:
    """Identity of the grading code: part of every attempt's evaluation hash."""
    return fingerprint({"evaluator": EVALUATOR_ID, "files": _files_under(GRADER_PARTS)})


def attempt_key(task_hash: str, execution: dict, attempt: int, implementation: str) -> str:
    return fingerprint({"task_hash": task_hash, "config": execution, "attempt": attempt,
                        "implementation": implementation})


class AttemptStore:
    def __init__(self, results: Path):
        self.results = Path(results)
        self.dir = self.results / "attempts"

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def load(self, key: str, default: dict) -> dict:
        p = self.path(key)
        return json.loads(p.read_text()) if p.exists() else dict(default)

    def save(self, record: dict) -> None:
        atomic(self.path(record["id"]), record)

    def all(self):
        for p in sorted(self.dir.glob("*.json")):
            yield json.loads(p.read_text())


def _passed(record: dict) -> bool:
    grade = record.get("grade") or {}
    return bool(grade.get("passed"))


def report(results: Path) -> dict:
    """Both statistics, never pooled across configuration, implementation or evaluation.

    ``pass_rate`` is the published PDL-Bench statistic: passes / valid attempts,
    where valid attempts are graded attempts whose outcome is not ``void_*``
    (``invalid_*`` counts as a failed attempt). ``observed_pass_at_k`` is the
    non-SWE packages' statistic: over tasks whose first k attempts are all
    graded, the fraction with at least one pass.
    """
    store = AttemptStore(results)
    groups = defaultdict(list)
    for row in store.all():
        key = fingerprint({"config": row["config"], "implementation": row["implementation"],
                           "evaluation_hash": row.get("evaluation_hash")})
        groups[key].append(row)
    out = []
    for key, rows in groups.items():
        graded = [r for r in rows if r["status"] == "GRADED"]
        void = [r for r in graded if str((r.get("grade") or {}).get("outcome", "")).startswith("void")]
        valid = [r for r in graded if r not in void]
        scored = [r for r in valid if (r.get("grade") or {}).get("outcome") == "valid_scored"]
        invalid = [r for r in valid if r not in scored]
        passes = [r for r in valid if _passed(r)]
        rewards = [r["grade"]["reward"] for r in scored if r["grade"].get("reward") is not None]
        by_task = defaultdict(list)
        for r in rows:
            by_task[r["task_hash"]].append(r)
        by_k = {}
        for k in range(1, max(r["attempt"] for r in rows) + 1):
            eligible = [rs for rs in by_task.values()
                        if {r["attempt"] for r in rs if r["status"] == "GRADED"} >= set(range(1, k + 1))]
            passing = sum(any(_passed(r) for r in rs if r["attempt"] <= k) for rs in eligible)
            by_k[str(k)] = {"complete_tasks": len(eligible), "tasks_with_at_least_one_pass": passing,
                            "fraction": passing / len(eligible) if eligible else None}
        terminations = defaultdict(int)
        for r in rows:
            terminations[str((r.get("episode") or {}).get("termination_reason"))] += 1
        out.append({
            "configuration_hash": key,
            "model": rows[0]["config"].get("id"),
            "config": rows[0]["config"],
            "attempts_recorded": len(rows),
            "graded": len(graded),
            "ungraded": len(rows) - len(graded),
            "void": len(void),
            "valid_attempts": len(valid),
            "invalid_attempts": len(invalid),
            "passes": len(passes),
            "pass_rate": len(passes) / len(valid) if valid else None,
            "mean_reward_valid_scored": sum(rewards) / len(rewards) if rewards else None,
            "termination_reasons": dict(terminations),
            "observed_pass_at_k": by_k,
        })
    value = {
        "groups": out,
        "statistics": {
            "pass_rate": "passes / valid attempts; a pass is reward > 0.95; void_* outcomes are excluded, invalid_* count as failures (the published PDL-Bench statistic)",
            "observed_pass_at_k": "over tasks whose first k attempts are all graded, the fraction with at least one pass (the non-SWE packages' statistic; observed rate, not an unbiased estimator)",
        },
        "note": "Technical failures (SOLVER_FAILED, GRADING_FAILED) are ungraded and never counted as model failures.",
    }
    atomic(Path(results) / "report.json", value)
    return value
