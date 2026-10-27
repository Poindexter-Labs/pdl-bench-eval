"""Command-line interface: pdl-bench-eval <command> [options]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import EVALUATOR_ID, __version__


def _root(args) -> Path:
    from .bundles import fetch

    if args.dataset_dir:
        return Path(args.dataset_dir)
    return fetch(revision=args.revision)


def _calibration() -> dict:
    from importlib import resources

    return json.loads(resources.files("pdl_bench_eval").joinpath("calibration.json").read_text())["tasks"]


def _emit(record: dict, out: str | None):
    text = json.dumps(record, indent=2, ensure_ascii=False)
    if out:
        Path(out).write_text(text + "\n")
    print(text)


def cmd_fetch(args):
    root = _root(args)
    print(root)


def cmd_grade_repo(args):
    from .bundles import find_task
    from .docker_env import DockerEnv
    from .repo_grader import grade_repository_task

    root = _root(args)
    task = find_task(root, args.task)
    if task.kind != "tests":
        sys.exit(f"{args.task} is rubric-scored; use grade-sweqna")
    env = DockerEnv.ensure(root, task.image_kind)
    patch_text = Path(args.patch).read_text()
    record = grade_repository_task(task, patch_text, env=env, work_root=Path(args.work_dir),
                                   policy_id=args.policy, calibration=_calibration(),
                                   per_test_timeout=None if args.no_per_test_timeout else 60,
                                   keep_work=args.keep_work)
    _emit(record, args.out)


def cmd_grade_sweqna(args):
    from .bundles import find_task
    from .docker_env import DockerEnv
    from .judges import parse_members
    from .sweqna_grader import grade_sweqna_task

    root = _root(args)
    task = find_task(root, args.task)
    if task.kind != "judge":
        sys.exit(f"{args.task} is test-scored; use grade-repo")
    env = DockerEnv.ensure(root, task.image_kind)
    members = parse_members(args.judges.split(",")) if args.judges else None
    kwargs = {"answer_text": Path(args.answer).read_text()} if args.answer else {"patch_text": Path(args.patch).read_text()}
    record = grade_sweqna_task(task, env=env, work_root=Path(args.work_dir), members=members,
                               policy_id=args.policy, keep_work=args.keep_work, **kwargs)
    _emit(record, args.out)


def cmd_grade_qa(args):
    from .bundles import qa_row
    from .judges import parse_member
    from .qa_grader import grade_qa

    root = _root(args)
    row = qa_row(root, args.task)
    record = grade_qa(row, Path(args.response).read_text(),
                      match_member=parse_member(args.match_judge) if args.match_judge else None,
                      rubric_member=parse_member(args.rubric_judge) if args.rubric_judge else None,
                      reasoning=args.reasoning)
    _emit(record, args.out)


def cmd_calibrate(args):
    """Replay gold and baseline for every executable task; print per-task totals and rewards."""
    from .bundles import load_task
    from .docker_env import DockerEnv
    from .repo_grader import grade_repository_task

    root = _root(args)
    results = {}
    for task_dir in sorted((root / "tasks").iterdir()):
        task = load_task(task_dir)
        if task.kind != "tests" or not (task_dir / "gold.diff").exists():
            continue
        env = DockerEnv.ensure(root, task.image_kind)
        gold = grade_repository_task(task, (task_dir / "gold.diff").read_text(), env=env, work_root=Path(args.work_dir),
                                     policy_id=args.policy, calibration=None)
        results[task.task_id] = {"gold": gold}
        print(f"{task.task_id:36} gold outcome={gold['outcome']} reward={gold['reward']} "
              f"hidden={gold['hidden_tests_passed']}/{gold['hidden_tests_total']}", flush=True)
    _emit(results, args.out)


def cmd_harbor_export(args):
    from .harbor_export import export

    root = _root(args)
    print(json.dumps(export(root, Path(args.out_dir), eval_ref=args.eval_ref)))


def cmd_summarize(args):
    """Aggregate grade records into the published pass-rate statistic.

    pass rate = passes / valid attempts, where a pass is reward > 0.95 (repository and
    SWEQnA tasks) or the Q&A pass flag, and valid attempts exclude void_* outcomes.
    """
    records = []
    for path in args.records:
        p = Path(path)
        files = sorted(p.glob("*.json")) if p.is_dir() else [p]
        for f in files:
            data = json.loads(f.read_text())
            records.extend(data if isinstance(data, list) else [data])
    by_task = {}
    for r in records:
        t = by_task.setdefault(r.get("task_id"), {"attempts": 0, "valid": 0, "passes": 0, "void": 0})
        t["attempts"] += 1
        if str(r.get("outcome", "")).startswith("void"):
            t["void"] += 1
            continue
        t["valid"] += 1
        t["passes"] += 1 if r.get("passed") else 0
    total_valid = sum(t["valid"] for t in by_task.values())
    total_pass = sum(t["passes"] for t in by_task.values())
    summary = {"tasks": by_task, "valid_attempts": total_valid, "passes": total_pass,
               "pass_rate": round(total_pass / total_valid, 4) if total_valid else None}
    _emit(summary, args.out)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdl-bench-eval", description=f"{EVALUATOR_ID}: reference evaluator for PDL-Bench")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--dataset-dir", help="local snapshot of the dataset (default: fetch from the Hub)")
        p.add_argument("--revision", help="dataset revision to fetch (commit, tag or branch)")
        p.add_argument("--out", help="write the JSON record here as well as printing it")

    p = sub.add_parser("fetch", help="download the dataset files the graders need"); common(p); p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("grade-repo", help="grade a patch against an executable repository task"); common(p)
    p.add_argument("--task", required=True); p.add_argument("--patch", required=True, help="unified diff file")
    p.add_argument("--policy", default="scoring_v0.2"); p.add_argument("--work-dir", default=".pdl-bench-work")
    p.add_argument("--no-per-test-timeout", action="store_true"); p.add_argument("--keep-work", action="store_true")
    p.set_defaults(fn=cmd_grade_repo)

    p = sub.add_parser("grade-sweqna", help="grade a written answer for a SWEQnA task"); common(p)
    p.add_argument("--task", required=True)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--answer", help="answers/answer.md content file"); g.add_argument("--patch", help="unified diff adding answers/answer.md")
    p.add_argument("--judges", help="comma-separated provider:model[@base_url]; default: the reference panel")
    p.add_argument("--policy", default="scoring_llm_v0.3"); p.add_argument("--work-dir", default=".pdl-bench-work")
    p.add_argument("--keep-work", action="store_true"); p.set_defaults(fn=cmd_grade_sweqna)

    p = sub.add_parser("grade-qa", help="grade a response to a single-turn Q&A task"); common(p)
    p.add_argument("--task", required=True, help="task ID, e.g. PDS030"); p.add_argument("--response", required=True, help="model response text file")
    p.add_argument("--match-judge", help="provider:model for the answer comparator (default: reference)")
    p.add_argument("--rubric-judge", help="provider:model for the rubric judge (default: reference)")
    p.add_argument("--reasoning", help="reasoning effort for OpenAI-style judges (default: reference)")
    p.set_defaults(fn=cmd_grade_qa)

    p = sub.add_parser("calibrate", help="replay every executable task's gold patch"); common(p)
    p.add_argument("--policy", default="scoring_v0.2"); p.add_argument("--work-dir", default=".pdl-bench-work")
    p.set_defaults(fn=cmd_calibrate)

    p = sub.add_parser("harbor-export", help="emit the repository tasks as a Harbor dataset"); common(p)
    p.add_argument("--out-dir", required=True); p.add_argument("--eval-ref", default="main", help="git ref of pdl-bench-eval the SWEQnA verifier installs")
    p.set_defaults(fn=cmd_harbor_export)

    p = sub.add_parser("summarize", help="pass rate over grade records (files or directories)")
    p.add_argument("records", nargs="+"); p.add_argument("--out"); p.set_defaults(fn=cmd_summarize)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
