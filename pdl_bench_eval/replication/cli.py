"""pdl-bench-replicate: plan | doctor | run | grade | report | build-data.

The same verbs and guarantees as the non-SWE replication packages: ``plan``
and ``doctor`` never call a provider or start an episode; ``run`` is resumable
and never re-solves an attempt that has an answer; ``grade`` only grades saved
episodes; ``report`` never pools configurations.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import PACKAGE_DIR, data, presets
from .records import AttemptStore, attempt_key, fingerprint, grader_hash, implementation_hash, report


def censored(error) -> str:
    text = f"{type(error).__name__}: {error}"
    for key, value in os.environ.items():
        if value and ("KEY" in key or "TOKEN" in key or "SECRET" in key):
            text = text.replace(value, "[redacted]")
    return text[:2000]


def _calibration() -> dict:
    return json.loads((PACKAGE_DIR.parent / "calibration.json").read_text())["tasks"]


def build_plan(args):
    preset = presets.load(args.preset)
    tasks = data.load(args.data, preset["id"])
    if args.family:
        tasks = [t for t in tasks if t["family"] == args.family]
    if args.task:
        tasks = [t for t in tasks if t["id"] in args.task]
        if len(tasks) != len(set(args.task)):
            raise ValueError("one or more requested task ids do not exist in this benchmark")
    if args.limit:
        tasks = tasks[: args.limit]
    configs = presets.resolve_models(preset, args.model, args.attempts, args.agent)
    judges = {}
    for fam in preset["families"].values():
        if fam["kind"] == "qna" and any(t["family"] == "qna" for t in tasks):
            judges["qna"] = presets.judges_for(fam, args.judges)
    return preset, tasks, configs, judges


def evaluation_for(preset, task_row, judges) -> dict:
    fam = presets.family_for(preset, task_row["family"])
    # Benchmark-wide only: anything task-specific here would split one configuration into
    # several report groups. The calibrated hidden total lives in the grade record instead.
    return {"family": {k: fam[k] for k in sorted(presets.FAMILY_KEYS)},
            "judges": [m.label for m in judges.get("qna", [])] if fam["kind"] == "qna" else [],
            "grader": grader_hash()}


def run_attempt(args, preset, task_row, config, judges, store, implementation, *, grade_only=False) -> str:
    from .episode import run_episode
    from .grading import grade_episode, write_record

    fam = presets.family_for(preset, task_row["family"])
    task = task_row["bundle"]
    execution = presets.execution_config(config)
    evaluation = evaluation_for(preset, task_row, judges)
    grade_hash = fingerprint(evaluation)
    lines = []
    for attempt in range(1, config["attempts"] + 1):
        key = attempt_key(task_row["task_hash"], execution, attempt, implementation)
        record = store.load(key, {"id": key, "task_id": task.task_id, "task_hash": task_row["task_hash"],
                                  "config": execution, "attempt": attempt, "implementation": implementation,
                                  "status": "NEW"})
        label = f"{task.task_id} / {config['id']} / {attempt}"
        if record["status"] == "GRADED" and record.get("grade_hash") == grade_hash:
            continue
        if record["status"] in ("SOLVING", "SOLVER_FAILED") and not args.retry_failed:
            lines.append(f"{label}: {record['status']} (use --retry-failed after checking the saved state)")
            continue
        episode_dir = Path(args.results) / "episodes" / key
        work_dir = Path(args.results) / "work" / key
        if "episode" not in record or record["status"] == "SOLVER_FAILED":
            if grade_only:
                continue
            record.update(status="SOLVING", started_at=time.time(), evaluation=evaluation, evaluation_hash=grade_hash,
                          error=None)
            record.pop("episode", None)
            record.pop("grade", None)
            store.save(record)
            try:
                episode = run_episode(task, config=config, family=fam, out_dir=episode_dir, work_dir=work_dir / "episode",
                                      log=lambda m, _l=label: print(f"[{_l}] {m}", flush=True))
            except Exception as exc:  # noqa: BLE001 - recorded, never raised past the attempt
                record.update(status="SOLVER_FAILED", error=censored(exc), updated_at=time.time())
                store.save(record)
                lines.append(f"{label}: SOLVER_FAILED ({record['error'][:120]})")
                continue
            clean = {k: v for k, v in episode.items() if k != "patch_text"}
            record["episode"] = clean
            if str(clean["outcome"] or "").startswith("void"):
                write_record(clean, episode_dir)
                record.update(status="SOLVER_FAILED", error=f"episode {clean['outcome']}", updated_at=time.time())
                store.save(record)
                lines.append(f"{label}: SOLVER_FAILED ({clean['outcome']})")
                continue
            record.update(status="EPISODE_SAVED", updated_at=time.time())
            store.save(record)
        # grading (also reached by `grade` and by a re-grade after an evaluation change)
        episode = dict(record["episode"])
        episode["patch_text"] = Path(episode["patch_uri"]).read_text() if Path(episode["patch_uri"]).exists() else ""
        try:
            if episode["outcome"] == "invalid_agent_fault":
                grade = {"outcome": "invalid_agent_fault", "reward": 0.0, "passed": False,
                         "notes": ["episode ended in an agent fault before grading (see logs.txt)"]}
            else:
                episode["outcome"] = None
                episode, grade = grade_episode(episode, task, fam, dataset_root=Path(args.data), work_root=work_dir / "grade",
                                               members=judges.get("qna") if fam["kind"] == "qna" else None,
                                               calibration=_calibration(), keep_work=args.keep_work)
            clean = write_record(episode, episode_dir)
            if str(grade.get("outcome") or "").startswith("void"):
                # an evaluator or judge fault, not a result: stays ungraded and is retried by `grade`
                record.update(episode=clean, grade=grade, status="GRADING_FAILED",
                              error="; ".join(grade.get("notes") or [grade["outcome"]])[:2000])
            else:
                record.update(episode=clean, grade=grade, status="GRADED", grade_hash=grade_hash, evaluation=evaluation,
                              evaluation_hash=grade_hash, error=None)
        except Exception as exc:  # noqa: BLE001
            record.update(status="GRADING_FAILED", error=censored(exc))
        record["updated_at"] = time.time()
        store.save(record)
        g = record.get("grade") or {}
        lines.append(f"{label}: {record['status']} outcome={g.get('outcome')} reward={g.get('reward')}")
        if not args.keep_work:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)
    return "\n".join(lines)


def cmd_build_data(args):
    from ..bundles import fetch

    root = (Path(args.dataset_dir) if args.dataset_dir else fetch(revision=args.revision)).resolve()
    revision = args.revision
    if revision is None and "snapshots" in root.parts:
        revision = root.parts[root.parts.index("snapshots") + 1]
    manifest = data.build(root, args.preset, Path(args.data), source_revision=revision)
    print(json.dumps(manifest, indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pdl-bench-replicate", description=__doc__)
    ap.add_argument("command", choices=["plan", "doctor", "run", "grade", "report", "build-data"])
    ap.add_argument("--preset", default=os.getenv("BENCHMARK_PRESET", "software-delivery"))
    ap.add_argument("--data", type=Path, default=Path(os.getenv("PDL_DATA", "/data")))
    ap.add_argument("--results", type=Path, default=Path(os.getenv("PDL_RESULTS", "/results")))
    ap.add_argument("--env-file", type=Path)
    ap.add_argument("--model", action="append", help="configuration id from plan; may be repeated")
    ap.add_argument("--task", action="append", help="task id; may be repeated")
    ap.add_argument("--family", choices=["executable", "qna"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--attempts", type=int, choices=range(1, 9))
    ap.add_argument("--agent", choices=["gold", "noop"], help="run a scripted baseline instead of the preset's models")
    ap.add_argument("--judges", help="comma-separated provider:model[@base_url] for the SWEQnA panel (default: the reference panel)")
    ap.add_argument("--parallel", type=int, default=1, help="concurrent episodes (each is its own Docker sandbox)")
    ap.add_argument("--retry-failed", action="store_true", help="explicitly authorise re-solving failed or interrupted attempts")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--dataset-dir", help="build-data: local dataset snapshot (default: fetch from the Hub)")
    ap.add_argument("--revision", help="build-data: dataset revision")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")
    if args.parallel < 1:
        ap.error("--parallel must be positive")
    if args.env_file:
        from dotenv import load_dotenv

        load_dotenv(args.env_file, override=False)
    presets.apply_env_aliases()
    if args.command == "build-data":
        return cmd_build_data(args)
    if args.command == "report":
        print(json.dumps(report(args.results), indent=2))
        return
    preset, tasks, configs, judges = build_plan(args)
    plan = presets.plan_view(preset, tasks, configs, judges, parallel=args.parallel)
    print(json.dumps(plan, indent=2), flush=True)
    if args.command == "plan":
        return
    required = presets.required_keys(configs, judges, solving=args.command != "grade")
    missing = presets.missing_keys(required)
    if missing:
        hint = (" (the question-answering tasks need the judge panel; add --family executable to run the "
                "executable tasks without judge keys)" if any(t["family"] == "qna" for t in tasks) and not args.family else "")
        raise SystemExit("Missing .env settings: " + ", ".join(missing) + hint)
    if args.command == "doctor":
        import subprocess

        from .. import docker_env
        from .._vendor.pd_rl.sandbox import DockerSandbox, IMAGES
        from .agents import MODEL_FAMILIES  # noqa: F401
        for cfg in configs:
            if cfg["provider"] == "openai":
                import openai  # noqa: F401
            elif cfg["provider"] == "anthropic":
                import anthropic  # noqa: F401
            elif cfg["provider"] in ("gemini", "maas"):
                from google import genai  # noqa: F401
        proc = subprocess.run(["docker", "version", "--format", "{{.Server.Version}}"], capture_output=True, text=True)
        if proc.returncode != 0:
            raise SystemExit("docker is not reachable: " + proc.stderr.strip()[-300:])
        for fam in preset["families"].values():
            if fam["kind"] not in {t["family"] for t in tasks}:
                continue
            DockerSandbox(image=fam["sandbox_image"]).ensure_image(PACKAGE_DIR)
            docker_env.DockerEnv.ensure(args.data, "sweqna" if fam["kind"] == "qna" else "swe")
        args.results.mkdir(parents=True, exist_ok=True)
        seen = docker_env.host_path(args.results)       # raises if the host daemon could not see it
        if seen != str(args.results.resolve()):
            print(f"Running inside a container: {args.results} is host path {seen}; evaluator mounts are translated.")
        print(f"Bundle hashes, preset, imports, Docker {proc.stdout.strip()} and required environment variables OK. "
              f"Images: {', '.join(tag for tag, _ in IMAGES.values())}. No provider calls made.")
        return
    args.results.mkdir(parents=True, exist_ok=True)
    store = AttemptStore(args.results)
    implementation = implementation_hash()
    with (args.results / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise SystemExit(f"{args.results} is in use by another run")
        from .records import atomic

        atomic(args.results / "plan.json", {
            **plan, "task_hashes": {t["id"]: t["task_hash"] for t in tasks}, "preset_hash": fingerprint(preset),
            "implementation": implementation,
            "source_manifest": json.loads((PACKAGE_DIR.parent / "_vendor" / "SOURCE_MANIFEST.json").read_text()),
        })
        items = [(task_row, cfg) for cfg in configs for task_row in tasks]
        grade_only = args.command == "grade"

        def work(item):
            task_row, cfg = item
            try:
                return run_attempt(args, preset, task_row, cfg, judges, store, implementation, grade_only=grade_only)
            except Exception as exc:  # noqa: BLE001 - one task must not take the run down
                return f"{task_row['id']} / {cfg['id']}: runner error {censored(exc)}"

        before = {r["id"]: (r["status"], r.get("updated_at")) for r in store.all()}
        if args.parallel > 1:
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                for line in pool.map(work, items):
                    if line:
                        print(line, flush=True)
        else:
            for item in items:
                line = work(item)
                if line:
                    print(line, flush=True)
        after = {r["id"]: (r["status"], r.get("updated_at")) for r in store.all()}
        changed = sum(1 for k, v in after.items() if before.get(k) != v)
        print(f"[{args.command}] {len(items)} task/configuration pairs examined; {changed} attempt record(s) written"
              + (" (select --model/--agent to reach other saved attempts)" if changed == 0 else ""), flush=True)
        print(json.dumps(report(args.results), indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # noqa: BLE001
        print(censored(exc), file=sys.stderr)
        sys.exit(1)
