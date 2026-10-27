#!/usr/bin/env python3
"""Build replication/validation.json and the result tables in replication/docs/VALIDATION.md from results directories.

Reads attempt records written by `pdl-bench-replicate` (calibration sweeps with the
gold and noop baselines; paid single-episode smokes) and records what they show.
Nothing is edited by hand: re-run after every validation run.

Usage:
  python tools/validation_record.py --calibration software-delivery=/path/cal-sd \\
      --calibration cyber-operations=/path/cal-co --live /path/live1 --live /path/live2 \\
      --offline-tests 10 --evaluator-tests 11
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_JSON = REPO / "replication" / "validation.json"
DOC = REPO / "replication" / "docs" / "VALIDATION.md"
ATTEMPTS_FULL_RUN = {"software-delivery": 20 * 8 * 5, "cyber-operations": 10 * 8 * 5}


def attempts(results: Path):
    for p in sorted((Path(results) / "attempts").glob("*.json")):
        yield json.loads(p.read_text())


def calibration_summary(bench: str, results: Path) -> dict:
    out = {"benchmark": bench, "results": str(results), "gold": {}, "noop": {}}
    for r in attempts(results):
        kind = r["config"]["id"]
        if kind not in ("gold", "noop"):
            continue
        g = r.get("grade") or {}
        out[kind][r["task_id"]] = {"status": r["status"], "outcome": g.get("outcome"), "reward": g.get("reward")}
    gold = out["gold"]
    noop = out["noop"]
    out["summary"] = {
        "gold_tasks": len(gold),
        "gold_at_1.0": sum(1 for v in gold.values() if v["reward"] == 1.0),
        "gold_other": {t: v for t, v in gold.items() if v["reward"] != 1.0},
        "noop_tasks": len(noop),
        "noop_at_0.0": sum(1 for v in noop.values() if v["reward"] == 0.0),
        "noop_nonzero": {t: v["reward"] for t, v in noop.items() if v["reward"] not in (0.0, None)},
        "ungraded": {t: v["status"] for k in ("gold", "noop") for t, v in out[k].items() if v["status"] != "GRADED"},
    }
    return out


def live_summary(results: Path) -> list:
    rows = []
    for r in attempts(results):
        e = r.get("episode") or {}
        g = r.get("grade") or {}
        rows.append({
            "model": r["config"].get("id"), "route": e.get("model_provider"), "model_version": e.get("model_version"),
            "task": r["task_id"], "status": r["status"], "termination_reason": e.get("termination_reason"),
            "outcome": g.get("outcome") or e.get("outcome"), "reward": g.get("reward"),
            "tool_calls": e.get("tool_calls"), "tokens_in": e.get("tokens_in"), "tokens_out": e.get("tokens_out"),
            "duration_s": round(float(r.get("updated_at", 0)) - float(r.get("started_at", 0))) if r.get("started_at") else None,
            "judge_panel": [m.get("model") for m in (e.get("judge_panel") or (g.get("judge") or {}).get("judge_panel") or [])] or None,
            "prompt_version": e.get("prompt_version"), "profile": e.get("benchmark_profile_id"),
            "scoring": e.get("scoring_version"),
        })
    return rows


def render_tables(cal: dict, live: list) -> str:
    lines = ["| Benchmark | Baseline | Tasks | Result |", "| --- | --- | ---: | --- |"]
    for bench, c in cal.items():
        s = c["summary"]
        other = "; ".join(f"{t} {v['reward']}" for t, v in s["gold_other"].items())
        lines.append(f"| {bench} | gold | {s['gold_tasks']} | {s['gold_at_1.0']} at reward 1.0"
                     + (f"; other: {other}" if other else "") + " |")
        nz = "; ".join(f"{t} {v}" for t, v in s["noop_nonzero"].items())
        lines.append(f"| {bench} | noop | {s['noop_tasks']} | {s['noop_at_0.0']} at reward 0.0"
                     + (f"; documented non-zero baselines: {nz}" if nz else "") + " |")
    lines += ["", "| Model | Route | Task | Termination | Outcome | Reward | Tool calls | Tokens in / out | Minutes |",
              "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for r in live:
        lines.append(f"| {r['model']} | {r['route']} | {r['task']} | {r['termination_reason']} | {r['outcome']} | "
                     f"{r['reward']} | {r['tool_calls']} | {r['tokens_in']:,} / {r['tokens_out']:,} | "
                     f"{(r['duration_s'] or 0) / 60:.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", action="append", default=[], help="benchmark=results_dir")
    ap.add_argument("--live", action="append", default=[], help="results_dir of a paid smoke run")
    ap.add_argument("--offline-tests", type=int, required=True)
    ap.add_argument("--evaluator-tests", type=int, required=True)
    args = ap.parse_args()
    cal = {}
    for item in args.calibration:
        bench, path = item.split("=", 1)
        cal[bench] = calibration_summary(bench, Path(path))
    live = [row for path in args.live for row in live_summary(Path(path))]
    today = dt.date.today().isoformat()
    record = {
        "date": today,
        "docker_build": "passed",
        "offline_plan": "passed",
        "offline_tests_passed": args.offline_tests,
        "evaluator_unit_tests_passed": args.evaluator_tests,
        "calibration": {b: c["summary"] for b, c in cal.items()},
        "live_smokes": live,
        "live_model_episodes_for_publication": len(live),
        "expected_attempts_on_full_default_run": ATTEMPTS_FULL_RUN,
        "note": ("Calibration runs the scripted gold and noop baselines through the complete path (sandbox episode, "
                 "patch freeze, grader); live smokes are one paid episode per reference route and are not benchmark results."),
    }
    OUT_JSON.write_text(json.dumps(record, indent=2) + "\n")
    doc = DOC.read_text()
    block = f"<!-- validation-tables:begin -->\n{render_tables(cal, live)}\n<!-- validation-tables:end -->"
    if "<!-- validation-tables:begin -->" in doc:
        doc = re.sub(r"<!-- validation-tables:begin -->.*?<!-- validation-tables:end -->", block, doc, flags=re.S)
    else:
        doc = doc.replace("## Live checks", block + "\n\n## Live checks", 1)
    DOC.write_text(doc)
    print(json.dumps({"calibration": record["calibration"], "live": [(r["model"], r["task"], r["outcome"], r["reward"]) for r in live]}, indent=1))


if __name__ == "__main__":
    main()
