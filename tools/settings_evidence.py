#!/usr/bin/env python3
"""Generate settings-evidence/<benchmark>.json for the SWE replication packages.

Reads the episode table in BigQuery (every recorded episode carries its prompt
version, tool schema, scoring policy, evaluator version, model parameters,
image digest and sandbox backend) and writes, per benchmark, the observed
configurations with counts and date ranges. Maintainer tool: needs the `bq`
CLI and read access to the table; the output is what ships, not the query.

Usage: python tools/settings_evidence.py --dataset-dir .dataset --out replication/settings-evidence
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import pandas as pd

TABLE = "poindexter-labs-rl-environment.pdl_rl.episodes"
PROFILES = ("python_standard_2m_v0.1", "python_qa_2m_v0.1")
BENCHES = {"software-delivery": "software_delivery", "cyber-operations": "cyber_operations"}
# The console jobs behind the published report (docs/REPRODUCTION.md).
PUBLISHED_JOBS = {
    "software-delivery": {
        "executable": ["rlj-20260911-182105-d16163", "rlj-20260925-204349-0090bb", "rlj-20261013-232626-6fc6bd"],
        "qna": ["rlj-20261004-175352-1f4a04", "rlj-20261005-005743-894c05", "rlj-20261014-113935-1586f8"],
    },
    "cyber-operations": {
        "executable": ["rlj-20260911-181246-68ed7c", "rlj-20260925-210731-44bc8a", "rlj-20261013-232626-6fc6bd"],
    },
}
SQL = """
SELECT task_id, benchmark_profile_id, prompt_version, tool_schema_version, scoring_version, evaluator_version,
       model_provider, model_name, TO_JSON_STRING(model_parameters) AS model_parameters,
       sandbox_backend, environment_image_digest,
       COUNT(*) AS episodes,
       COUNTIF(outcome = 'valid_scored') AS valid_scored,
       COUNTIF(outcome LIKE 'invalid%') AS invalid,
       COUNTIF(outcome LIKE 'void%') AS void,
       COUNTIF(outcome = 'valid_scored' AND reward_total > 0.95) AS passes,
       FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', MIN(start_time)) AS first_start,
       FORMAT_TIMESTAMP('%Y-%m-%dT%H:%M:%SZ', MAX(end_time)) AS last_end
FROM `{table}`
WHERE task_id IN ({ids}) AND benchmark_profile_id IN ({profiles}) AND (condition IS NULL OR condition = '')
GROUP BY 1,2,3,4,5,6,7,8,9,10,11
ORDER BY task_id, model_name, first_start
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.dataset_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ids = {b: sorted(pd.read_parquet(root / f"benchmark_dataset_{cfg}.parquet")["ID"].tolist()) for b, cfg in BENCHES.items()}
    all_ids = sorted(set().union(*ids.values()))
    sql = SQL.format(table=TABLE, ids=",".join(f"'{i}'" for i in all_ids), profiles=",".join(f"'{p}'" for p in PROFILES))
    raw = subprocess.run(["bq", "query", "--project_id=poindexter-labs-rl-environment", "--use_legacy_sql=false",
                          "--format=json", "--max_rows=100000"], input=sql, capture_output=True, text=True, check=True).stdout
    rows = json.loads(raw)
    for bench, members in ids.items():
        mine = [r for r in rows if r["task_id"] in members]
        grouped = {}
        for r in mine:
            key = json.dumps({k: r[k] for k in ("benchmark_profile_id", "prompt_version", "tool_schema_version", "scoring_version",
                                                "evaluator_version", "model_provider", "model_name", "model_parameters",
                                                "sandbox_backend", "environment_image_digest")}, sort_keys=True)
            g = grouped.setdefault(key, {"config": json.loads(key), "episodes": 0, "valid_scored": 0, "invalid": 0, "void": 0,
                                         "passes": 0, "tasks": set(), "first_start": r["first_start"], "last_end": r["last_end"]})
            for f in ("episodes", "valid_scored", "invalid", "void", "passes"):
                g[f] += int(r[f])
            g["tasks"].add(r["task_id"])
            g["first_start"] = min(g["first_start"], r["first_start"])
            g["last_end"] = max(g["last_end"], r["last_end"])
        observed = []
        for g in grouped.values():
            g["config"]["model_parameters"] = json.loads(g["config"]["model_parameters"] or "{}")
            g["tasks"] = len(g["tasks"])
            observed.append(g)
        observed.sort(key=lambda g: (g["config"]["benchmark_profile_id"], g["config"]["model_name"], g["first_start"]))
        record = {
            "benchmark": bench,
            "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": f"BigQuery `{TABLE}`; every episode record carries these fields, so counts are exact, not reconstructed.",
            "tasks": len(members),
            "published_jobs": PUBLISHED_JOBS[bench],
            "basis": ("The default preset reproduces the configurations behind the published report: profile, prompt, tool schema and "
                      "scoring policy as recorded on those jobs' episodes. Earlier configurations (other scoring policies, image "
                      "digests or sandboxes) are listed here as evidence and are not pooled with the published numbers."),
            "warning": ("Counts include every episode recorded under the two published profiles, including probes and re-runs that "
                        "are not part of the published report. The published report is the subset under the published jobs."),
            "query": " ".join(sql.split()),
            "observed_runs": observed,
        }
        (out / f"{bench}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(bench, "configs:", len(observed), "episodes:", sum(g["episodes"] for g in observed))


if __name__ == "__main__":
    main()
