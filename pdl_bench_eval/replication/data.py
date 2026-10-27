"""The package's local copy of the tasks: ``data/tasks.jsonl`` plus the bundles it indexes.

``build`` writes a benchmark's data directory from a dataset snapshot; ``load``
reads one back and refuses to run on a file whose hash differs from the
index, before anything is spent.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ..bundles import TaskBundle, load_task, sha256_file

BENCH_CONFIG = {"software-delivery": "software_delivery", "cyber-operations": "cyber_operations"}
BUNDLE_FILES = ("manifest.yaml", "repository.zip", "hidden_tests.zip", "gold.diff")
EVALUATOR_IMAGES = ("swe", "sweqna")


def family_of(task: TaskBundle) -> str:
    return "qna" if task.kind == "judge" else "executable"


def _calibration() -> dict:
    from importlib import resources

    return json.loads(resources.files("pdl_bench_eval").joinpath("calibration.json").read_text())["tasks"]


def build(dataset_root: Path, benchmark: str, out_dir: Path, source_revision: str | None = None) -> dict:
    dataset_root, out_dir = Path(dataset_root), Path(out_dir)
    config = BENCH_CONFIG[benchmark]
    frame = pd.read_parquet(dataset_root / f"benchmark_dataset_{config}.parquet")
    calibration = _calibration()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "tasks").mkdir(parents=True)
    rows = []
    for _, row in frame.sort_values("ID").iterrows():
        task = load_task(dataset_root / "tasks" / row["ID"])
        dest = out_dir / "tasks" / task.task_id
        dest.mkdir()
        files = {}
        for name in BUNDLE_FILES:
            src = task.task_dir / name
            if src.exists():
                shutil.copy2(src, dest / name)
                files[name] = sha256_file(src)
        if "gold.diff" not in files:
            # QnA bundles carry their reference patch inside additional_task_files.zip
            import zipfile
            with zipfile.ZipFile(task.task_dir / "additional_task_files.zip") as z:
                member = next((n for n in z.namelist() if n.endswith("gold.diff")), None)
                if member:
                    (dest / "gold.diff").write_bytes(z.read(member))
                    files["gold.diff"] = sha256_file(dest / "gold.diff")
        rows.append({
            "id": task.task_id,
            "benchmark": benchmark,
            "family": family_of(task),
            "task_version": task.manifest.get("task_version"),
            "prompt": task.issue_text,
            "patch_policy": task.patch_policy,
            "manifest_profile": (task.manifest.get("resources") or {}).get("profile"),
            "manifest_scoring_policy": task.evaluation.get("scoring_policy"),
            "image_kind": task.image_kind,
            "calibrated_hidden_total": (calibration.get(task.task_id) or {}).get("hidden_total"),
            "files": files,
        })
    for kind in EVALUATOR_IMAGES:
        src = dataset_root / "environments" / kind / "Dockerfile"
        (out_dir / "environments" / kind).mkdir(parents=True)
        shutil.copy2(src, out_dir / "environments" / kind / "Dockerfile")
    tasks_path = out_dir / "tasks.jsonl"
    tasks_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    manifest = {
        "benchmark": benchmark,
        "tasks": len(rows),
        "families": {f: sum(1 for r in rows if r["family"] == f) for f in ("executable", "qna")},
        "bundle_files": sum(len(r["files"]) for r in rows),
        "snapshot_date": dt.date.today().isoformat(),
        "tasks_sha256": sha256_file(tasks_path),
        "source_repository": "Poindexter-Labs/PDL-Bench",
        "source_revision": source_revision,
        "source_subset": config,
        "protocol_notes": [
            "Executable tasks pin python_standard_1m_v0.1 in their manifests; the published report ran them under python_standard_2m_v0.1 as an operator override, and the preset does the same.",
            "QnA tasks pin scoring_llm_v0.2 in their manifests; the published report graded them under scoring_llm_v0.3 as an operator override, and the preset does the same.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load(data_dir: Path, benchmark: str) -> list:
    """Rows of tasks.jsonl with their verified bundles attached (``row["bundle"]``) and a task hash."""
    data_dir = Path(data_dir).resolve()
    path = data_dir / "tasks.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing; point --data at the package's data directory")
    rows, ids = [], set()
    for line in path.read_text().split("\n"):
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("benchmark") != benchmark:
            raise ValueError(f"tasks.jsonl is for benchmark {row.get('benchmark')!r}, the preset is {benchmark!r}")
        if not isinstance(row.get("id"), str) or row["id"] in ids:
            raise ValueError("each task needs a unique id")
        ids.add(row["id"])
        task_dir = data_dir / "tasks" / row["id"]
        for name, expected in row["files"].items():
            actual = sha256_file(task_dir / name) if (task_dir / name).is_file() else None
            if actual != expected:
                raise ValueError(f"bundle file hash mismatch: {row['id']}/{name}")
        bundle = load_task(task_dir)
        if bundle.task_id != row["id"] or family_of(bundle) != row["family"]:
            raise ValueError(f"bundle for {row['id']} does not match its index row")
        declared = (bundle.manifest.get("repository") or {}).get("sha256")
        if declared and declared != row["files"].get("repository.zip"):
            raise ValueError(f"{row['id']}: repository.zip does not match the manifest's sha256")
        row["task_hash"] = hashlib.sha256(json.dumps({k: v for k, v in row.items() if k != "bundle"},
                                                     sort_keys=True).encode()).hexdigest()
        row["bundle"] = bundle
        rows.append(row)
    if not rows:
        raise ValueError("tasks.jsonl is empty")
    return rows
