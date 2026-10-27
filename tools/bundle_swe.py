#!/usr/bin/env python3
"""Write the two SWE replication packages (environments/software-delivery, environments/cyber-operations).

Mirrors pointdexter's tools/replication/bundle.py: each folder is self-contained
(controller code vendored, data and bundles copied, documents rendered,
CHECKSUMS.json over every file) and never uploaded by this tool.

Usage:
  python tools/bundle_swe.py --dataset-dir .dataset --out dist/environments [--revision <dataset commit>]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / "pdl_bench_eval"
SRC = REPO / "replication"
BENCHES = ("software-delivery", "cyber-operations")
VENDOR_SKIP = {"__pycache__", ".pytest_cache"}
VENDOR_SUFFIXES = {".py", ".yaml", ".json", ""}     # "" = Dockerfile


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_package(dest: Path) -> dict:
    """pdl_bench_eval as an importable copy under vendor/; returns {relative path: sha256}."""
    files = {}
    for path in sorted(PKG.rglob("*")):
        if not path.is_file() or VENDOR_SKIP & set(path.parts) or path.suffix not in VENDOR_SUFFIXES:
            continue
        if path.suffix == "" and path.name != "Dockerfile":
            continue
        rel = path.relative_to(PKG)
        target = dest / "vendor" / "pdl_bench_eval" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        files[str(Path("pdl_bench_eval") / rel)] = digest_file(path)
    return files


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 - not a git repository yet
        return None


def render(template: Path, values: dict) -> str:
    text = template.read_text()
    for key, value in values.items():
        text = text.replace(f"@@{key}@@", str(value))
    leftover = [line for line in text.splitlines() if "@@" in line]
    if leftover:
        raise SystemExit(f"{template.name}: unrendered placeholder in: {leftover[0]!r}")
    return text


def build_one(bench: str, dataset_dir: Path, out_root: Path, revision: str | None, validation: dict | None) -> dict:
    from pdl_bench_eval.replication import data as data_mod

    dest = out_root / bench
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    # data
    manifest = data_mod.build(dataset_dir, bench, dest / "data", source_revision=revision)
    # code
    vendored = copy_package(dest)
    shutil.copy2(SRC / "bench.py", dest / "bench.py")
    shutil.copy2(SRC / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(SRC / "requirements.lock", dest / "requirements.lock")
    shutil.copy2(SRC / ".env.example", dest / ".env.example")
    (dest / ".dockerignore").write_text("**\n!Dockerfile\n!requirements.lock\n!bench.py\n!vendor/**\n!presets/**\n"
                                        "!settings-evidence/**\n!tests/**\n**/__pycache__/**\n**/*.pyc\n**/.env*\n")
    (dest / ".gitignore").write_text(".env\n.env.*\n!.env.example\nresults/\ndist/\n__pycache__/\n*.pyc\n")
    for sub in ("presets", "settings-evidence"):
        (dest / sub).mkdir()
    shutil.copy2(PKG / "replication" / "presets" / f"{bench}.json", dest / "presets" / f"{bench}.json")
    shutil.copy2(SRC / "settings-evidence" / f"{bench}.json", dest / "settings-evidence" / f"{bench}.json")
    for kind in ("swe", "sweqna"):
        (dest / "sandbox" / kind).mkdir(parents=True)
        shutil.copy2(PKG / "replication" / "sandbox" / kind / "Dockerfile", dest / "sandbox" / kind / "Dockerfile")
    shutil.copytree(SRC / "tests", dest / "tests", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
    (dest / "licenses").mkdir()
    shutil.copy2(REPO / "LICENSE", dest / "licenses" / "pdl-bench-eval-LICENSE")
    # provenance of the vendored code
    rl_manifest = json.loads((PKG / "_vendor" / "SOURCE_MANIFEST.json").read_text())
    (dest / "vendor" / "SOURCE_MANIFEST.json").write_text(json.dumps({
        "pdl_bench_eval": {"source_repository": "Poindexter-Labs/pdl-bench-eval", "source_commit": git_commit(),
                           "files": vendored},
        "pd_rl_agent_loop": {"source_repository": rl_manifest["source_repository"],
                             "source_commit": rl_manifest["source_commit"],
                             "exported_files": rl_manifest["files"], "adaptations": rl_manifest["adaptations"],
                             "detail": "vendor/pdl_bench_eval/_vendor/SOURCE_MANIFEST.json"},
    }, indent=2) + "\n")
    # documents
    preset = json.loads((PKG / "replication" / "presets" / f"{bench}.json").read_text())
    families = manifest["families"]
    values = {
        "BENCH": bench,
        "TITLE": {"software-delivery": "Software Delivery", "cyber-operations": "Cyber Operations"}[bench],
        "TASKS": manifest["tasks"],
        "EXECUTABLE": families.get("executable", 0),
        "QNA": families.get("qna", 0),
        "REVISION": revision or "(unpinned local snapshot)",
        "SNAPSHOT_DATE": manifest["snapshot_date"],
        "MODELS": ", ".join(m["id"] for m in preset["models"]),
        "QNA_SECTION": "",      # rendered below, after the scalar values are known
        "QNA_START": ("Software Delivery includes 10 question-answering tasks judged by a three-member panel; "
                      "set the judge keys too, or name another panel with PDL_JUDGES.") if families.get("qna") else "",
        "EXEC_JOBS": ", ".join(preset["selection_evidence"]["executable"]["jobs"]),
        "QNA_JOBS": ", ".join(preset["selection_evidence"].get("qna", {}).get("jobs", [])) or "n/a",
        "DATE": dt.date.today().isoformat(),
    }
    if families.get("qna"):
        values["QNA_SECTION"] = render(SRC / "docs" / "README.qna-section.md", values)
    for name in ("README.md", "START_HERE.md", "DATASET-NOTES.md", "THIRD_PARTY.md"):
        (dest / name).write_text(render(SRC / "docs" / name, values))
    if (SRC / "docs" / "VALIDATION.md").exists():
        (dest / "VALIDATION.md").write_text(render(SRC / "docs" / "VALIDATION.md", values))
    if validation:
        (dest / "PUBLICATION-VALIDATION.json").write_text(json.dumps({**validation, "benchmark": bench,
                                                                     "tasks": manifest["tasks"]}, indent=2) + "\n")
    # checksums over everything (except the checksum file itself)
    sums = {str(p.relative_to(dest)): digest_file(p) for p in sorted(dest.rglob("*"))
            if p.is_file() and p.name != "CHECKSUMS.json"}
    (dest / "CHECKSUMS.json").write_text(json.dumps(sums, indent=2) + "\n")
    return {"benchmark": bench, "tasks": manifest["tasks"], "files": len(sums),
            "bytes": sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--revision")
    ap.add_argument("--validation", help="JSON file with the publication validation record to include")
    ap.add_argument("--bench", action="append", choices=BENCHES)
    args = ap.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    revision = args.revision
    if revision is None and "snapshots" in dataset_dir.parts:
        revision = dataset_dir.parts[dataset_dir.parts.index("snapshots") + 1]
    validation = json.loads(Path(args.validation).read_text()) if args.validation else None
    out = Path(args.out)
    results = [build_one(b, dataset_dir, out, revision, validation) for b in (args.bench or BENCHES)]
    (out / "RELEASES.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
