#!/usr/bin/env python3
"""Upload the SWE replication work to the dataset in two commits. Run only after explicit approval.

Commit 1 (dataset repair): tasks/contributor_task_08/gold.diff with its missing final context line
restored, so the reference patch applies with `git apply` as well as GNU `patch`.

Commit 2 (packages): environments/software-delivery, environments/cyber-operations, the updated
environments/README.md and the Harbor tree under harbor/. The packages are rebuilt by the caller
from the repaired dataset so their pinned revision is commit 1.

  python tools/upload_environments.py --dry-run
  python tools/upload_environments.py --fix-only          # commit 1, prints its revision
  python tools/upload_environments.py --packages-only     # commit 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = Path(__file__).resolve().parents[1]
DATASET = "Poindexter-Labs/PDL-Bench"
FIX_SOURCE = Path("/private/tmp/claude-504/-Users-jacobcrouch/78b806c2-7e4a-4f77-9276-d08bdf6cda79/scratchpad/"
                  "dataset-fixed/tasks/contributor_task_08/gold.diff")
FIX_MESSAGE = "Restore the missing final context line of contributor_task_08/gold.diff"
FIX_DESCRIPTION = """The reference patch's last hunk declares 17 new lines but the published file ended one
whitespace-only context line early (stripped at export), so `git apply` reported
"corrupt patch at line 460" while GNU `patch` applied it. The line is restored; the patch
is otherwise byte-identical, applies cleanly, and scores 93/93 hidden tests."""
MESSAGE = "Add Software Delivery and Cyber Operations replication packages and the Harbor tree"
DESCRIPTION = """environments/software-delivery and environments/cyber-operations: self-contained Docker
replication packages for the 30 repository tasks, in the same shape as the four single-turn packages
(plan / doctor / run / grade / report, presets, settings evidence, attempt records, CHECKSUMS). They run
the reference agent loop (exported file-for-file from the PD RL Environment repository, provenance in
vendor/SOURCE_MANIFEST.json) in Docker sandboxes under the published prompts, tool schema and budgets,
and grade with the reference evaluator pdl-bench-eval. Validated: gold 1.0 and noop 0.0 on all 30 tasks
through the full path, one paid episode per reference route (PUBLICATION-VALIDATION.json, VALIDATION.md).

environments/README.md: index rows and notes for the two packages.

harbor/: the 30 repository tasks as a Harbor dataset (any Harbor-compatible agent harness can run them;
the verifier is the reference grader)."""


def ops_for(local_root: Path, repo_prefix: str) -> list:
    out = []
    for path in sorted(local_root.rglob("*")):
        if path.is_file() and not (set(path.parts) & {"__pycache__", ".pytest_cache"}):
            out.append(CommitOperationAdd(path_in_repo=f"{repo_prefix}/{path.relative_to(local_root).as_posix()}",
                                          path_or_fileobj=str(path)))
    return out


def package_ops(include_harbor: bool) -> list:
    ops = []
    for bench in ("software-delivery", "cyber-operations"):
        ops += ops_for(REPO / "dist" / "environments" / bench, f"environments/{bench}")
    ops.append(CommitOperationAdd(path_in_repo="environments/README.md",
                                  path_or_fileobj=str(REPO / "replication" / "docs" / "environments-README.md")))
    if include_harbor:
        ops += ops_for(REPO / "dist" / "harbor", "harbor")
    return ops


def summarize(ops):
    total = sum(Path(o.path_or_fileobj).stat().st_size for o in ops)
    print(f"{len(ops)} files, {total / 1e6:.1f} MB")
    for prefix in ("tasks/", "environments/software-delivery", "environments/cyber-operations", "environments/README.md", "harbor"):
        n = sum(1 for o in ops if o.path_in_repo.startswith(prefix))
        if n:
            print(f"  {prefix}: {n} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--fix-only", action="store_true")
    ap.add_argument("--packages-only", action="store_true")
    ap.add_argument("--no-harbor", action="store_true")
    args = ap.parse_args()
    api = HfApi()
    fix_ops = [CommitOperationAdd(path_in_repo="tasks/contributor_task_08/gold.diff", path_or_fileobj=str(FIX_SOURCE))]
    pkg_ops = package_ops(not args.no_harbor)
    if args.dry_run:
        print("commit 1:"); summarize(fix_ops)
        print("commit 2:"); summarize(pkg_ops)
        return
    if not args.packages_only:
        info = api.create_commit(repo_id=DATASET, repo_type="dataset", operations=fix_ops,
                                 commit_message=FIX_MESSAGE, commit_description=FIX_DESCRIPTION)
        print("commit 1:", info.oid, info.commit_url)
    if not args.fix_only:
        info = api.create_commit(repo_id=DATASET, repo_type="dataset", operations=pkg_ops,
                                 commit_message=MESSAGE, commit_description=DESCRIPTION)
        print("commit 2:", info.oid, info.commit_url)


if __name__ == "__main__":
    main()
