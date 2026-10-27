"""Patch handling: which paths a diff touches, the patch policy, and applying it."""
from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

_HEADER_RE = re.compile(r'^(?:\+\+\+|---)\s+(?:"([^"]+)"|(\S+))')
# a/ b/ are git's defaults; i/ w/ c/ o/ appear under diff.mnemonicPrefix
_PREFIX_RE = re.compile(r"^[abciow]/")
_DIFF_GIT_RE = re.compile(r"^diff --git (?:[abciow]/)?(\S+) (?:[abciow]/)?(\S+)$")


def patch_paths(diff_text: str) -> set:
    """Paths a diff touches, from its ---/+++ headers, with ``diff --git`` as fallback.

    Raises ValueError when the diff contains hunks but yields no paths: the
    patch-policy check must never pass vacuously.
    """
    paths = set()
    for line in diff_text.splitlines():
        m = _HEADER_RE.match(line)
        if not m:
            continue
        path = m.group(1) or m.group(2)
        if path == "/dev/null":
            continue
        paths.add(_PREFIX_RE.sub("", path, 1))
    if not paths:
        for line in diff_text.splitlines():
            m = _DIFF_GIT_RE.match(line)
            if m:
                paths.update(m.groups())
    if not paths and re.search(r"^@@ ", diff_text, flags=re.M):
        raise ValueError("diff contains hunks but no parseable file paths — the patch-policy check would be vacuous")
    return paths


def policy_violations(paths, policy: dict) -> list:
    """Paths outside ``allowed_paths`` or inside ``forbidden_paths`` (fnmatch globs)."""
    policy = policy or {}
    violations = []
    for path in sorted(paths):
        allowed = any(fnmatch.fnmatch(path, p) for p in policy.get("allowed_paths", ["*"]))
        forbidden = any(fnmatch.fnmatch(path, p) for p in policy.get("forbidden_paths", []))
        if not allowed or forbidden:
            violations.append(path)
    return violations


def apply_patch(repo_dir: Path, patch_file: Path) -> tuple:
    """Apply a unified diff to ``repo_dir``; (ok, output). git apply first, patch -p1 as fallback."""
    repo_dir, patch_file = Path(repo_dir), Path(patch_file).resolve()
    g = subprocess.run(["git", "apply", "--whitespace=nowarn", str(patch_file)],
                       cwd=repo_dir, capture_output=True, text=True)
    if g.returncode == 0:
        return True, "git apply"
    p = subprocess.run(["patch", "-p1", "--forward", "--batch", "-i", str(patch_file)],
                       cwd=repo_dir, capture_output=True, text=True)
    if p.returncode == 0:
        return True, "patch -p1 (git apply failed: " + g.stderr.strip()[-300:] + ")"
    return False, (g.stderr.strip() + "\n" + p.stdout.strip() + p.stderr.strip())[-1500:]
