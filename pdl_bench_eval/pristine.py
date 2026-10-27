"""Pristine evaluated tree: bytecode purge and planted-interpreter-hook detection.

CPython trusts a .pyc whose header matches the source's mtime and size, so a
cache compiled from different source can hand the hidden suite different code
than the patch shows. The reference evaluator purges every cache from the
graded tree before any evaluator command runs, and refuses a tree that carries
a hook-named file (conftest.py, sitecustomize.py, *.pth, ...) the fixture did
not ship.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

HOOK_BASENAMES = frozenset({
    "sitecustomize.py", "usercustomize.py", "conftest.py", "pytest.ini", "tox.ini",
    "setup.cfg", "pyproject.toml", ".coveragerc", ".pytest.ini",
})
HOOK_SUFFIXES = frozenset({".pth"})
BYTECODE_DIRS = frozenset({"__pycache__"})
BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})
SKIP_DIRS = frozenset({".git"})
EVALUATOR_ENV_HARDENING = {"PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1"}


def is_hook_path(rel_path: str) -> bool:
    name = rel_path.rsplit("/", 1)[-1]
    return name in HOOK_BASENAMES or any(name.endswith(s) for s in HOOK_SUFFIXES)


def _walk(tree: Path):
    for dirpath, dirnames, filenames in os.walk(tree, followlinks=False):
        dirnames[:] = [d for d in dirnames
                       if d not in SKIP_DIRS and not os.path.islink(os.path.join(dirpath, d))]
        yield dirpath, dirnames, filenames


def fixture_index(fixture_dir: Path) -> frozenset:
    """Relative POSIX paths of every regular file the fixture ships."""
    paths = set()
    fixture_dir = Path(fixture_dir)
    for dirpath, _dirnames, filenames in _walk(fixture_dir):
        for name in filenames:
            if name == ".DS_Store":
                continue
            full = Path(dirpath) / name
            if full.is_symlink() or not full.is_file():
                continue
            paths.add(full.relative_to(fixture_dir).as_posix())
    return frozenset(paths)


def purge_bytecode(tree: Path) -> dict:
    tree = Path(tree)
    removed = {"dirs": 0, "files": 0}
    for dirpath, dirnames, filenames in _walk(tree):
        for d in list(dirnames):
            if d in BYTECODE_DIRS:
                shutil.rmtree(Path(dirpath) / d, ignore_errors=True)
                dirnames.remove(d)
                removed["dirs"] += 1
        for name in filenames:
            if any(name.endswith(s) for s in BYTECODE_SUFFIXES):
                try:
                    (Path(dirpath) / name).unlink()
                    removed["files"] += 1
                except OSError:
                    pass
    return removed


def interpreter_hooks(tree: Path, fixture_paths) -> list:
    """Hook-named files in ``tree`` that the fixture did not ship, sorted."""
    tree = Path(tree)
    fixture_paths = frozenset(fixture_paths or ())
    found = []
    if not tree.is_dir():
        return found
    for dirpath, _dirnames, filenames in _walk(tree):
        for name in filenames:
            rel = (Path(dirpath) / name).relative_to(tree).as_posix()
            if is_hook_path(rel) and rel not in fixture_paths:
                found.append(rel)
    return sorted(found)
