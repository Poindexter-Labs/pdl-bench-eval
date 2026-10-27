"""Per-test time limit for the hidden and public suites (60 s in the published profiles).

The reference evaluator loads a small pytest plugin through PYTEST_ADDOPTS from
a staged copy of the hidden bundle, so one test that never returns fails alone
instead of taking the whole suite (and the episode) down with it.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from .docker_env import HIDDEN_MOUNT

PLUGIN = "pdl_per_test_timeout"
PLUGIN_SRC = Path(__file__).resolve().parent / "eval_plugins" / f"{PLUGIN}.py"
STAGED_SUBDIR = "_pdl_eval"
DEFAULT_LIMIT_S = 60


def stage(hidden_dir: Path, base_dir: Path | None = None) -> Path:
    """A temporary copy of the hidden bundle with the plugin added; caller removes its parent.

    ``base_dir`` keeps the copy beside the work tree, which matters when the evaluator runs inside
    a controller container: the copy must sit under a bind mount the host daemon can see.
    """
    staged = Path(tempfile.mkdtemp(prefix="pdl-eval-hidden-", dir=str(base_dir) if base_dir else None)) / "hidden_tests"
    shutil.copytree(hidden_dir, staged, symlinks=True)
    (staged / STAGED_SUBDIR).mkdir()
    shutil.copy2(PLUGIN_SRC, staged / STAGED_SUBDIR / PLUGIN_SRC.name)
    return staged


def env(limit_s, base_env: dict) -> dict:
    plugin_dir = f"{HIDDEN_MOUNT}/{STAGED_SUBDIR}"
    pythonpath = base_env.get("PYTHONPATH")
    addopts = base_env.get("PYTEST_ADDOPTS")
    return {"PYTHONPATH": f"{plugin_dir}:{pythonpath}" if pythonpath else plugin_dir,
            "PYTEST_ADDOPTS": f"{addopts} -p {PLUGIN}" if addopts else f"-p {PLUGIN}",
            "PDL_PER_TEST_TIMEOUT_S": str(limit_s)}
