"""Locate, fetch and unpack PDL-Bench task bundles from the Hugging Face dataset."""
from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path

import yaml

DATASET = "Poindexter-Labs/PDL-Bench"
TASK_FILES = ("manifest.yaml", "repository.zip", "hidden_tests.zip")


def fetch(revision: str | None = None, cache_dir: str | None = None, patterns=None) -> Path:
    """Download (or reuse from cache) the dataset files the graders need; returns the snapshot root."""
    from huggingface_hub import snapshot_download

    patterns = list(patterns or ["tasks/*", "environments/*", "benchmark_dataset_*.parquet", "README.md"])
    return Path(snapshot_download(DATASET, repo_type="dataset", revision=revision,
                                  cache_dir=cache_dir, allow_patterns=patterns))


@dataclass
class TaskBundle:
    task_dir: Path
    manifest: dict

    @property
    def task_id(self) -> str:
        return self.manifest["task_id"]

    @property
    def kind(self) -> str:
        """'judge' for rubric-scored (SWEQnA) tasks, 'tests' for hidden-suite tasks."""
        pin = str((self.manifest.get("evaluation") or {}).get("scoring_policy") or "")
        return "judge" if pin.startswith("scoring_llm") else "tests"

    @property
    def image_kind(self) -> str:
        image = str((self.manifest.get("environment") or {}).get("prebuilt_image") or "")
        return "sweqna" if "sweqna" in image else "swe"

    @property
    def issue_text(self) -> str:
        return (self.manifest.get("issue") or {}).get("text", "")

    @property
    def evaluation(self) -> dict:
        return self.manifest.get("evaluation") or {}

    @property
    def patch_policy(self) -> dict:
        return self.manifest.get("patch_policy") or {}

    @property
    def setup_command(self) -> str | None:
        return (self.manifest.get("environment") or {}).get("setup_command")


def load_task(task_dir: Path) -> TaskBundle:
    task_dir = Path(task_dir)
    for name in TASK_FILES:
        if not (task_dir / name).exists():
            raise FileNotFoundError(f"{task_dir}: missing {name}")
    manifest = yaml.safe_load((task_dir / "manifest.yaml").read_text())
    return TaskBundle(task_dir=task_dir, manifest=manifest)


def find_task(root: Path, task_id: str) -> TaskBundle:
    return load_task(Path(root) / "tasks" / task_id)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def unpack(task: TaskBundle, work_dir: Path, verify: bool = True) -> tuple:
    """Extract repository.zip and hidden_tests.zip; returns (repo_dir, hidden_dir)."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    expected = (task.manifest.get("repository") or {}).get("sha256")
    if verify and expected:
        actual = sha256_file(task.task_dir / "repository.zip")
        if actual != expected:
            raise ValueError(f"{task.task_id}: repository.zip sha256 {actual} != manifest {expected}")
    with zipfile.ZipFile(task.task_dir / "repository.zip") as z:
        z.extractall(work_dir)
    hidden_root = work_dir / "hidden"
    hidden_root.mkdir(exist_ok=True)
    with zipfile.ZipFile(task.task_dir / "hidden_tests.zip") as z:
        z.extractall(hidden_root)
    repo_dir = work_dir / "repository"
    hidden_dir = hidden_root / "hidden_tests"
    if not repo_dir.is_dir() or not hidden_dir.is_dir():
        raise ValueError(f"{task.task_id}: unexpected archive layout under {work_dir}")
    return repo_dir, hidden_dir


def overall_rows(root: Path):
    """The ``overall`` config as a DataFrame (ID, bench, grading_type, PROMPT, ...)."""
    import pandas as pd

    return pd.read_parquet(Path(root) / "benchmark_dataset_overall.parquet")


def qa_row(root: Path, task_id: str) -> dict:
    df = overall_rows(root)
    hit = df[df["ID"] == task_id]
    if hit.empty:
        raise KeyError(f"{task_id} is not in the overall config")
    return hit.iloc[0].to_dict()
