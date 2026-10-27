"""Replication runner: run the reference agent on the repository tasks and grade the result.

Mirrors the non-SWE replication packages' contract (``plan`` / ``doctor`` /
``run`` / ``grade`` / ``report``, presets as the authority, attempt records keyed
by task, configuration, attempt and implementation, technical failures never
counted as model failures) on top of the agent loop vendored from the PD RL
Environment repository (``pdl_bench_eval._vendor``) and this package's
validated graders.
"""
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
SANDBOX_ROOT = PACKAGE_DIR          # sandbox/<kind>/Dockerfile lives here
PRESET_DIR = PACKAGE_DIR / "presets"
