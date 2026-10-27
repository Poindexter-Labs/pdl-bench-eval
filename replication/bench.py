#!/usr/bin/env python3
"""Poindexter SWE replication CLI: plan | doctor | run | grade | report. Planning never calls a provider.

The verbs, attempt records and report follow the other PDL-Bench replication
packages; the runner is ``pdl_bench_eval.replication`` (this folder's
``vendor/pdl_bench_eval`` is the exact copy named in SOURCE_MANIFEST.json).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "vendor"))

from pdl_bench_eval.replication.cli import main  # noqa: E402

if __name__ == "__main__":
    main()
