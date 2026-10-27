"""Test counts from pytest's output, exactly as the reference evaluator reads them.

Sums every "N passed" / "N failed" / "N error" token in the output. "error"
deliberately matches both the summary line and "Interrupted: N errors during
collection", so a broken import can count as MORE than the suite's size — the
grader compares the collected total against the calibrated total for that reason.
"""
import re


def pytest_counts(output: str) -> dict:
    passed = sum(int(n) for n in re.findall(r"(\d+) passed", output or ""))
    failed = sum(int(n) for n in re.findall(r"(\d+) failed", output or ""))
    failed += sum(int(n) for n in re.findall(r"(\d+) error", output or ""))
    return {"passed": passed, "failed": failed}
