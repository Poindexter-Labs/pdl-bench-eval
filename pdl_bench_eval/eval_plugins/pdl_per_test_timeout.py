"""Per-test time limit for PD-RL evaluator runs (P1-049).

Loaded by the evaluator with ``-p pdl_per_test_timeout`` when the run profile
sets ``timing.per_test_timeout_seconds``; the runner stages this file beside
the hidden bundle, so it is never in the agent's workspace. Standard library
plus pytest only: it runs in the evaluator image, where pd_rl is not installed.

Without it, one test that never returns takes the whole suite down with it:
the evaluator's command timeout kills pytest before it prints a summary, the
runner sees zero tests collected, and the episode voids — hiding a model
failure as an evaluator one. Measured on cyber task 05: 12 episodes looped
forever in their own index allocator on hidden test 39 of 60 and voided.

The expiry raises through ``pytest.fail``, whose exception derives from
BaseException, so agent code that catches ``Exception`` cannot swallow it.
The same mechanism as pytest-timeout's signal method, which no evaluator image
ships. Limits: needs SIGALRM (POSIX, main thread); a test blocked inside a
single C call is only stopped when that call returns, and code that catches
BaseException can still swallow it — the evaluator's command timeout remains
the backstop for both.
"""

import os
import signal

import pytest

LIMIT_S = float(os.environ.get("PDL_PER_TEST_TIMEOUT_S") or 0)


def _expired(signum, frame):
    pytest.fail(
        f"PD-RL per-test time limit: this test ran longer than {LIMIT_S:g}s "
        "and was stopped (P1-049)",
        pytrace=False,
    )


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    if LIMIT_S <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return
    previous = signal.signal(signal.SIGALRM, _expired)
    signal.setitimer(signal.ITIMER_REAL, LIMIT_S)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
