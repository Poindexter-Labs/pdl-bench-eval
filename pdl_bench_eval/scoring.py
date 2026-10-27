"""Scoring policies: weighted buckets with absent-bucket renormalisation.

``scoring_v0.2`` (the policy behind every published repository-task number)
multiplies the renormalised weighted sum by the hidden-test fraction, so an
episode that fixes nothing scores exactly 0 however tidy it left the tree.
``scoring_v0.1`` is the plain weighted sum (kept for comparison).
``scoring_llm_v0.3`` has one bucket, the judge-checkpoint aggregate.

A published "pass" is a reward STRICTLY greater than ``PASS_THRESHOLD``.
"""
from __future__ import annotations

from importlib import resources

import yaml

PASS_THRESHOLD = 0.95


def load_policy(policy_id: str) -> dict:
    text = resources.files("pdl_bench_eval").joinpath("policies", f"{policy_id}.yaml").read_text()
    policy = yaml.safe_load(text)
    if policy.get("scoring_policy_id") != policy_id:
        raise ValueError(f"policy file for {policy_id} declares {policy.get('scoring_policy_id')!r}")
    return policy


def score(policy: dict, buckets: dict) -> float | None:
    """Reward for the bucket sub-scores under ``policy``; None if no bucket is present."""
    weights = policy["weights"]
    present = {k: v for k, v in buckets.items() if v is not None}
    if not present:
        return None
    unweighted = sorted(set(present) - set(weights))
    if unweighted:
        raise ValueError(
            f"{policy['scoring_policy_id']} does not weight {unweighted} — the policy and the "
            f"task are mismatched (policy buckets: {sorted(weights)})"
        )
    total_weight = sum(weights[k] for k in present)
    weighted = sum(weights[k] * present[k] / total_weight for k in present)
    aggregation = policy.get("aggregation") or {}
    if aggregation.get("mode") != "progress_scaled":
        return round(weighted, 6)
    progress_key = aggregation["progress_bucket"]
    if progress_key not in present:
        raise ValueError(
            f"{policy['scoring_policy_id']} scales by '{progress_key}', which this task does not "
            f"produce (buckets: {sorted(present)})"
        )
    progress = present[progress_key]
    inner = sum(
        weights[k] * (1.0 if k == progress_key else present[k]) / total_weight for k in present
    )
    return round(progress * inner, 6)


def passed(reward: float | None, threshold: float = PASS_THRESHOLD) -> bool:
    return reward is not None and reward > threshold
