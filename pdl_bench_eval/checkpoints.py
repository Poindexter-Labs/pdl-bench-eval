"""Rubric criteria as binary checkpoints, and the weighted-mean aggregate.

Every public SWEQnA task ships ``hidden_tests/rubric.md``; its criteria become
equal-weight LLM checkpoints unless a criterion states a weight. The reward is
the weighted mean of per-item scores, where an item's score is the median of
the panel members' binary votes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .rubric import parse_rubric

TIERS = ("critical", "important", "sanity")
_PERCENT = re.compile(r"\((\d+(?:\.\d+)?)\s*%\)")
_POINTS = re.compile(r"\((\d+(?:\.\d+)?)\s*(?:points?|pts?)\)", re.IGNORECASE)


class CheckpointError(ValueError):
    """The task cannot be scored under a checkpoint policy as authored."""


@dataclass
class Checkpoint:
    id: str
    method: str = "llm"
    tier: str = "important"
    weight: float = 1.0
    question: str = ""
    pass_anchor: str = ""
    files: list = field(default_factory=list)

    @property
    def is_gate(self) -> bool:
        return self.method == "rule" and self.tier == "sanity"


def derive_from_rubric(text: str) -> list:
    """Numbered/R-id rubric criteria -> llm checkpoints, equal weight unless stated."""
    rubric = parse_rubric(text)
    if not rubric.criteria:
        raise CheckpointError("rubric.md has no parseable criteria: " + "; ".join(rubric.problems))
    items = []
    for c in rubric.criteria:
        body, weight = c.text, c.weight
        if weight is None:
            m = _PERCENT.search(body) or _POINTS.search(body)
            if m:
                weight = float(m.group(1))
                body = (body[:m.start()] + body[m.end():]).strip()
        body = re.sub(r"\s+", " ", body).strip(" :.-—*")
        items.append(Checkpoint(id=str(c.id), question=body, weight=weight if weight else 1.0))
    return items


def validate_checkpoints(items) -> list:
    problems, seen = [], set()
    for it in items:
        where = f"checkpoint {it.id!r}"
        if not it.id:
            problems.append("checkpoint with an empty id")
        if it.id in seen:
            problems.append(f"{where}: duplicate id")
        seen.add(it.id)
        if it.method not in ("llm", "rule"):
            problems.append(f"{where}: unknown method {it.method!r}")
        if it.tier not in TIERS:
            problems.append(f"{where}: unknown tier {it.tier!r}")
        if not (it.weight > 0):
            problems.append(f"{where}: weight must be > 0")
        if it.method == "llm" and not it.question:
            problems.append(f"{where}: llm item has no question")
        if it.method == "llm" and it.tier == "sanity":
            problems.append(f"{where}: sanity (gate) tier is for rule items only")
    return problems


def load_checkpoints(hidden_dir: Path) -> tuple:
    """(items, source) for a task's hidden bundle; raises CheckpointError."""
    rubric_path = Path(hidden_dir) / "rubric.md"
    if not rubric_path.exists():
        raise CheckpointError(f"no rubric.md in {hidden_dir}")
    items = derive_from_rubric(rubric_path.read_text(encoding="utf-8"))
    problems = validate_checkpoints(items)
    if problems:
        raise CheckpointError("rubric.md: " + "; ".join(problems))
    return items, "rubric.md"


def render_for_judge(items) -> str:
    """The checkpoint block of the judge prompt: llm items only, ids verbatim."""
    lines = []
    for it in items:
        if it.method != "llm":
            continue
        head = f"[{it.id}] ({it.tier}, weight {it.weight:g})"
        if it.files:
            head += f" — judge from: {', '.join(it.files)}"
        lines.append(head)
        lines.append(f"  question: {it.question}")
        if it.pass_anchor:
            lines.append(f"  ground truth (the worker never saw this): {it.pass_anchor}")
    return "\n".join(lines)


def weighted_score(items, item_scores: dict) -> tuple:
    """(score, gate_failed): weighted mean over non-gate items; a failed gate zeroes it."""
    missing = [it.id for it in items if it.id not in item_scores]
    if missing:
        raise CheckpointError(f"no score for checkpoint(s) {missing}")
    gate_failed = [it.id for it in items if it.is_gate and item_scores[it.id] < 0.999]
    scored = [it for it in items if not it.is_gate]
    total = sum(it.weight for it in scored)
    value = sum(it.weight * float(item_scores[it.id]) for it in scored) / total if total else 0.0
    if gate_failed:
        return 0.0, gate_failed
    return round(value, 6), []
