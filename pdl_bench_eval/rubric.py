"""Rubric parsing: numbered ("1. ...") or R-id ("R1 ...") criteria with optional weights.

A port of the parser behind every published SWEQnA number. One scheme per
rubric; a Markdown heading ends the criterion before it; weights come from
"(2 points)" after an R-id or a "Weight: 2" line under a numbered item.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_RID = re.compile(
    r"""^[ \t]*
        (?:[-*][ \t]*)?          # optional list bullet
        (?:\#+[ \t]*)?           # optional heading marker
        (?:\*\*)?                # optional bold
        (?P<id>R\d+[a-z]?)       # the id, optionally suffixed R001a/b/c
        (?:\*\*)?
        [ \t]*
        (?:                      # ... optionally followed by a weight
            \(\s*(?P<w1>[\d.]+)\s*(?:points?)?\s*\)   # (2)  (3 points)
          | [-—][ \t]*(?P<w2>[\d.]+)[ \t]*points?     # - 2 points
        )?
    """,
    re.M | re.X,
)
_NUM = re.compile(r"^[ \t]*(?P<id>\d+)\.[ \t]+(?=\S)", re.M)
_WEIGHT_LINE = re.compile(
    r"^[ \t]*\**[ \t]*Weight\**[ \t]*[:：][ \t]*\**[ \t]*(?P<w>\d+(?:\.\d+)?)[ \t]*(?:points?)?[ \t]*\**[ \t]*$",
    re.M | re.I,
)
_HEADING = re.compile(r"^[ \t]*#{1,6}[ \t]+\S", re.M)
_DECLARED = re.compile(
    r"divided by\s+(\d+)"
    r"|Criteria\s*\(\s*(\d+)\s*\)"
    r"|are\s+(\d+)\s+binary criteria"
    r"|(\d+)\s+binary criteria"
    r"|meeting all\s+(\d+)",
    re.I,
)
_EXPLAINS = re.compile(
    r"\bexplain\w*|\bjustif\w*|\bwhy\b|\brather than\b|\bargues?\b"
    r"|\bbelongs? (?:to|in)\b|\bcounterfactual\b",
    re.I,
)
_NAMES_VALUE = re.compile(r"`[^`]+`|\b[0-9a-f]{6,}\b|\b\d+\b|\b[A-Z][A-Z0-9_]{3,}\b")

VALUE, COMPOUND, JUDGMENT = "value", "compound", "judgment"


@dataclass
class Criterion:
    id: str
    text: str
    weight: float | None = None
    kind: str = JUDGMENT


@dataclass
class Rubric:
    criteria: list = field(default_factory=list)
    style: str = ""
    declared_total: int | None = None
    problems: list = field(default_factory=list)


def _classify(text: str) -> str:
    has_value, has_explain = bool(_NAMES_VALUE.search(text)), bool(_EXPLAINS.search(text))
    if has_value and has_explain:
        return COMPOUND
    return VALUE if has_value else JUDGMENT


def parse_rubric(text: str) -> Rubric:
    """Structured criteria, plus every disagreement found while parsing."""
    rid = list(_RID.finditer(text))
    num = list(_NUM.finditer(text))
    problems = []
    if rid and num:
        problems.append(f"mixed numbering: {len(rid)} R-ids and {len(num)} numbered items — one scheme per rubric")
    marks, style = (rid, "r-id") if rid else (num, "numbered")
    if not marks:
        return Rubric(problems=["no criteria found"], style="unknown")
    criteria = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        heading = _HEADING.search(text, m.end(), end)
        if heading:
            end = heading.start()
        raw_body = text[m.end():end]
        weight = None
        if style == "r-id":
            raw = m.group("w1") or m.group("w2")
            weight = float(raw) if raw else None
        else:
            wm = _WEIGHT_LINE.search(raw_body)
            if wm:
                weight = float(wm.group("w"))
                raw_body = raw_body[:wm.start()] + raw_body[wm.end():]
        body = " ".join(raw_body.split()).lstrip(" :.-—)*")
        criteria.append(Criterion(id=m.group("id"), text=body, weight=weight, kind=_classify(body)))
    seen, order = set(), []
    for c in criteria:
        n = int(re.sub(r"\D", "", c.id))
        if n not in seen:
            seen.add(n)
            order.append(n)
    if order != list(range(1, len(order) + 1)):
        problems.append(f"criterion numbers are not contiguous 1..{len(order)}: {order}")
    declared = {int(g) for m in _DECLARED.finditer(text) for g in m.groups() if g}
    total = next((d for d in sorted(declared) if d == len(criteria)), None)
    if declared and total is None:
        problems.append(f"declared {sorted(declared)} but parsed {len(criteria)} criteria (a points total is not a criterion count)")
    return Rubric(criteria=criteria, style=style, declared_total=total, problems=problems)
